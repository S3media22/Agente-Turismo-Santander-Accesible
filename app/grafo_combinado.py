"""Grafo combinado guía + web (LangGraph) — el patrón router → nodos → supervisor.

Curso "Agentes con LangGraph". Adaptado al proyecto: en vez de *web + científico (ArXiv)*,
combinamos las dos fuentes que SÍ tienen sentido para turismo:

  - **guía** (PDF, lo ESTÁTICO de Santander) vía el RAG.
  - **web** (Tavily, lo ACTUAL: eventos, clima, precios) vía la herramienta ``busca_web``.

Flujo (barato y determinista):

    START → router → { guia | web | ambas } → supervisor → END

**Diseño de bajo costo** (la razón de usar grafo en vez de un ReAct libre):
  - `router`: **0** llamadas si hay señales claras de info actual (atajo por palabras clave);
    si no, **1** llamada de clasificación con salida estructurada (Literal).
  - `guia`: 1 llamada (RAG); `web`: 1 llamada (Tavily, 0 LLM, + 1 síntesis en ESPAÑOL,
    porque Tavily suele responder en inglés).
  - `supervisor`: **0** llamadas (combina en código, como el `supervisor_node` del curso).

Es una vía PARALELA: se expone en ``/grafo/combinado`` sin alterar los demás endpoints.
"""
from __future__ import annotations

import re
from typing import Literal, TypedDict

from . import llm

# Señales fuertes de información ACTUAL (que la guía estática no puede tener). Si aparecen,
# enrutamos directo a la web SIN gastar la llamada del router (más barato y más fiable que
# depender de que el LLM clasifique bien; el router falló mandando "eventos" a la guía).
_SENAL_WEB = re.compile(
    r"\b(evento|eventos|feria|ferias|fiesta|fiestas|festival|festivales|concierto|conciertos|"
    r"agenda|cartelera|hoy|mañana|manana|ahora|actual|actuales|reciente|recientes|proxim\w*|"
    r"clima|temperatura|llueve|lluvia|pron[oó]stico|"
    r"precio|precios|cu[aá]nto cuesta|tarifa|tarifas|costo|202\d)\b",
    re.IGNORECASE,
)

SYSTEM_ROUTER = (
    "Eres un enrutador de un asistente turístico de Santander, Colombia. Según la pregunta, "
    "elige la fuente más adecuada:\n"
    "- 'guia': información ESTÁTICA de la guía (destinos, gastronomía, rutas, deportes, mejor "
    "época). Ej.: '¿dónde hago rafting?', '¿qué comer?'.\n"
    "- 'web': información ACTUAL o en tiempo real (eventos/ferias próximas, clima de hoy, "
    "precios vigentes, noticias). Ej.: '¿qué eventos hay este fin de semana?', '¿clima hoy?'.\n"
    "- 'ambas': cuando conviene combinar lo estático con lo actual. Ej.: '¿qué hago este "
    "sábado en San Gil?' (actividades de la guía + eventos/clima de hoy)."
)


class EstadoTurismo(TypedDict, total=False):
    pregunta: str
    ruta: str  # "guia" | "web" | "ambas"
    respuesta_guia: str
    respuesta_web: str
    respuesta_final: str


# --------------------------- Fuentes (helpers) --------------------------- #
def _consultar_guia(pregunta: str) -> str:
    from .rag import rag

    return rag.preguntar(pregunta).get("respuesta", "")


_URL_RE = re.compile(r"https?://[^\s)]+")
_ERRORES_WEB = ("La búsqueda web no", "No encontré", "No pude")


def _resumir_es(pregunta: str, crudo: str) -> str:
    """Reescribe en ESPAÑOL SOLO la prosa (Tavily suele responder en inglés). Las URLs se
    manejan aparte (ver ``_consultar_web``). 1 llamada barata; si falla, devuelve el crudo.
    """
    system = (
        "Eres un guía turístico de Santander, Colombia. Con base ÚNICAMENTE en la información "
        "de búsqueda web proporcionada, responde en ESPAÑOL, claro y conciso, a la pregunta. "
        "No inventes nada que no esté en esa información. NO incluyas URLs ni una sección de "
        "fuentes (se agregan por separado); escribe solo el texto de la respuesta."
    )
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([("system", "{s}"), ("human", "{h}")])
        cadena = prompt | llm.construir_chat_model(temperature=0.2) | StrOutputParser()
        human = f"Pregunta: {pregunta}\n\nInformación de búsqueda web:\n{crudo}"
        return (cadena.invoke({"s": system, "h": human}) or crudo).strip()
    except Exception:  # noqa: BLE001 - si la síntesis falla, al menos devolvemos el crudo
        return crudo


def _consultar_web(pregunta: str) -> str:
    """Busca en la web y devuelve la respuesta en ESPAÑOL + las fuentes (URLs) limpias.

    La prosa la reescribe el LLM (Tavily contesta en inglés); las URLs se extraen del crudo
    y se re-anexan por código, para un formato de fuentes consistente (sin guiones dobles).
    """
    from .tools import busca_web

    crudo = busca_web.invoke(pregunta)
    if not crudo or crudo.startswith(_ERRORES_WEB):
        return crudo
    urls = []
    for u in _URL_RE.findall(crudo):
        u = u.rstrip(".,)")
        if u not in urls:
            urls.append(u)
    prosa = _resumir_es(pregunta, crudo)
    # Quita cualquier bloque de fuentes que el modelo haya dejado y re-anexa uno limpio.
    prosa = re.split(r"\n\s*Fuentes?\s*:", prosa, flags=re.IGNORECASE)[0].strip()
    if urls:
        prosa += "\n\nFuentes:\n" + "\n".join(f"- {u}" for u in urls[:3])
    return prosa


# ------------------------------- Nodos ---------------------------------- #
def nodo_router(state: EstadoTurismo) -> dict:
    """Clasifica la pregunta en guia / web / ambas.

    Atajo determinista: si hay señales claras de info actual (eventos, clima, precios,
    fechas), va directo a ``web`` sin llamar al LLM (0 costo y sin depender de que el
    router acierte). Para el resto, 1 llamada barata de clasificación estructurada.
    """
    from pydantic import BaseModel, Field

    if _SENAL_WEB.search(state.get("pregunta", "")):
        return {"ruta": "web"}

    class RutaOut(BaseModel):
        destino: Literal["guia", "web", "ambas"] = Field(
            description="La fuente más adecuada para la pregunta"
        )

    try:
        modelo = llm.construir_chat_model(temperature=0, con_respaldo=False)
        ruta = modelo.with_structured_output(RutaOut).invoke(
            [("system", SYSTEM_ROUTER), ("human", state["pregunta"])]
        )
        return {"ruta": ruta.destino}
    except Exception:  # noqa: BLE001 - ante cualquier fallo, usamos la guía
        return {"ruta": "guia"}


def nodo_guia(state: EstadoTurismo) -> dict:
    return {"respuesta_guia": _consultar_guia(state["pregunta"])}


def nodo_web(state: EstadoTurismo) -> dict:
    return {"respuesta_web": _consultar_web(state["pregunta"])}


def nodo_ambas(state: EstadoTurismo) -> dict:
    # Guía (1 llamada al LLM) + web (0 llamadas, Tavily). Secuencial y simple.
    return {
        "respuesta_guia": _consultar_guia(state["pregunta"]),
        "respuesta_web": _consultar_web(state["pregunta"]),
    }


def nodo_supervisor(state: EstadoTurismo) -> dict:
    """Combina las respuestas en Markdown. SIN LLM (solo formatea) → costo cero."""
    partes = []
    guia = state.get("respuesta_guia")
    web = state.get("respuesta_web")
    if guia:
        partes.append("### 🗺️ Según la guía de Santander\n\n" + guia.strip())
    if web:
        partes.append("### 🌐 Información actual (web)\n\n" + web.strip())
    final = "\n\n".join(partes) if partes else "No encontré información para tu consulta."
    return {"respuesta_final": final}


# ------------------------------- Grafo ---------------------------------- #
_grafo_cache = None


def _construir_grafo():
    from langgraph.graph import END, START, StateGraph

    wf = StateGraph(EstadoTurismo)
    wf.add_node("router", nodo_router)
    wf.add_node("guia", nodo_guia)
    wf.add_node("web", nodo_web)
    wf.add_node("ambas", nodo_ambas)
    wf.add_node("supervisor", nodo_supervisor)

    wf.add_edge(START, "router")
    # Arista condicional: el router decide a qué nodo ir (sin fan-out, robusto).
    wf.add_conditional_edges(
        "router",
        lambda s: s.get("ruta", "guia"),
        {"guia": "guia", "web": "web", "ambas": "ambas"},
    )
    wf.add_edge("guia", "supervisor")
    wf.add_edge("web", "supervisor")
    wf.add_edge("ambas", "supervisor")
    wf.add_edge("supervisor", END)
    return wf.compile()


def _get_grafo():
    global _grafo_cache
    if _grafo_cache is None:
        _grafo_cache = _construir_grafo()
    return _grafo_cache


def responder(pregunta: str) -> dict:
    """Ejecuta el grafo combinado y devuelve la respuesta final + la ruta elegida."""
    pregunta = (pregunta or "").strip()
    if not pregunta:
        return {"respuesta": "Escribe una pregunta.", "ruta": "", "fuentes": []}
    estado = _get_grafo().invoke({"pregunta": pregunta})
    fuentes = [
        f for f, k in (("guía", "respuesta_guia"), ("web", "respuesta_web")) if estado.get(k)
    ]
    return {
        "respuesta": estado.get("respuesta_final", ""),
        "ruta": estado.get("ruta", ""),
        "fuentes": fuentes,
    }


def diagrama_mermaid() -> str:
    """Devuelve el grafo en sintaxis Mermaid (para visualizarlo)."""
    return _get_grafo().get_graph().draw_mermaid()
