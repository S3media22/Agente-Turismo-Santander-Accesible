"""Generador de itinerarios de Santander — multiagente con LangGraph.

Adapta el patrón multiagente del curso (planificar → investigar → redactar → criticar →
revisar) al dominio turístico: en vez de escribir ensayos, arma un ITINERARIO de viaje por
Santander combinando la **guía (PDF)** con **info actual de la web (Tavily)**, con un loop de
revisión opcional.

Flujo:  START → plan → investigar → redactar → [reflexionar → redactar]* → END

⚠️ **Costo:** hace varias llamadas al LLM por request (plan + investigación + redacción +
cada revisión). ``ITINERARIO_MAX_REVISIONES`` acota el gasto (1 = un solo borrador). Usa el
respaldo de proveedores como el resto del sistema.
"""
from __future__ import annotations

import re
from typing import List, TypedDict

from . import llm
from .config import settings

PLAN_PROMPT = (
    "Eres un planificador de viajes experto en el departamento de Santander, Colombia. "
    "Dada la solicitud del usuario, esboza un itinerario de ALTO NIVEL (días, zonas a cubrir, "
    "ritmo). Sé conciso; responde en español."
)
RESEARCH_PROMPT = (
    "Genera hasta 3 consultas de búsqueda web para reunir información ACTUAL y útil para el "
    "itinerario en Santander (eventos/ferias, clima, horarios, precios vigentes). Devuelve "
    "SOLO las consultas, una por línea, sin numeración ni texto extra."
)
WRITER_PROMPT = (
    "Eres un guía turístico de Santander, Colombia. Escribe un itinerario de viaje claro y "
    "práctico, organizado POR DÍAS, para la solicitud del usuario, usando el plan y la "
    "información disponible. Incluye destinos, gastronomía típica, actividades y consejos "
    "prácticos. Si se te da una crítica previa, entrega una versión MEJORADA. Responde en "
    "español, en Markdown.\n\nInformación disponible:\n{contenido}"
)
REFLECT_PROMPT = (
    "Eres un revisor de itinerarios de viaje por Santander. Da una crítica BREVE y concreta: "
    "¿faltan comidas típicas, tiempos realistas entre lugares, opciones de aventura, o algo "
    "del clima/eventos actuales? Recomienda mejoras puntuales."
)


class EstadoItinerario(TypedDict, total=False):
    solicitud: str
    plan: str
    borrador: str
    critica: str
    contenido: List[str]
    num_revision: int
    max_revisiones: int


def _texto(system: str, human: str, temperatura: float = 0.3) -> str:
    """Invoca la cadena con respaldo (system + human) y devuelve el texto."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([("system", "{sys}"), ("human", "{hum}")])
    cadena = prompt | llm.construir_chat_model(temperature=temperatura) | StrOutputParser()
    return (cadena.invoke({"sys": system, "hum": human}) or "").strip()


# ------------------------------- Nodos ---------------------------------- #
def nodo_plan(state: EstadoItinerario) -> dict:
    return {"plan": _texto(PLAN_PROMPT, state["solicitud"])}


def nodo_investigar(state: EstadoItinerario) -> dict:
    """Genera consultas, busca en la web (Tavily) y suma un poco de la guía (PDF)."""
    salida = _texto(RESEARCH_PROMPT, state["solicitud"], temperatura=0.4)
    consultas = [
        linea.strip(" -*\"'") for linea in salida.splitlines() if linea.strip()
    ][:3]

    from .tools import busca_web

    contenido = list(state.get("contenido") or [])
    for q in consultas:
        try:
            contenido.append(busca_web.invoke(q))
        except Exception:  # noqa: BLE001 - la búsqueda no debe tumbar el flujo
            pass
    # Contexto de la guía oficial (lo estático de Santander).
    try:
        from .rag import rag

        contenido.append(rag.preguntar(state["solicitud"]).get("respuesta", ""))
    except Exception:  # noqa: BLE001
        pass
    return {"contenido": [c for c in contenido if c]}


def nodo_redactar(state: EstadoItinerario) -> dict:
    contenido = "\n\n".join(state.get("contenido") or [])[:6000]
    human = f"Solicitud: {state['solicitud']}\n\nPlan:\n{state.get('plan', '')}"
    if state.get("critica"):
        human += f"\n\nCrítica a incorporar:\n{state['critica']}"
    borrador = _texto(WRITER_PROMPT.format(contenido=contenido), human)
    return {"borrador": borrador, "num_revision": state.get("num_revision", 0) + 1}


def nodo_reflexionar(state: EstadoItinerario) -> dict:
    return {"critica": _texto(REFLECT_PROMPT, state.get("borrador", ""))}


def _debe_continuar(state: EstadoItinerario) -> str:
    """Corta cuando ya se alcanzó el máximo de revisiones (control de costo)."""
    if state.get("num_revision", 0) >= state.get("max_revisiones", 1):
        return "fin"
    return "seguir"


# ------------------------------- Grafo ---------------------------------- #
_grafo_cache = None


def _construir_grafo():
    from langgraph.graph import END, START, StateGraph

    wf = StateGraph(EstadoItinerario)
    wf.add_node("plan", nodo_plan)
    wf.add_node("investigar", nodo_investigar)
    wf.add_node("redactar", nodo_redactar)
    wf.add_node("reflexionar", nodo_reflexionar)
    wf.add_edge(START, "plan")
    wf.add_edge("plan", "investigar")
    wf.add_edge("investigar", "redactar")
    # Loop de revisión: redactar → (reflexionar → redactar)* hasta max_revisiones.
    wf.add_conditional_edges(
        "redactar", _debe_continuar, {"seguir": "reflexionar", "fin": END}
    )
    wf.add_edge("reflexionar", "redactar")
    return wf.compile()


def _get_grafo():
    global _grafo_cache
    if _grafo_cache is None:
        _grafo_cache = _construir_grafo()
    return _grafo_cache


def responder(solicitud: str, max_revisiones: int | None = None) -> dict:
    """Genera un itinerario de viaje por Santander. Returns {itinerario, plan, revisiones}."""
    solicitud = (solicitud or "").strip()
    if not solicitud:
        return {"itinerario": "Cuéntame qué viaje quieres planear en Santander.", "plan": ""}
    mr = max_revisiones or settings.itinerario_max_revisiones
    estado = _get_grafo().invoke(
        {"solicitud": solicitud, "max_revisiones": mr, "num_revision": 0, "contenido": []}
    )
    return {
        "itinerario": estado.get("borrador", ""),
        "plan": estado.get("plan", ""),
        "revisiones": estado.get("num_revision", 0),
    }


def diagrama_mermaid() -> str:
    """Devuelve el grafo del multiagente en sintaxis Mermaid."""
    return _get_grafo().get_graph().draw_mermaid()
