"""Herramientas (Tools) para el agente orquestador.

Cada herramienta es una función que el agente ReAct puede decidir usar. Se
definen con el decorador `@tool` de LangChain; su docstring es la "descripción"
que el agente lee para decidir cuándo usarla.

Este módulo es **paralelo**: no toca los endpoints existentes (`/ask`, `/vision`).
Está pensado para crecer (p. ej. una futura herramienta de base de datos).

**Anti-bucle:** cada herramienta se memoiza *por ejecución del agente* (ver
``ejecucion_aislada``): si el agente vuelve a llamar la misma herramienta con el
mismo argumento, se devuelve el resultado ya calculado + una nota para que dé la
respuesta final. Así se rompen los bucles de re-consulta y se ahorra cuota (no se
reejecuta el RAG ni la búsqueda web, ni el LLM re-razona en vano).
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager

from langchain_core.tools import tool

from . import llm
from .agent import TourismAgent
from .config import settings
from .memory import ConversationMemory

# Instancias propias del orquestador (aisladas del resto de la app).
_agente = TourismAgent()
_memoria = ConversationMemory()


def _asegurar_indexado() -> None:
    if not _agente.esta_listo():
        _agente.indexar()


# --------------------------- Anti-bucle (caché por ejecución) --------------------------- #
_cache_run: contextvars.ContextVar = contextvars.ContextVar("cache_herramientas", default=None)


@contextmanager
def ejecucion_aislada():
    """Aísla una ejecución del agente: activa la caché de herramientas (anti-bucle).

    Envuelve la llamada al agente en el orquestador. Fuera de este contexto, las
    herramientas funcionan igual pero sin caché (p. ej. usadas desde el grafo).
    """
    token = _cache_run.set({})
    try:
        yield
    finally:
        _cache_run.reset(token)


def _consulta_cacheada(nombre: str, arg: str, calcular):
    """Devuelve el resultado; si ya se pidió (misma herramienta + argumento) en esta
    ejecución, devuelve el cacheado + una nota para que el agente NO repita la herramienta."""
    cache = _cache_run.get()
    clave = (nombre, (arg or "").strip().lower())
    if cache is not None and clave in cache:
        return (
            cache[clave]
            + "\n\n(Nota: ya consultaste esto antes en esta conversación; usa esta "
            "información para dar la respuesta final, no vuelvas a llamar la herramienta.)"
        )
    resultado = calcular()
    if cache is not None:
        cache[clave] = resultado
    return resultado


# ------------------------------- Herramientas ------------------------------- #
@tool
def guia_turistica(pregunta: str) -> str:
    """Responde preguntas sobre turismo en Santander, Colombia (destinos, gastronomía,
    deportes de aventura, transporte, mejor época) usando la guía oficial en PDF.
    Úsala para cualquier consulta sobre qué visitar, comer o hacer en Santander."""
    _asegurar_indexado()
    return _consulta_cacheada(
        "guia_turistica", pregunta, lambda: _agente.preguntar(pregunta)["respuesta"]
    )


@tool
def buscar_historial(termino: str) -> str:
    """Busca en el historial de conversaciones guardadas las que mencionan un término.
    Úsala cuando pregunten qué han consultado antes los usuarios o para recordar
    conversaciones previas."""

    def _run() -> str:
        resultados = _memoria.buscar(termino)
        if not resultados:
            return f"No hay conversaciones guardadas que mencionen '{termino}'."
        return "\n".join(
            f"- {r['pregunta']} -> {r['respuesta'][:80]}" for r in resultados[:5]
        )

    return _consulta_cacheada("buscar_historial", termino, _run)


@tool
def explicar(tema: str) -> str:
    """Explica un tema de forma didáctica y sencilla, con ejemplos cotidianos y pensando
    en el público colombiano. Úsala cuando pidan EXPLICAR o ENSEÑAR un concepto (no para
    datos concretos de la guía turística)."""

    def _run() -> str:
        plantilla = (
            "Asume el papel de un profesor didáctico. Explica el tema de forma sencilla, para "
            "estudiantes de secundaria, con ejemplos cotidianos y, si aplica, pensando en el "
            "contexto colombiano. Responde en español.\n\nTema: {tema}"
        )
        # Esta herramienta usa Cohere específicamente (fuerte en contexto en español);
        # si no hay clave de Cohere, cae al proveedor por defecto (cadena de respaldo).
        if settings.cohere_api_key:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_cohere import ChatCohere

            cadena = (
                PromptTemplate(template=plantilla, input_variables=["tema"])
                | ChatCohere(cohere_api_key=settings.cohere_api_key, model=settings.cohere_model)
                | StrOutputParser()
            )
            return cadena.invoke({"tema": tema})
        return llm.generar_texto(
            plantilla.format(tema=tema), "Eres un profesor didáctico. Responde en español.", 1024
        )

    return _consulta_cacheada("explicar", tema, _run)


@tool
def busca_web(consulta: str) -> str:
    """Busca información ACTUAL en internet (eventos y ferias próximas, clima de hoy, precios
    vigentes, noticias recientes) que NO está en la guía turística en PDF. Úsala SOLO cuando
    la pregunta necesite datos recientes o en tiempo real; para lo estático de Santander
    (destinos, gastronomía, rutas) usa la herramienta guia_turistica. Cita siempre la fuente."""
    if not settings.tavily_api_key:
        return "La búsqueda web no está disponible ahora (falta configurar TAVILY_API_KEY)."

    def _run() -> str:
        try:
            from datetime import date

            from langchain_tavily import TavilySearch

            buscador = TavilySearch(
                max_results=4,
                tavily_api_key=settings.tavily_api_key,
                topic="general",
                include_answer=True,      # respuesta ya sintetizada por Tavily (mucho mejor
                                          # que los resultados crudos, que traían ruido).
                search_depth="advanced",  # resultados más relevantes (mejor para eventos/precios).
            )
            # Consulta enfocada: se quitan signos (¿? confundían al buscador, que devolvía la
            # definición de "que") y se añade contexto geográfico + la FECHA ACTUAL, calculada
            # en cada llamada, para traer info reciente sin quedar desactualizada.
            fecha_hoy = date.today().strftime("%d/%m/%Y")
            limpia = consulta.strip().strip("¿?").strip()
            datos = buscador.invoke(
                {"query": f"{limpia} — Santander, Colombia (información al {fecha_hoy})"}
            )
            if not isinstance(datos, dict):
                datos = {"results": datos or []}
            respuesta = (datos.get("answer") or "").strip()
            resultados = datos.get("results", []) or []
            partes = []
            if respuesta:
                partes.append(respuesta)
            fuentes = []
            for r in resultados[:3]:
                titulo = (r.get("title") or "").strip()
                url = (r.get("url") or "").strip()
                if url:
                    fuentes.append(f"- {titulo}: {url}" if titulo else f"- {url}")
            if fuentes:
                partes.append("Fuentes:\n" + "\n".join(fuentes))
            if not partes:
                return "No encontré resultados web para esa consulta."
            return "\n\n".join(partes)
        except Exception as exc:  # noqa: BLE001 - la búsqueda web no debe tumbar al agente
            return f"No pude completar la búsqueda web en este momento ({exc})."

    return _consulta_cacheada("busca_web", consulta, _run)


# Lista de herramientas disponibles para el orquestador.
HERRAMIENTAS = [guia_turistica, buscar_historial, explicar, busca_web]
