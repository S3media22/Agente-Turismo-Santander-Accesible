"""Orquestación del agente de preguntas y respuestas.

Estrategia: **inyección de contexto completo**. Como el documento fuente es
pequeño, en lugar de usar recuperación por embeddings se entrega el texto
completo del PDF como contexto al LLM en cada pregunta.

Además, el agente puede recibir el **contexto de conversación** de la sesión
(memoria) para "recordar" a un usuario que ya interactuó antes. La llamada al
modelo se delega a `app.llm`, que elige el proveedor y aplica el respaldo.
"""
from __future__ import annotations

import os

from . import llm
from .config import settings
from .pdf_loader import leer_pdf

SYSTEM_PROMPT = (
    "Eres un guía turístico experto y amable del departamento de Santander, Colombia. "
    "Tu fuente principal es la guía proporcionada: úsala para los DATOS CONCRETOS de "
    "Santander (lugares, actividades, gastronomía, rutas, precios, horarios). "
    "Además, ayuda con recomendaciones y consejos prácticos de viaje aunque no estén "
    "literalmente en la guía —por ejemplo, qué calzado usar para caminar, cómo prepararse "
    "para acampar o hacer senderismo, consejos de seguridad o de clima— aplicando sentido "
    "común y buenas prácticas de turismo, y relacionándolo con Santander cuando puedas. "
    "NO inventes datos específicos que no puedas saber (precios exactos, horarios, cifras "
    "o lugares concretos que no aparezcan en la guía), y en especial NUNCA inventes EVENTOS, "
    "FERIAS, FIESTAS, FESTIVALES o FECHAS: la guía no incluye agenda de eventos. Si te "
    "preguntan por eventos/ferias/fiestas actuales, dilo con honestidad y sugiere usar la "
    "fase '🌐 Guía + Web' del asistente, que consulta información actual en internet. En los "
    "demás casos sin dato exacto, ofrece una orientación general. Responde que no tienes "
    "esa información SOLO cuando te pidan un dato ESPECÍFICO de Santander que no esté en la "
    "guía y que no puedas cubrir con un consejo general útil. Mantente en el tema de turismo. "
    "Si hay memoria de una conversación previa, tenla en cuenta para dar continuidad. "
    "Responde en español, claro y conciso; devuelve ÚNICAMENTE la respuesta final para el "
    "usuario, sin notas de proceso ni texto de razonamiento. NO saludes ni te presentes en "
    "cada respuesta (nada de '¡Hola! Soy un guía turístico...'); ve directo a responder. Solo "
    "preséntate, en una frase breve, si te preguntan explícitamente quién eres, y no repitas "
    "frases de respuestas anteriores."
)


class TourismAgent:
    """Agente que responde preguntas sobre la guía turística de Santander."""

    def __init__(self) -> None:
        self.contexto: str | None = None
        self.fuente: str = ""

    # ------------------------------------------------------------------ #
    # Carga del documento
    # ------------------------------------------------------------------ #
    def indexar(self, pdf_path: str | None = None, forzar: bool = False) -> int:
        """Carga el texto del documento fuente en memoria."""
        ruta_pdf = pdf_path or settings.pdf_path
        self.contexto = leer_pdf(ruta_pdf)
        self.fuente = os.path.basename(ruta_pdf)
        return len(self.contexto)

    def esta_listo(self) -> bool:
        return bool(self.contexto)

    # ------------------------------------------------------------------ #
    # Consulta
    # ------------------------------------------------------------------ #
    def preguntar(self, pregunta: str, contexto_conversacion: str = "") -> dict:
        """Responde una pregunta usando el documento (y la memoria) como contexto.

        Args:
            pregunta: la pregunta del usuario.
            contexto_conversacion: memoria de la sesión (últimos intercambios).

        Returns:
            {"respuesta": str, "fuente": str}
        """
        if not self.esta_listo():
            raise RuntimeError("El documento no está cargado. Llama a indexar() primero.")

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {"respuesta": "Por favor, escribe una pregunta.", "fuente": self.fuente}

        bloques = [f"### Documento de referencia:\n{self.contexto}"]
        if contexto_conversacion:
            bloques.append(f"### Memoria de la conversación:\n{contexto_conversacion}")
        bloques.append(f"### Pregunta del usuario:\n{pregunta}")
        mensaje = "\n\n".join(bloques)

        respuesta = self._llamar_modelo(
            mensaje, SYSTEM_PROMPT, max_tokens=settings.max_output_tokens
        )
        return {"respuesta": respuesta, "fuente": self.fuente}

    def preguntar_stream(self, pregunta: str, contexto_conversacion: str = ""):
        """Igual que `preguntar`, pero hace `yield` de la respuesta en fragmentos (streaming)."""
        if not self.esta_listo():
            raise RuntimeError("El documento no está cargado. Llama a indexar() primero.")
        pregunta = (pregunta or "").strip()
        if not pregunta:
            yield "Por favor, escribe una pregunta."
            return
        bloques = [f"### Documento de referencia:\n{self.contexto}"]
        if contexto_conversacion:
            bloques.append(f"### Memoria de la conversación:\n{contexto_conversacion}")
        bloques.append(f"### Pregunta del usuario:\n{pregunta}")
        mensaje = "\n\n".join(bloques)
        yield from llm.stream_texto(mensaje, SYSTEM_PROMPT, temperature=0.2)

    # ------------------------------------------------------------------ #
    # Llamada al modelo (con reintentos)
    # ------------------------------------------------------------------ #
    def _llamar_modelo(
        self, mensaje: str, system_instruction: str, max_tokens: int = 2048
    ) -> str:
        """Genera texto con el proveedor de LLM configurado (Gemini, Groq o Cohere)."""
        return llm.generar_texto(mensaje, system_instruction, max_tokens)
