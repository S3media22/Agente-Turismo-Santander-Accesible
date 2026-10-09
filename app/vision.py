"""Análisis de imágenes (visión) con Gemini vía LangChain — salida estructurada.

Usa la mejor práctica de LangChain para salida estructurada:
`modelo.with_structured_output(ModeloPydantic)`, que aprovecha el *function-calling*
nativo del modelo y devuelve un objeto **validado** (más confiable que parsear texto
con `JsonOutputParser`). El modelo Pydantic usa un campo `Literal` para restringir el
tipo de imagen a valores válidos.

Como se vio en el curso, **solo Gemini** hace visión en esta configuración
(Groq/Cohere son de texto), así que se usa `ChatGoogleGenerativeAI`. La imagen se
codifica en base64 y se envía en un mensaje multimodal.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import settings

SYSTEM_VISION = (
    "Eres un guía turístico experto en el departamento de Santander, Colombia. "
    "Analiza la imagen de forma objetiva y, si reconoces un lugar, plato típico o "
    "actividad relacionada con el turismo de Santander, indícalo. Responde en español."
)


class AnalisisImagen(BaseModel):
    """Estructura de salida del análisis de una imagen (modelo Pydantic)."""

    titulo: str = Field(description="Un título breve y adecuado para la imagen")
    descripcion: str = Field(description="Descripción clara y objetiva de la imagen, en español")
    etiquetas: list[str] = Field(description="Entre 3 y 5 palabras clave en minúsculas, sin tildes")
    # Literal: restringe la salida a un conjunto fijo de valores válidos.
    tipo: Literal["lugar", "plato", "actividad", "otro"] = Field(
        description="Qué muestra principalmente la imagen"
    )
    relacion_santander: str = Field(
        description="Relación con el turismo de Santander (lugar, plato o actividad), o 'ninguna'"
    )


def encode_image(ruta: str | Path) -> str:
    """Lee una imagen y la devuelve codificada en base64 (texto)."""
    with open(ruta, "rb") as archivo:
        return base64.b64encode(archivo.read()).decode("utf-8")


def _modelo():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.gemini_api_key:
        raise ValueError("Falta GEMINI_API_KEY para el análisis de imágenes.")
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
        max_output_tokens=settings.max_output_tokens,
    )


def analizar_imagen(imagen_b64: str, mime: str = "image/jpeg", pregunta: str = "") -> dict:
    """Analiza una imagen y devuelve un dict con la estructura de `AnalisisImagen`."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    chat = _modelo()
    data_uri = f"data:{mime};base64,{imagen_b64}"
    texto = pregunta.strip() or "Analiza esta imagen."
    template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_VISION),
            (
                "user",
                [
                    {"type": "text", "text": "{pregunta}"},
                    {"type": "image_url", "image_url": "{imagen}"},
                ],
            ),
        ]
    )
    try:
        # Mejor práctica: salida estructurada nativa (validada por el modelo Pydantic).
        cadena = template | chat.with_structured_output(AnalisisImagen)
        resultado = cadena.invoke({"pregunta": texto, "imagen": data_uri})
        return resultado.model_dump() if hasattr(resultado, "model_dump") else dict(resultado)
    except Exception as exc:  # noqa: BLE001 - respaldo a texto plano si falla
        print(f"[vision] salida estructurada falló ({exc}); uso texto plano.")
        texto_plano = (template | chat | StrOutputParser()).invoke(
            {"pregunta": texto, "imagen": data_uri}
        )
        return {
            "titulo": "",
            "descripcion": texto_plano.strip(),
            "etiquetas": [],
            "tipo": "otro",
            "relacion_santander": "",
        }
