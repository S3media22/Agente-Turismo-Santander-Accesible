"""Análisis de las conversaciones guardadas (acción de administrador).

Combina lo aprendido en el curso:
  - **pandas**: lee el historial (CSV) y extrae la columna de preguntas.
  - **LLM + JSON**: pide al modelo que clasifique cada pregunta en una categoría
    y devuelva un JSON; luego se convierte a diccionario con `json.loads`.

Sirve para responder "¿qué es lo que más preguntan los turistas?".
"""
from __future__ import annotations

import json
import re
from collections import Counter

from . import llm
from .memory import ConversationMemory

SYSTEM_ANALISIS = (
    "Eres un analista de datos de turismo. Clasificas preguntas de usuarios en temas "
    "breves y devuelves únicamente JSON válido, sin texto adicional."
)

# Se limita la cantidad de preguntas para no exceder los tokens del modelo.
_MAX_PREGUNTAS = 40


def _parsear_json(texto: str):
    """Convierte el texto del modelo en objeto Python, tolerando 'ruido'.

    A veces el modelo envuelve el JSON en ```json ... ``` o agrega texto; se limpia
    y, si hace falta, se extrae el primer bloque de lista `[...]`.
    """
    limpio = texto.strip()
    if limpio.startswith("```"):
        limpio = limpio.strip("`")
        if limpio.lower().startswith("json"):
            limpio = limpio[4:]
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        bloque = re.search(r"\[.*\]", limpio, re.DOTALL)
        if bloque:
            return json.loads(bloque.group(0))
        raise


def analizar(memory: ConversationMemory) -> dict:
    """Clasifica las preguntas del historial y devuelve un resumen por categoría."""
    df = memory.cargar_df()
    if df.empty:
        return {"total_preguntas": 0, "categorias": [], "detalle": []}

    preguntas = df["pregunta"].tolist()[-_MAX_PREGUNTAS:]
    listado = "\n".join(f"- {p}" for p in preguntas)

    prompt = (
        "Clasifica cada una de las siguientes preguntas de turistas en UNA categoría "
        "breve (1-2 palabras, en minúsculas y sin tildes). Ejemplos de categorías: "
        "destinos, gastronomia, transporte, aventura, alojamiento, clima, general.\n\n"
        "Devuelve SOLO un JSON válido: una lista de objetos con las claves "
        '"pregunta" y "categoria". No agregues texto fuera del JSON.\n\n'
        f"Preguntas:\n{listado}"
    )

    texto = llm.generar_texto(prompt, SYSTEM_ANALISIS, max_tokens=2048)
    detalle = _parsear_json(texto)
    if not isinstance(detalle, list):  # el modelo debe devolver una lista de objetos
        detalle = []

    # Contamos cuántas preguntas hay por categoría (Counter).
    conteo = Counter(
        str(item.get("categoria", "general")).lower()
        for item in detalle
        if isinstance(item, dict)
    )
    categorias = [
        {"categoria": cat, "cantidad": n} for cat, n in conteo.most_common()
    ]

    return {
        "total_preguntas": len(preguntas),
        "categorias": categorias,
        "detalle": detalle,
    }
