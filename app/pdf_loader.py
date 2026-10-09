"""Lectura y extracción de texto de un documento PDF."""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def _limpiar(texto: str) -> str:
    """Normaliza espacios y saltos de línea sobrantes."""
    texto = texto.replace("­", "")          # guiones de división invisibles
    texto = re.sub(r"[ \t]+", " ", texto)         # espacios múltiples
    texto = re.sub(r"\n{3,}", "\n\n", texto)      # saltos de línea excesivos
    return texto.strip()


def leer_pdf(ruta: str | Path) -> str:
    """Devuelve todo el texto del PDF como una sola cadena limpia.

    Args:
        ruta: ruta al archivo PDF.

    Returns:
        El texto completo extraído del documento.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el PDF: {ruta}")

    lector = PdfReader(str(ruta))
    partes: list[str] = []
    for pagina in lector.pages:
        texto = pagina.extract_text() or ""
        if texto.strip():
            partes.append(texto)

    return _limpiar("\n\n".join(partes))
