"""Servicio de memoria de conversaciones en un archivo CSV (con pandas).

Todo el historial se guarda en un único CSV (`data/historial.csv`). Cada fila es
un intercambio (pregunta + respuesta) con su `session_id`, fecha e IP.

Antes de responder, el sistema **lee el CSV y filtra por `session_id`** (aplicando
los filtros de pandas vistos en el curso) para recuperar la memoria de esa sesión
si el usuario ya interactuó antes. Así se puede buscar y reutilizar fácilmente.

Usar pandas para el CSV (en vez de escribir el texto a mano) evita que las comas o
los saltos de línea dentro de una respuesta rompan el archivo: pandas se encarga
del entrecomillado automáticamente.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import settings

COLUMNAS = ["session_id", "fecha", "ip", "pregunta", "respuesta"]


class ConversationMemory:
    """Lee y escribe el historial de conversaciones en un CSV usando pandas."""

    def __init__(self, ruta_csv: str | Path | None = None, max_turnos: int | None = None):
        self.ruta = Path(ruta_csv or settings.history_csv)
        # Cuántos intercambios recientes se incluyen como contexto (acota tokens).
        self.max_turnos = max_turnos or settings.memory_max_turns

    # ------------------------------------------------------------------ #
    # Lectura base
    # ------------------------------------------------------------------ #
    def _leer_df(self) -> pd.DataFrame:
        """Carga el CSV completo como DataFrame (vacío si aún no existe)."""
        if self.ruta.exists():
            return pd.read_csv(self.ruta, dtype=str, keep_default_na=False)
        return pd.DataFrame(columns=COLUMNAS)

    def cargar_df(self) -> pd.DataFrame:
        """Devuelve todo el historial como DataFrame (para análisis)."""
        return self._leer_df()

    # ------------------------------------------------------------------ #
    # Escritura de un turno
    # ------------------------------------------------------------------ #
    def guardar_turno(self, session_id: str, pregunta: str, respuesta: str, ip: str = "") -> dict:
        """Agrega un intercambio como una fila nueva del CSV."""
        fila = {
            "session_id": session_id,
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "ip": ip,
            "pregunta": pregunta,
            "respuesta": respuesta,
        }
        df = self._leer_df()
        df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.ruta, index=False, encoding="utf-8")
        return fila

    # ------------------------------------------------------------------ #
    # Búsqueda / filtros (principios de pandas del curso)
    # ------------------------------------------------------------------ #
    def historial_sesion(self, session_id: str) -> pd.DataFrame:
        """Filtra el historial de una sesión: df[df['session_id'] == session_id]."""
        df = self._leer_df()
        if df.empty or not session_id:
            return df.iloc[0:0]  # DataFrame vacío con las mismas columnas
        return df[df["session_id"] == session_id]

    def es_recurrente(self, session_id: str) -> bool:
        """True si la sesión ya tiene intercambios guardados (usa .shape)."""
        return self.historial_sesion(session_id).shape[0] > 0

    def buscar(self, termino: str) -> list[dict]:
        """Busca un término en las preguntas o respuestas (filtro str.contains)."""
        df = self._leer_df()
        if df.empty or not termino:
            return []
        mascara = df["pregunta"].str.contains(termino, case=False, na=False) | df[
            "respuesta"
        ].str.contains(termino, case=False, na=False)
        return df[mascara].to_dict(orient="records")

    def construir_contexto(self, session_id: str) -> str:
        """Arma el contexto de memoria con los últimos intercambios de la sesión."""
        sesion = self.historial_sesion(session_id)
        if sesion.empty:
            return ""
        # .tail(N): solo los últimos N intercambios, para no gastar tokens de más.
        ultimos = sesion.tail(self.max_turnos)
        lineas = [
            f"Usuario: {fila.pregunta}\nAsistente: {fila.respuesta}"
            for fila in ultimos.itertuples()
        ]
        return "Historial de esta conversación con el usuario:\n" + "\n".join(lineas)

    def listar_sesiones(self) -> list[dict]:
        """Resumen por sesión (agrupando con groupby): nº de turnos, última fecha, IP."""
        df = self._leer_df()
        if df.empty:
            return []
        resumen = (
            df.groupby("session_id")
            .agg(turnos=("pregunta", "count"), ultima=("fecha", "max"), ip=("ip", "last"))
            .reset_index()
        )
        return resumen.to_dict(orient="records")
