"""Tests unitarios que NO requieren conexión a la API de Gemini.

Validan las piezas locales: lectura del PDF y la carga del documento en el agente.

Ejecuta con:  py -m pytest -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.agent import TourismAgent  # noqa: E402
from app.memory import ConversationMemory  # noqa: E402
from app.pdf_loader import leer_pdf  # noqa: E402

PDF = Path(__file__).resolve().parent.parent / "data" / "guia_turistica_santander.pdf"


def test_lectura_pdf_extrae_texto():
    texto = leer_pdf(PDF)
    assert len(texto) > 500
    assert "Santander" in texto


def test_agente_carga_documento():
    agente = TourismAgent()
    assert agente.esta_listo() is False

    n = agente.indexar(pdf_path=str(PDF))
    assert n > 500
    assert agente.esta_listo() is True
    assert agente.fuente == "guia_turistica_santander.pdf"


def test_pregunta_vacia_no_llama_al_modelo():
    agente = TourismAgent()
    agente.indexar(pdf_path=str(PDF))
    # Una pregunta vacía se responde localmente, sin llamar a la API.
    resultado = agente.preguntar("   ")
    assert "respuesta" in resultado and "fuente" in resultado


def test_memoria_csv_guarda_filtra_y_recuerda(tmp_path):
    csv = tmp_path / "historial.csv"
    memoria = ConversationMemory(ruta_csv=csv, max_turnos=6)
    sid = "sess-test"
    assert memoria.es_recurrente(sid) is False

    memoria.guardar_turno(sid, "¿Capital?", "Bucaramanga", ip="1.2.3.4")
    memoria.guardar_turno(sid, "¿Rafting?", "En San Gil, río Fonce")
    memoria.guardar_turno("otra-sesion", "¿Comida?", "Hormigas culonas")

    # El CSV se creó y contiene las 3 filas.
    assert csv.exists()

    # Filtro por session_id: solo los turnos de esa sesión.
    sesion = memoria.historial_sesion(sid)
    assert sesion.shape[0] == 2
    assert memoria.es_recurrente(sid) is True

    # El contexto incluye la memoria de esa sesión.
    contexto = memoria.construir_contexto(sid)
    assert "Bucaramanga" in contexto and "San Gil" in contexto

    # Sesiones distintas no comparten memoria.
    assert memoria.historial_sesion("otra-sesion").shape[0] == 1
    assert memoria.es_recurrente("no-existe") is False


def test_memoria_busqueda_por_termino(tmp_path):
    memoria = ConversationMemory(ruta_csv=tmp_path / "historial.csv")
    memoria.guardar_turno("s1", "¿Dónde hago rafting?", "En San Gil, río Fonce")
    memoria.guardar_turno("s1", "¿Qué comer?", "Hormigas culonas")

    # Búsqueda por término (filtro str.contains, sin distinguir mayúsculas).
    resultados = memoria.buscar("rafting")
    assert len(resultados) == 1
    assert resultados[0]["respuesta"] == "En San Gil, río Fonce"

    assert memoria.buscar("SAN GIL")  # case-insensitive
    assert memoria.buscar("inexistente") == []


def test_memoria_listar_sesiones(tmp_path):
    memoria = ConversationMemory(ruta_csv=tmp_path / "historial.csv")
    memoria.guardar_turno("s1", "p1", "r1")
    memoria.guardar_turno("s1", "p2", "r2")
    memoria.guardar_turno("s2", "p3", "r3")

    sesiones = {s["session_id"]: s for s in memoria.listar_sesiones()}
    assert sesiones["s1"]["turnos"] == 2
    assert sesiones["s2"]["turnos"] == 1


# ---------------------- Estrategia de respaldo del LLM ---------------------- #
def _error_429() -> Exception:
    exc = Exception("cuota agotada")
    exc.status_code = 429  # type: ignore[attr-defined]
    return exc


_PROVEEDORES = ("groq", "gemini", "cohere")


def test_llm_devuelve_primer_exito(monkeypatch):
    for prov in _PROVEEDORES:
        monkeypatch.setitem(llm._GENERADORES, prov, lambda *a, **k: "RESPUESTA")
    assert llm.generar_texto("hola", "sys", 100) == "RESPUESTA"


def test_llm_pasa_al_siguiente_si_no_hay_cupo(monkeypatch):
    estado = {"n": 0}

    def generador(mensaje, system, max_tokens, modelo):
        estado["n"] += 1
        if estado["n"] == 1:  # el primer candidato se queda sin cupo
            raise _error_429()
        return "OK-RESPALDO"

    for prov in _PROVEEDORES:
        monkeypatch.setitem(llm._GENERADORES, prov, generador)
    assert llm.generar_texto("hola", "sys", 100) == "OK-RESPALDO"
    assert estado["n"] >= 2  # usó el respaldo


def test_llm_sin_cupo_lanza_error(monkeypatch):
    def siempre_sin_cupo(*a, **k):
        raise _error_429()

    for prov in _PROVEEDORES:
        monkeypatch.setitem(llm._GENERADORES, prov, siempre_sin_cupo)
    with pytest.raises(llm.SinCupoError):
        llm.generar_texto("hola", "sys", 100)


# --------------------- Análisis de conversaciones (admin) ------------------- #
def test_analytics_categoriza(monkeypatch, tmp_path):
    from app import analytics

    memoria = ConversationMemory(ruta_csv=tmp_path / "historial.csv")
    memoria.guardar_turno("s1", "¿Dónde comer?", "...")
    memoria.guardar_turno("s1", "¿Dónde hago rafting?", "...")

    # LLM simulado que devuelve JSON (como en la clase de JSON).
    fake = (
        '[{"pregunta": "¿Dónde comer?", "categoria": "gastronomia"}, '
        '{"pregunta": "¿Dónde hago rafting?", "categoria": "aventura"}]'
    )
    monkeypatch.setattr(analytics.llm, "generar_texto", lambda *a, **k: fake)

    res = analytics.analizar(memoria)
    assert res["total_preguntas"] == 2
    cats = {c["categoria"]: c["cantidad"] for c in res["categorias"]}
    assert cats == {"gastronomia": 1, "aventura": 1}


def test_analytics_limpia_json_con_fences(monkeypatch, tmp_path):
    from app import analytics

    memoria = ConversationMemory(ruta_csv=tmp_path / "historial.csv")
    memoria.guardar_turno("s1", "¿Qué clima hace?", "...")

    # El modelo a veces envuelve el JSON en ```json ... ```
    fake = '```json\n[{"pregunta": "¿Qué clima hace?", "categoria": "clima"}]\n```'
    monkeypatch.setattr(analytics.llm, "generar_texto", lambda *a, **k: fake)

    res = analytics.analizar(memoria)
    assert res["categorias"] == [{"categoria": "clima", "cantidad": 1}]


def test_analytics_sin_datos(tmp_path):
    from app import analytics

    memoria = ConversationMemory(ruta_csv=tmp_path / "historial.csv")
    res = analytics.analizar(memoria)
    assert res["total_preguntas"] == 0 and res["categorias"] == []


# ------------------------------ Visión (imágenes) --------------------------- #
def test_encode_image(tmp_path):
    import base64

    from app import vision

    ruta = tmp_path / "img.bin"
    contenido = b"\x89PNG\r\n\x1a\n datos de prueba"
    ruta.write_bytes(contenido)

    b64 = vision.encode_image(ruta)
    # Lo codificado debe decodificar de vuelta al contenido original.
    assert base64.b64decode(b64) == contenido


# ---------------------- Orquestador (agente con herramientas) --------------- #
def test_orquestador_registra_herramientas():
    from app.tools import HERRAMIENTAS

    nombres = {t.name for t in HERRAMIENTAS}
    assert {"guia_turistica", "buscar_historial", "explicar"} <= nombres


# ------------------------------ RAG (embeddings + FAISS) -------------------- #
def test_rag_chunking_del_documento():
    # No requiere red: solo verifica que el documento se divide en chunks.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from app.pdf_loader import leer_pdf

    texto = leer_pdf(PDF)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    fragmentos = splitter.split_text(texto)
    assert len(fragmentos) > 1
    assert all(len(f) <= 1000 for f in fragmentos)  # margen por el corte


def test_rag_sin_indexar():
    from app.rag import RagSantander

    assert RagSantander().esta_listo() is False


# ------------------------ Grafo de estados (LangGraph) ---------------------- #
def test_grafo_arista_decision_enruta():
    from app.graph import arista_decision_triaje

    assert arista_decision_triaje({"triaje": {"decision": "auto_resolver"}}) == "rag"
    assert arista_decision_triaje({"triaje": {"decision": "pedir_info"}}) == "info"
    assert arista_decision_triaje({"triaje": {"decision": "abrir_ticket"}}) == "ticket"
    assert arista_decision_triaje({}) == "rag"  # por defecto, intenta resolver


def test_grafo_arista_decision_rag():
    from app.graph import arista_decision_rag

    # RAG respondió -> termina.
    assert arista_decision_rag({"rag_exito": True}) == "ok"
    # RAG falló y hay palabra clave de gestión -> ticket.
    assert arista_decision_rag({"rag_exito": False, "pregunta": "Quiero reservar un tour"}) == "ticket"
    # RAG falló sin palabra clave -> pedir info.
    assert arista_decision_rag({"rag_exito": False, "pregunta": "algo raro"}) == "info"
