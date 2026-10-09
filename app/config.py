"""Carga y centraliza la configuración del proyecto desde variables de entorno."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Carga las variables definidas en el archivo .env (si existe).
load_dotenv()


def _get(nombre: str, por_defecto: str = "") -> str:
    return os.getenv(nombre, por_defecto).strip()


@dataclass(frozen=True)
class Settings:
    """Configuración inmutable de la aplicación."""

    # Proveedor de LLM principal: "gemini", "groq" o "cohere". El resto queda como
    # respaldo automático en ese orden (p. ej. gemini -> groq -> cohere).
    llm_provider: str = _get("LLM_PROVIDER", "gemini")

    # --- Límite de peticiones por MINUTO por proveedor (rate limiter del cliente) ---
    # Evita superar el RPM del free tier — clave para el agente ReAct, que hace varias
    # llamadas seguidas. El limitador es compartido por proveedor y ESPACIA (no descarta)
    # las llamadas. 0 = sin límite. Defaults por debajo del tope real de cada free tier.
    gemini_rpm: int = int(_get("GEMINI_RPM", "12") or 12)   # tope real ~15/min
    groq_rpm: int = int(_get("GROQ_RPM", "25") or 25)       # tope real ~30/min
    cohere_rpm: int = int(_get("COHERE_RPM", "18") or 18)   # trial ~20/min

    # --- Google Gemini ---
    gemini_api_key: str = _get("GEMINI_API_KEY")
    # Modelo de chat. Se fija en 2.5 Flash porque tiene el free tier más holgado
    # (15 RPM / 1M TPM / 1500 RPD). El alias "latest" puede saltar a Gemini 3 Flash,
    # que baja a 10 RPM / 250k TPM; para la demo preferimos el margen de 2.5 Flash.
    chat_model: str = _get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    # Tope de tokens de salida (amplio para que el "pensamiento" del modelo no
    # trunque la respuesta final).
    max_output_tokens: int = int(_get("MAX_OUTPUT_TOKENS", "50000") or 50000)

    # --- Groq (modelos open source: Llama, Gemma) ---
    groq_api_key: str = _get("GROQ_API_KEY")
    groq_model: str = _get("GROQ_MODEL", "llama-3.3-70b-versatile")
    # Modelos de respaldo de Groq (cada uno con su cuota diaria propia). Si el
    # modelo principal se queda sin cupo, se intenta con estos, en orden.
    groq_fallback_models: str = _get(
        "GROQ_FALLBACK_MODELS", "llama-3.1-8b-instant,gemma2-9b-it"
    )

    # --- Cohere (fuerte en contexto en español/latinoamericano) ---
    cohere_api_key: str = _get("COHERE_API_KEY")
    cohere_model: str = _get("COHERE_MODEL", "command-r-08-2024")
    # Modelo de embeddings de Cohere (para el RAG real).
    cohere_embed_model: str = _get("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")

    # --- RAG (recuperación por embeddings + base vectorial), vía paralela a /ask ---
    # Directorio con los documentos (PDF) que indexa el RAG (multi-documento).
    rag_docs_dir: str = _get("RAG_DOCS_DIR", "data")
    # Backend de base vectorial (strategy): "faiss" (local/persistido) o "pinecone" (nube).
    rag_vectorstore: str = _get("RAG_VECTORSTORE", "faiss")
    # Carpeta donde se PERSISTE el índice FAISS (para no re-indexar en cada arranque).
    rag_index_dir: str = _get("RAG_INDEX_DIR", "data/faiss_index")
    # Proveedor de embeddings: "cohere" (gratis en el trial) o "gemini".
    rag_embed_provider: str = _get("RAG_EMBED_PROVIDER", "cohere")
    # Estrategia de chunking: "recursive" (por caracteres) o "semantic" (por significado).
    rag_chunking: str = _get("RAG_CHUNKING", "recursive")
    rag_chunk_size: int = int(_get("RAG_CHUNK_SIZE", "800") or 800)
    rag_chunk_overlap: int = int(_get("RAG_CHUNK_OVERLAP", "100") or 100)
    rag_top_k: int = int(_get("RAG_TOP_K", "5") or 5)
    # Umbral mínimo de similitud (0-1). Con 0 (por defecto) se usa `similarity` top-k
    # puro (mejor recall; el prompt estricto + doble chequeo "No lo sé" evitan alucinar).
    # Con un valor > 0 se usa `similarity_score_threshold` y se descartan los flojos.
    rag_score_threshold: float = float(_get("RAG_SCORE_THRESHOLD", "0") or 0)
    # Multi-query (RAG avanzado, opt-in): reescribe la pregunta en varias versiones y
    # une los documentos recuperados. Mejora el recall a costa de una llamada extra al LLM.
    rag_multiquery: bool = _get("RAG_MULTIQUERY", "false").lower() == "true"
    rag_multiquery_n: int = int(_get("RAG_MULTIQUERY_N", "3") or 3)

    # --- Pinecone (base vectorial en la nube, RAG_VECTORSTORE=pinecone) ---
    pinecone_api_key: str = _get("PINECONE_API_KEY")
    pinecone_index: str = _get("PINECONE_INDEX", "santander-rag")
    pinecone_cloud: str = _get("PINECONE_CLOUD", "aws")
    pinecone_region: str = _get("PINECONE_REGION", "us-east-1")

    # Documento fuente
    pdf_path: str = _get("PDF_PATH", "data/guia_turistica_santander.pdf")

    # Memoria de conversaciones: un único CSV para todas las sesiones.
    history_csv: str = _get("HISTORY_CSV", "data/historial.csv")
    # Cuántos intercambios recientes de la sesión se usan como contexto.
    memory_max_turns: int = int(_get("MEMORY_MAX_TURNS", "6") or 6)

    # Clave de administrador para acciones protegidas (p. ej. análisis de datos).
    admin_key: str = _get("ADMIN_KEY")

    # Tavily: búsqueda web para el agente orquestador (info actual que no está en el PDF).
    tavily_api_key: str = _get("TAVILY_API_KEY")
    # Tope de pasos del orquestador ReAct (equivale al max_iterations del curso): evita que
    # el agente razone en bucle y gaste cuota de más. Cada ~2 pasos = 1 uso de herramienta.
    agente_max_pasos: int = int(_get("AGENTE_MAX_PASOS", "12") or 12)
    # Máximo de revisiones del generador de itinerarios (multiagente). 1 = un solo borrador
    # (más barato); 2 = una ronda de crítica + mejora. Cada revisión suma llamadas al LLM.
    itinerario_max_revisiones: int = int(_get("ITINERARIO_MAX_REVISIONES", "1") or 1)

    # LangSmith (observabilidad): LangChain lo activa por sí solo leyendo LANGSMITH_*
    # de las variables de entorno (cargadas por load_dotenv). Esto es solo para reportar
    # su estado; opt-in con LANGSMITH_TRACING=true.
    langsmith_tracing: bool = _get("LANGSMITH_TRACING", "false").lower() == "true"

    def validar(self) -> None:
        """Lanza un error claro si falta la API key del proveedor activo."""
        prov = self.llm_provider.lower()
        claves = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "cohere": self.cohere_api_key,
        }
        urls = {
            "groq": "https://console.groq.com/keys",
            "gemini": "https://aistudio.google.com/app/apikey",
            "cohere": "https://dashboard.cohere.com/api-keys",
        }
        if prov not in claves:
            raise ValueError(
                f"LLM_PROVIDER inválido: '{self.llm_provider}'. Usa 'groq', 'gemini' o 'cohere'."
            )
        if not claves[prov]:
            raise ValueError(
                f"Falta la API key para LLM_PROVIDER={prov}. Consíguela en {urls[prov]}"
            )


settings = Settings()
