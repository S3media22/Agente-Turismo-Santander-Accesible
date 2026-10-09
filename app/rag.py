"""RAG real (Retrieval-Augmented Generation) con embeddings y base vectorial.

A diferencia del agente por **contexto completo** (`app/agent.py`), aquí el flujo es
el RAG clásico:

  1. **Cargar** los PDFs y dividirlos en *chunks* (`RecursiveCharacterTextSplitter`).
  2. **Embeddings**: convertir cada chunk en un vector semántico (Cohere).
  3. **Indexar** los vectores en una **base vectorial** (strategy configurable):
       - ``faiss``    → local, **persistida en disco** (no re-indexa en cada arranque).
       - ``pinecone`` → en la **nube** (persistente y escalable a muchos documentos).
  4. **Recuperar** (retrieval): por cada pregunta se buscan los chunks más
     similares y solo esos se pasan al LLM para **generar** la respuesta.

El backend se elige con ``RAG_VECTORSTORE`` sin tocar el resto del pipeline. Es una
vía **paralela** (endpoint `/rag/ask`): no altera `/ask`. Sirve para escalar a
documentos grandes o a múltiples fuentes, donde inyectar todo el texto no es viable.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from . import llm
from .config import settings

# Palabras demasiado comunes en la guía para medir si un fragmento fue realmente usado.
# Se excluyen del solape porque aparecen en casi todos los chunks (o son relleno) y
# provocaban CITAS IRRELEVANTES: p. ej. una respuesta sobre "fiestas" citaba el chunk del
# aeropuerto solo por compartir "Bucaramanga"/"cultura"/"gastronomia".
_PALABRAS_COMUNES = {
    "santander", "colombia", "departamento", "lugares", "informacion", "tambien",
    "puedes", "sobre", "donde", "cuando", "estas", "estos", "entre", "principales",
    "bucaramanga", "turistico", "turistica", "turismo", "ciudad", "ciudades", "region",
    "viaje", "viajar", "visitar", "cultura", "cultural", "gastronomia", "tradicion",
    "actividades", "importante", "recomiendo", "ademas", "mejor", "puede", "tener",
    "cuenta", "opciones", "disfrutar", "ofrece", "encuentra",
}


def _terminos(texto: str) -> set:
    """Términos significativos (>=5 letras, sin palabras comunes) para medir solape."""
    return {t for t in re.findall(r"[a-záéíóúñü]{5,}", (texto or "").lower())} - _PALABRAS_COMUNES


# Caracteres de control (p. ej. el glifo de viñeta que pypdf extrae como \x7f): se quitan
# para que no ensucien el contexto del LLM ni las citaciones mostradas al usuario.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _limpiar(texto: str) -> str:
    return _CTRL.sub("", texto or "").strip()

SYSTEM_PROMPT_RAG = (
    "Eres un guía turístico experto y amable de Santander, Colombia. Responde en español, "
    "de forma clara y concisa. Para los DATOS CONCRETOS de Santander (lugares, actividades, "
    "gastronomía, rutas) básate en el contexto proporcionado. Puedes complementar con "
    "consejos prácticos de viaje y sentido común (p. ej. qué calzado usar para caminar, cómo "
    "prepararse para acampar o hacer senderismo) aunque no estén textualmente en el contexto. "
    "REGLA CLAVE: NUNCA inventes datos específicos que no estén en el contexto: precios, "
    "horarios, lugares concretos, y en especial EVENTOS, FERIAS, FIESTAS, FESTIVALES, "
    "CONCIERTOS o FECHAS. La guía NO incluye una agenda de eventos actuales. Si te preguntan "
    "por eventos/ferias/fiestas/fechas y NO están en el contexto, NO los inventes: aclara con "
    "honestidad que la guía no incluye agenda de eventos y sugiere usar la fase '🌐 Guía + "
    "Web' del asistente, que consulta información actual en internet. "
    "Responde exactamente 'No lo sé' SOLO cuando te pidan un dato ESPECÍFICO de Santander que "
    "no esté en el contexto y que no puedas cubrir ni con un consejo general útil ni con la "
    "aclaración anterior. Si se te proporciona el historial de la conversación, tenlo en "
    "cuenta para entender preguntas de seguimiento ('eso', 'ahí', 'y entonces', 'para eso'). "
    "IMPORTANTE: NO saludes ni te presentes en cada respuesta (nada de '¡Hola! Soy un guía "
    "turístico...'). Ve DIRECTO a responder la pregunta. Solo preséntate, en UNA frase breve, "
    "si te preguntan explícitamente quién eres. No repitas frases de una respuesta anterior."
)

# Multi-query (RAG avanzado): una LLM reescribe la pregunta en varias versiones para
# recuperar más documentos relevantes y superar los límites de la búsqueda por distancia.
MULTIQUERY_TEMPLATE = (
    "Eres un asistente de IA. Genera {n} versiones diferentes de la siguiente pregunta del "
    "usuario para recuperar documentos relevantes de una base de datos vectorial. Al ofrecer "
    "múltiples perspectivas, ayudas a superar las limitaciones de la búsqueda por similitud. "
    "Devuelve SOLO las preguntas, una por línea, sin numeración ni comillas ni texto extra.\n\n"
    "Pregunta original: {pregunta}"
)


class RagSantander:
    """Base vectorial (FAISS o Pinecone) de los documentos y recuperación semántica."""

    def __init__(self) -> None:
        self.retriever = None
        self.document_chain = None
        self._vectorstore = None  # referencia para el fallback de recuperación

    def _embeddings(self):
        """Modelo de embeddings: Cohere por defecto (gratis en el trial) o Gemini."""
        if settings.rag_embed_provider.lower() == "gemini" and settings.gemini_api_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001", google_api_key=settings.gemini_api_key
            )
        from langchain_cohere import CohereEmbeddings

        if not settings.cohere_api_key:
            raise ValueError("El RAG necesita COHERE_API_KEY (o RAG_EMBED_PROVIDER=gemini).")
        return CohereEmbeddings(
            cohere_api_key=settings.cohere_api_key, model=settings.cohere_embed_model
        )

    def _cargar_fragmentos(self, embeddings, docs_dir: str | None):
        """Carga los PDFs del directorio y los divide en chunks (loader + splitter)."""
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # DirectoryLoader + PyPDFLoader: carga todos los PDFs del directorio (una página
        # por Document con metadata source/page), conservada al dividir -> citaciones
        # trazables. Agregar más PDFs a la carpeta los indexa automáticamente.
        directorio = docs_dir or settings.rag_docs_dir
        paginas = DirectoryLoader(
            directorio, glob="**/*.pdf", loader_cls=PyPDFLoader
        ).load()
        if not paginas:
            raise FileNotFoundError(f"No se encontraron PDFs en '{directorio}' para el RAG.")

        # Chunking: "semantic" divide por significado (usa embeddings); "recursive"
        # (por defecto) divide por caracteres con solapamiento.
        if settings.rag_chunking.lower() == "semantic":
            from langchain_experimental.text_splitter import SemanticChunker

            splitter = SemanticChunker(embeddings)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap
            )
        return splitter.split_documents(paginas)

    def _vs_faiss(self, embeddings, docs_dir, forzar):
        """Backend FAISS: base vectorial local **persistida en disco**.

        Si el índice ya existe y ``forzar`` es False, se **carga** (rápido y sin gastar
        cuota de embeddings); si no, se construye desde los PDFs y se guarda.
        """
        from langchain_community.vectorstores import FAISS

        ruta = Path(settings.rag_index_dir)
        if not forzar and (ruta / "index.faiss").exists():
            vs = FAISS.load_local(
                str(ruta), embeddings, allow_dangerous_deserialization=True
            )
            return vs, vs.index.ntotal
        fragmentos = self._cargar_fragmentos(embeddings, docs_dir)
        vs = FAISS.from_documents(fragmentos, embeddings)
        ruta.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(ruta))  # index.faiss + index.pkl
        return vs, len(fragmentos)

    def _vs_pinecone(self, embeddings, docs_dir, forzar):
        """Backend Pinecone: base vectorial en la **nube** (persistente y escalable).

        El índice vive en Pinecone, así que los vectores sobreviven a los reinicios y
        el proyecto puede crecer a muchos documentos sin recalcular todo. Solo se suben
        los fragmentos si el índice está vacío (o si ``forzar`` es True).
        """
        import time

        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        if not settings.pinecone_api_key:
            raise ValueError(
                "RAG_VECTORSTORE=pinecone requiere PINECONE_API_KEY "
                "(consíguela en https://app.pinecone.io)."
            )
        pc = Pinecone(api_key=settings.pinecone_api_key)
        nombre = settings.pinecone_index

        # Crea el índice serverless si no existe. La dimensión debe coincidir con el
        # modelo de embeddings (Cohere embed-multilingual-v3.0 = 1024); se calcula
        # embebiendo una cadena de prueba para no hardcodearla.
        existentes = [i["name"] for i in pc.list_indexes()]
        if nombre not in existentes:
            dimension = len(embeddings.embed_query("dimensión"))
            pc.create_index(
                name=nombre,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud, region=settings.pinecone_region
                ),
            )
            # El índice recién creado tarda unos segundos en quedar "listo".
            for _ in range(30):
                estado = pc.describe_index(nombre).status
                listo = getattr(estado, "ready", None)
                if listo is None and hasattr(estado, "get"):
                    listo = estado.get("ready")
                if listo:
                    break
                time.sleep(1)

        index = pc.Index(nombre)
        vs = PineconeVectorStore(index=index, embedding=embeddings)

        # Solo indexa si el índice está vacío o si se fuerza (evita re-subir en cada
        # arranque: los vectores ya persisten en la nube).
        stats = index.describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        if total is None:
            total = stats.get("total_vector_count", 0) if hasattr(stats, "get") else 0
        if forzar or not total:
            fragmentos = self._cargar_fragmentos(embeddings, docs_dir)
            vs.add_documents(fragmentos)
            return vs, len(fragmentos)
        return vs, total

    def indexar(self, docs_dir: str | None = None, forzar: bool = False) -> int:
        """Indexa los PDFs y arma retriever + cadena, según el backend elegido.

        **Strategy**: ``settings.rag_vectorstore`` selecciona la base vectorial
        (``faiss`` local/persistido o ``pinecone`` en la nube) sin tocar el resto del
        pipeline. Cambiar de una a otra es solo una variable de entorno.
        """
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        embeddings = self._embeddings()
        backends = {"faiss": self._vs_faiss, "pinecone": self._vs_pinecone}
        backend = settings.rag_vectorstore.lower()
        if backend not in backends:
            raise ValueError(
                f"RAG_VECTORSTORE inválido: '{backend}'. Usa 'faiss' o 'pinecone'."
            )
        vectorstore, n_fragmentos = backends[backend](embeddings, docs_dir, forzar)
        self._vectorstore = vectorstore  # para el fallback de recuperación

        # Retriever: por defecto `similarity` top-k (mejor recall). Si se configura un
        # umbral (>0) se usa `similarity_score_threshold` para descartar los flojos. Ojo:
        # en FAISS el score es distancia L2 (no cosine), por lo que un umbral alto puede
        # filtrar fragmentos válidos; por eso el umbral está desactivado por defecto.
        if settings.rag_score_threshold > 0:
            self.retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": settings.rag_score_threshold,
                    "k": settings.rag_top_k,
                },
            )
        else:
            self.retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": settings.rag_top_k}
            )
        # Cadena "stuff" (equivalente moderno de create_stuff_documents_chain en LCEL):
        # inserta los documentos recuperados en {context} y genera la respuesta.
        prompt_rag = ChatPromptTemplate(
            [
                ("system", SYSTEM_PROMPT_RAG),
                (
                    "human",
                    "{historial}Contexto de la guía:\n{context}\n\nPregunta actual: {input}",
                ),
            ]
        )
        self.document_chain = (
            prompt_rag | llm.construir_chat_model(temperature=0.2) | StrOutputParser()
        )
        return n_fragmentos

    def esta_listo(self) -> bool:
        return self.retriever is not None

    @staticmethod
    def _no_encontrado(respuesta: str = "No lo sé.") -> dict:
        return {"respuesta": respuesta, "citaciones": [], "documentos_encontrados": False}

    def _generar_consultas(self, pregunta: str) -> list[str]:
        """Multi-query: genera variantes de la pregunta con la LLM (incluye la original).

        Si la generación falla (p. ej. sin cupo), degrada con elegancia a la pregunta
        original, para que el RAG nunca se caiga por esta mejora opcional.
        """
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate

        plantilla = PromptTemplate.from_template(MULTIQUERY_TEMPLATE)
        cadena = plantilla | llm.construir_chat_model(temperature=0.4) | StrOutputParser()
        try:
            salida = cadena.invoke(
                {"pregunta": pregunta, "n": settings.rag_multiquery_n}
            )
        except Exception:  # noqa: BLE001 - cualquier fallo degrada a consulta simple
            return [pregunta]
        variantes = [
            linea.strip(" -*\"'[]") for linea in salida.splitlines() if linea.strip()
        ]
        variantes = [v for v in variantes if v and v.lower() != pregunta.lower()]
        return [pregunta] + variantes[: settings.rag_multiquery_n]

    def _recuperar(self, pregunta: str):
        """Recupera documentos; con multi-query une (dedup) los de todas las variantes.

        Robustez: si el retriever (p. ej. con umbral) no devuelve nada para una consulta,
        cae a una búsqueda ``similarity`` top-k directa sobre el vectorstore. Así un umbral
        mal configurado (o alto en FAISS) nunca deja al RAG sin contexto: el "No lo sé"
        pasa a decidirlo el LLM con el prompt estricto, no la falta de recuperación.
        """
        consultas = [pregunta]
        if settings.rag_multiquery:
            consultas = self._generar_consultas(pregunta)

        vistos, documentos = set(), []

        def _agregar(docs):
            for d in docs:
                clave = (
                    d.metadata.get("source"),
                    d.metadata.get("page"),
                    d.page_content[:80],
                )
                if clave not in vistos:
                    vistos.add(clave)
                    documentos.append(d)

        for consulta in consultas:
            docs = self.retriever.invoke(consulta)
            if not docs and self._vectorstore is not None:
                docs = self._vectorstore.similarity_search(consulta, k=settings.rag_top_k)
            _agregar(docs)

        # Última red de seguridad: si aún no hay nada, similarity puro sobre la pregunta.
        if not documentos and self._vectorstore is not None:
            _agregar(self._vectorstore.similarity_search(pregunta, k=settings.rag_top_k))
        return documentos

    def recuperar_para_stream(self, pregunta: str, contexto_conversacion: str = ""):
        """Prepara el streaming RAG: devuelve (inputs_para_la_cadena, documentos).

        Si no hay documentos relevantes, devuelve (None, []). El endpoint transmite la
        respuesta con ``self.document_chain.stream(inputs)`` y al final calcula las
        citaciones con ``citaciones_para``.
        """
        if not self.esta_listo():
            self.indexar()
        pregunta = (pregunta or "").strip()
        if not pregunta:
            return None, []
        documentos = self._recuperar(pregunta)
        if not documentos:
            return None, []
        contexto = "\n\n".join(d.page_content for d in documentos)
        historial = f"{contexto_conversacion}\n\n" if contexto_conversacion else ""
        return {"input": pregunta, "context": contexto, "historial": historial}, documentos

    def proceso_para(self, documentos) -> list:
        """Fragmentos recuperados (preview + página + fuente) para el acordeón 'Proceso'.

        Muestra al usuario QUÉ recuperó el RAG antes de redactar (transparencia), sin
        depender del solape con la respuesta (a diferencia de las citaciones finales).
        """
        return [
            {
                "preview": _limpiar(d.page_content)[:170] + "…",
                "pagina": (d.metadata.get("page", 0) or 0) + 1,
                "fuente": os.path.basename(d.metadata.get("source", "")),
            }
            for d in documentos
        ]

    def citaciones_para(self, respuesta: str, documentos) -> list:
        """Citaciones (fragmento + página + fuente) de los documentos que la respuesta usó."""
        if respuesta.strip().rstrip(".!?¡¿").lower() in ("no lo sé", "no lo se"):
            return []
        terminos_resp = _terminos(respuesta)
        usados = [
            d for d in documentos if len(terminos_resp & _terminos(d.page_content)) >= 2
        ]
        return [
            {
                "contenido": _limpiar(d.page_content)[:180] + "…",
                "pagina": (d.metadata.get("page", 0) or 0) + 1,
                "fuente": os.path.basename(d.metadata.get("source", "")),
            }
            for d in usados
        ]

    def preguntar(self, pregunta: str, contexto_conversacion: str = "") -> dict:
        """Recupera los chunks relevantes y genera la respuesta con citaciones.

        Args:
            pregunta: la pregunta del usuario.
            contexto_conversacion: memoria de la sesión (para entender preguntas de seguimiento).

        Returns:
            {"respuesta": str, "citaciones": list[str], "documentos_encontrados": bool}
        """
        if not self.esta_listo():
            self.indexar()  # indexación perezosa (en la primera consulta)

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return self._no_encontrado("Por favor, escribe una pregunta.")

        # 1) Recuperación (multi-query opcional): si nada supera el umbral -> "No lo sé".
        documentos = self._recuperar(pregunta)
        if not documentos:
            return self._no_encontrado()

        # 2) Generación: se "rellenan" (stuff) los documentos recuperados en el contexto,
        # anteponiendo el historial de la conversación para las preguntas de seguimiento.
        contexto = "\n\n".join(d.page_content for d in documentos)
        historial = f"{contexto_conversacion}\n\n" if contexto_conversacion else ""
        respuesta = self.document_chain.invoke(
            {"input": pregunta, "context": contexto, "historial": historial}
        )

        # 3) El modelo también puede decir "No lo sé" si el contexto no sirve.
        if respuesta.strip().rstrip(".!?¡¿").lower() in ("no lo sé", "no lo se"):
            return self._no_encontrado()

        # Citaciones trazables: SOLO los fragmentos que realmente comparten contenido con
        # la respuesta (>=2 términos significativos en común). Así, si el modelo responde
        # con su persona (p. ej. "¿quién eres?"), no se listan fuentes irrelevantes.
        terminos_resp = _terminos(respuesta)
        usados = [
            d for d in documentos if len(terminos_resp & _terminos(d.page_content)) >= 2
        ]
        citaciones = [
            {
                "contenido": _limpiar(d.page_content)[:180] + "…",
                "pagina": (d.metadata.get("page", 0) or 0) + 1,
                "fuente": os.path.basename(d.metadata.get("source", "")),
            }
            for d in usados
        ]
        return {
            "respuesta": respuesta,
            "citaciones": citaciones,
            "documentos_encontrados": True,
        }


# Instancia única (el índice se construye de forma perezosa en la primera pregunta).
rag = RagSantander()
