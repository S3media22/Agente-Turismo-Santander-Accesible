"""Capa de proveedor de LLM con **LangChain** (multi-proveedor con fallback).

Usa los chat models de LangChain (`ChatGroq`, `ChatGoogleGenerativeAI`), plantillas
de prompt (`ChatPromptTemplate`) y cadenas **LCEL** (`prompt | modelo | parser`).

Permite responder con **Gemini**, **Groq** (Llama/Gemma) o **Cohere** según
`LLM_PROVIDER`. Además, aplica una **estrategia de respaldo**: si un modelo/proveedor
se queda sin cupo (error 429) o falla, se intenta con el siguiente candidato de la
cadena (primero el proveedor activo con sus modelos, luego los demás que tengan
API key), sin romper la experiencia del usuario. Por ejemplo, con Groq activo:

    [Groq 70b] -> [Groq 8b] -> [Groq gemma] -> [Gemini] -> [Cohere] -> (sin cupo)

Cada modelo de Groq tiene su propia cuota diaria, así que probar otro modelo suele
resolver el agotamiento de tokens. Si TODOS los candidatos fallan, se lanza
`SinCupoError`, que la aplicación traduce en un mensaje amable.
"""
from __future__ import annotations

import time
from threading import Lock

from .config import settings

# Códigos HTTP transitorios del servidor que conviene reintentar en el mismo modelo.
_TRANSIENT = {500, 502, 503}

# --------------------------------------------------------------------- #
# Rate limiting por proveedor (para no superar el RPM del free tier)
# --------------------------------------------------------------------- #
# Un limitador COMPARTIDO por proveedor: todos los modelos de ese proveedor (incluido
# el ReAct agent, que encadena varias llamadas) pasan por el mismo "cubo" de tokens, así
# el ritmo global se mantiene bajo el límite por minuto. El limitador ESPACIA las
# llamadas (bloquea unos ms), no las descarta, así que evita el 429 sin perder respuestas.
_rate_limiters: dict[str, object] = {}
_rate_lock = Lock()


def _rate_limiter_para(proveedor: str):
    """Devuelve el rate limiter compartido del proveedor (o None si está desactivado)."""
    rpm = {
        "gemini": settings.gemini_rpm,
        "groq": settings.groq_rpm,
        "cohere": settings.cohere_rpm,
    }.get(proveedor, 0)
    if rpm <= 0:
        return None
    with _rate_lock:
        lim = _rate_limiters.get(proveedor)
        if lim is None:
            from langchain_core.rate_limiters import InMemoryRateLimiter

            bucket = max(2, rpm // 3)
            lim = InMemoryRateLimiter(
                requests_per_second=rpm / 60.0,
                check_every_n_seconds=0.1,
                # Permite una ráfaga (p. ej. un turno del agente o varias preguntas
                # seguidas) y luego regula; mantenerlo bajo respeta el tope por minuto.
                max_bucket_size=bucket,
            )
            # Pre-carga el bucket: así la PRIMERA petición (app ociosa) sale al instante
            # y no espera ~5s por el token. El límite por minuto se sigue respetando bajo
            # carga sostenida. Sin esto, el streaming parecía "congelado" antes del 1er token.
            lim.available_tokens = float(bucket)
            _rate_limiters[proveedor] = lim
    return lim


class SinCupoError(Exception):
    """Todos los proveedores/modelos disponibles fallaron o agotaron su cuota."""


def _codigo_http(exc: Exception) -> int | None:
    """Intenta extraer un código HTTP del error de cualquier proveedor."""
    for attr in ("code", "status_code"):
        valor = getattr(exc, attr, None)
        if isinstance(valor, int):
            return valor
    resp = getattr(exc, "response", None)
    if resp is not None and isinstance(getattr(resp, "status_code", None), int):
        return resp.status_code
    return None


# --------------------------------------------------------------------- #
# Proveedores concretos con LangChain (chat model + prompt + cadena LCEL)
# --------------------------------------------------------------------- #
def _invocar_cadena(modelo_chat, mensaje: str, system_instruction: str) -> str:
    """Arma una cadena LCEL (prompt | modelo | parser) y la ejecuta.

    Se usan variables ({sistema}, {entrada}) para que el contenido del documento
    no se interprete como marcadores de la plantilla.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", "{sistema}"), ("human", "{entrada}")]
    )
    cadena = prompt | modelo_chat | StrOutputParser()
    texto = cadena.invoke({"sistema": system_instruction, "entrada": mensaje})
    return (texto or "").strip()


def _generar_gemini(mensaje: str, system_instruction: str, max_tokens: int, modelo: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI

    chat = ChatGoogleGenerativeAI(
        model=modelo,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
        max_output_tokens=max_tokens,
        rate_limiter=_rate_limiter_para("gemini"),
    )
    return _invocar_cadena(chat, mensaje, system_instruction)


def _generar_groq(mensaje: str, system_instruction: str, max_tokens: int, modelo: str) -> str:
    from langchain_groq import ChatGroq

    if not settings.groq_api_key:
        raise ValueError("Falta GROQ_API_KEY.")
    # Groq (Llama) no "piensa"; con ~1024 tokens sobra y no se trunca. Mantenerlo
    # bajo evita superar el límite de tokens por minuto del free tier.
    chat = ChatGroq(
        model=modelo,
        api_key=settings.groq_api_key,
        temperature=0.2,
        max_tokens=min(max_tokens, 1024),
        rate_limiter=_rate_limiter_para("groq"),
    )
    return _invocar_cadena(chat, mensaje, system_instruction)


def _generar_cohere(mensaje: str, system_instruction: str, max_tokens: int, modelo: str) -> str:
    from langchain_cohere import ChatCohere

    if not settings.cohere_api_key:
        raise ValueError("Falta COHERE_API_KEY.")
    chat = ChatCohere(
        model=modelo,
        cohere_api_key=settings.cohere_api_key,
        temperature=0.2,
        rate_limiter=_rate_limiter_para("cohere"),
    )
    return _invocar_cadena(chat, mensaje, system_instruction)


_GENERADORES = {"gemini": _generar_gemini, "groq": _generar_groq, "cohere": _generar_cohere}


# --------------------------------------------------------------------- #
# Construcción de la cadena de intentos (estrategia de respaldo)
# --------------------------------------------------------------------- #
def _modelos_groq() -> list[str]:
    """Modelo de Groq configurado + alternativas (cada una con su cuota diaria)."""
    modelos = [settings.groq_model]
    for m in settings.groq_fallback_models.split(","):
        m = m.strip()
        if m and m not in modelos:
            modelos.append(m)
    return modelos


def _candidatos(proveedor: str) -> list[tuple[str, str]]:
    """(proveedor, modelo) disponibles de un proveedor, si tiene su API key."""
    if proveedor == "groq" and settings.groq_api_key:
        return [("groq", m) for m in _modelos_groq()]
    if proveedor == "gemini" and settings.gemini_api_key:
        return [("gemini", settings.chat_model)]
    if proveedor == "cohere" and settings.cohere_api_key:
        return [("cohere", settings.cohere_model)]
    return []


def _cadena_intentos() -> list[tuple[str, str]]:
    """Cadena de (proveedor, modelo): primero el proveedor activo, luego el resto."""
    prov = settings.llm_provider.lower()
    orden = [prov] + [p for p in ("groq", "gemini", "cohere") if p != prov]
    cadena: list[tuple[str, str]] = []
    for proveedor in orden:
        cadena += _candidatos(proveedor)
    return cadena


# --------------------------------------------------------------------- #
# Chat models de LangChain (para cadenas y agentes que necesitan el objeto)
# --------------------------------------------------------------------- #
def _construir_modelo(proveedor: str, modelo: str, temperature: float = 0.2):
    """Crea un chat model de LangChain para un (proveedor, modelo) concreto."""
    limiter = _rate_limiter_para(proveedor)
    if proveedor == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=modelo,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
            max_output_tokens=settings.max_output_tokens,
            rate_limiter=limiter,
            # Gemini 2.5 "piensa" antes de responder (~30s de latencia al primer token).
            # Para un chat turístico no aporta y arruina el streaming: se desactiva.
            thinking_budget=0,
            # Failover rápido: si está sin cupo (429) no reintenta 30s, cae al siguiente.
            max_retries=0,  # 429 = sin cupo: no reintentar, saltar YA al siguiente candidato
        )
    if proveedor == "cohere":
        from langchain_cohere import ChatCohere

        return ChatCohere(
            model=modelo,
            cohere_api_key=settings.cohere_api_key,
            temperature=temperature,
            rate_limiter=limiter,
            max_retries=0,  # 429 = sin cupo: no reintentar, saltar YA al siguiente candidato
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=modelo,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=min(settings.max_output_tokens, 1024),
        rate_limiter=limiter,
        max_retries=0,  # 429 = sin cupo: no reintentar, saltar YA al siguiente candidato
    )


def construir_chat_model(temperature: float = 0.2, con_respaldo: bool = True):
    """Devuelve un chat model de LangChain para el proveedor activo.

    Con `con_respaldo=True` encadena el resto de proveedores con `.with_fallbacks()`
    (útil para cadenas simples). Para agentes que usan `bind_tools`, pasar
    `con_respaldo=False` (un solo modelo).
    """
    cadena = _cadena_intentos()
    if not cadena:
        raise SinCupoError("no hay proveedores configurados")
    modelos = [_construir_modelo(p, m, temperature) for p, m in cadena]
    principal, resto = modelos[0], modelos[1:]
    return principal.with_fallbacks(resto) if (con_respaldo and resto) else principal


def construir_modelo_con_tools(tools, temperature: float = 0):
    """Modelo con `tools` atadas y **respaldo entre proveedores**.

    `bind_tools` necesita un modelo concreto (no acepta `RunnableWithFallbacks`), así que
    atamos las herramientas a CADA proveedor disponible y luego combinamos los modelos
    atados con `.with_fallbacks()`. Si el proveedor activo se queda sin cupo (429), el
    agente cae al siguiente — igual que el resto del sistema.
    """
    cadena = _cadena_intentos()
    if not cadena:
        raise SinCupoError("no hay proveedores configurados")
    atados = [_construir_modelo(p, m, temperature).bind_tools(tools) for p, m in cadena]
    principal, resto = atados[0], atados[1:]
    return principal.with_fallbacks(resto) if resto else principal


def stream_texto(mensaje: str, system_instruction: str, temperature: float = 0.2):
    """Genera texto en **streaming**: hace `yield` de cada fragmento a medida que llega.

    Usa la cadena con respaldo (`.with_fallbacks()`); si el primer proveedor falla al
    iniciar, se cae al siguiente. Pensado para respuestas fluidas token a token.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", "{sistema}"), ("human", "{entrada}")]
    )
    cadena = prompt | construir_chat_model(temperature=temperature) | StrOutputParser()
    for fragmento in cadena.stream({"sistema": system_instruction, "entrada": mensaje}):
        if fragmento:
            yield fragmento


# --------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------- #
def _intentar_modelo(generar, mensaje, system_instruction, max_tokens, modelo, reintentos=2):
    """Intenta un modelo, reintentando solo ante errores transitorios del servidor."""
    for intento in range(1, reintentos + 1):
        try:
            return generar(mensaje, system_instruction, max_tokens, modelo)
        except Exception as exc:  # noqa: BLE001
            codigo = _codigo_http(exc)
            if codigo in _TRANSIENT and intento < reintentos:
                time.sleep(1.5 * intento)
                continue
            raise  # 429/4xx u otro: se maneja arriba pasando al siguiente candidato


def generar_texto(mensaje: str, system_instruction: str, max_tokens: int) -> str:
    """Genera texto probando la cadena de proveedores/modelos hasta que uno responda."""
    errores: list[str] = []
    for proveedor, modelo in _cadena_intentos():
        generar = _GENERADORES.get(proveedor)
        if generar is None:
            continue
        try:
            return _intentar_modelo(generar, mensaje, system_instruction, max_tokens, modelo)
        except Exception as exc:  # noqa: BLE001 - se registra y se pasa al siguiente
            codigo = _codigo_http(exc)
            etiqueta = "sin cupo (429)" if codigo == 429 else f"error {codigo or '?'}"
            errores.append(f"{proveedor}/{modelo}: {etiqueta}")
            continue

    raise SinCupoError(" | ".join(errores) or "no hay proveedores configurados")
