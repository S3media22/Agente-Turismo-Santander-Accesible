"""Agente de análisis de datos (CSV) con LangChain + PythonAstREPLTool.

Curso "Automatizando el análisis de datos con agentes". El flujo es:

  1. Al LLM se le da la lista de columnas del DataFrame (`df.columns.to_list()`, más
     legible que `head().to_markdown()`) y escribe **código pandas** (separado por ';').
  2. La herramienta **PythonAstREPLTool** EJECUTA ese código de verdad sobre el `df`
     (no alucina el resultado): el LLM genera, Python calcula.
  3. El LLM redacta una **respuesta en lenguaje natural** con el resultado real.

La generación del código usa `bind_tools` (tool-calling) + `JsonOutputKeyToolsParser`
como en el curso, con **respaldo entre proveedores**; si el tool-calling no está
disponible (cuota/proveedor), degrada a generar el código como texto. En ambos casos el
código se ejecuta con la herramienta REPL.

⚠️ SEGURIDAD: ejecuta código Python generado por el LLM. Por eso el endpoint que lo
expone está protegido con ADMIN_KEY y NO debería abrirse al público sin un sandbox.
"""
from __future__ import annotations

import re
from typing import Literal

from . import llm
from .config import settings

# Catálogo de herramientas para el router: describe cuándo usar cada una (como las
# descripciones de las Tool del curso), para que el agente elija la adecuada.
SYSTEM_ROUTER = (
    "Eres un enrutador de herramientas de análisis de datos sobre un DataFrame. Según la "
    "solicitud del usuario, elige EXACTAMENTE una herramienta:\n"
    "- explorar: panorama general del DataFrame (nº de filas/columnas, tipos, nulos, "
    "duplicados). Ej.: 'dame información general', 'describe el dataset'.\n"
    "- estadisticas: resumen estadístico descriptivo COMPLETO (media, desviación, min, "
    "max, cuartiles de varias columnas). Ej.: 'muéstrame la estadística descriptiva'.\n"
    "- grafico: cuando pide una VISUALIZACIÓN. Ej.: 'crea un gráfico', 'plotea', "
    "'visualiza', 'muestra la distribución de...'.\n"
    "- pregunta: cálculos o consultas PUNTUALES (un promedio, un conteo, una correlación, "
    "un filtro, un top-N). Ej.: '¿cuántas consultas por sesión?', '¿cuál es el promedio de X?'."
)

SYSTEM_CODIGO = (
    "Tienes acceso a un dataframe pandas llamado `df` (pandas ya está importado como pd).\n"
    "Estas son sus columnas (`df.columns.to_list()`):\n```\n{columnas}\n```\n\n"
    "El dataframe `df` YA está cargado en memoria con datos reales. NUNCA lo recrees ni "
    "redefinas: no uses `pd.read_csv` ni `pd.DataFrame(...)`; usa el `df` existente tal cual.\n"
    "Dada una pregunta del usuario, escribe el código Python para responderla.\n"
    "Cada comando Python generado SIEMPRE debe estar separado por ';'.\n"
    "Usa ÚNICAMENTE las bibliotecas incorporadas de Python y pandas.\n"
    "El código debe imprimir (print) el resultado.\n"
    "Retorna ÚNICAMENTE el código Python, sin explicaciones."
)

SYSTEM_RESPUESTA = (
    "Eres un analista de datos. El código pandas YA FUE EJECUTADO sobre el dataframe y su "
    "RESULTADO REAL se te entrega a continuación. Redacta en español una respuesta clara y "
    "natural usando EXACTAMENTE las cifras de ese resultado. No recalcules, no supongas, no "
    "digas que no puedes ejecutar código: el resultado ya está calculado. No inventes "
    "unidades de medida que no aparezcan en la pregunta o el resultado. Sé conciso."
)

# --- Herramientas personalizadas (curso): exploradora y estadística ---
# Los metadatos se calculan en Python (reales) y el LLM redacta el informe.
PROMPT_EXPLORADORA = """Eres un analista de datos encargado de presentar un resumen informativo sobre un DataFrame.

================= INFORMACIÓN DEL DATAFRAME =================

Dimensiones: {shape}

Columnas y tipos de datos:
{columns}

Valores nulos por columna:
{nulos}

Cadenas 'nan' (en cualquier capitalización) por columna:
{nans_str}

Filas duplicadas: {duplicados}

============================================================

Con base en esta información, redacta un resumen claro y organizado en español que contenga:

1. Un título: ## Reporte de información general sobre el dataset
2. La dimensión total del DataFrame
3. La descripción de cada columna (nombre, tipo de dato y qué representa)
4. Las columnas con datos nulos y su cantidad
5. Las columnas con cadenas 'nan' y su cantidad
6. La existencia (o no) de datos duplicados
7. Un párrafo sobre los análisis que se pueden realizar con estos datos
8. Un párrafo sobre los tratamientos que se pueden aplicar a los datos
"""

PROMPT_ESTADISTICA = """Eres un analista de datos encargado de interpretar resultados estadísticos de una base de datos.

================= ESTADÍSTICAS DESCRIPTIVAS =================

{resumen}

============================================================

Con base en estos datos, elabora en español un resumen explicativo con lenguaje claro y fluido que incluya:

1. Un título: ## Informe de estadísticas descriptivas
2. Una visión general de las estadísticas de las columnas numéricas
3. Un párrafo sobre cada columna, comentando sus valores
4. Identificación de posibles valores atípicos según el mínimo y el máximo
5. Recomendaciones de próximos pasos en el análisis según los patrones identificados
"""

PROMPT_VISUAL = """Eres un especialista en visualización de datos. Tu tarea es generar ÚNICAMENTE el código Python para graficar con base en la solicitud del usuario.

## Solicitud del usuario:
"{pregunta}"

## Metadatos del DataFrame (ya existe como `df`):
{columnas}

## Muestra de los datos (3 primeras filas):
{muestra}

## Instrucciones obligatorias:
1. Usa `matplotlib.pyplot` (como `plt`) y `seaborn` (como `sns`); ya están importados.
2. Define el tema con `sns.set_theme()`.
3. Usa solo columnas que existan en el DataFrame `df`. NUNCA recrees `df`.
4. Elige el tipo de gráfico adecuado: histplot/kdeplot/boxplot/violinplot (distribución numérica),
   countplot (categórica), barplot (comparación entre categorías), scatterplot (relación), lineplot (series).
5. Configura el tamaño con `figsize=(8, 4)`.
6. Añade título y etiquetas de ejes apropiadas.
7. Posiciona el título a la izquierda con `loc='left'`, `pad=20`, `fontsize=14`.
8. Mantén los ticks del eje X sin rotación con `plt.xticks(rotation=0)`.
9. Elimina los bordes superior y derecho con `sns.despine()`.
10. Finaliza con `plt.show()`.

Devuelve ÚNICAMENTE el código Python separando cada instrucción con ';', sin texto adicional."""

# Proveedores en orden de preferencia para tool-calling (Groq tiene el mejor soporte).
_CANDIDATOS_TOOLS = ("groq", "gemini", "cohere")


def _limpiar_codigo(texto: str) -> str:
    """Extrae el código de una respuesta de texto (quita ```python ... ``` si viene)."""
    t = (texto or "").strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1)
    return t.strip()


# Sentencias que RECREAN el df (deben descartarse para no analizar datos inventados).
_REGROUND = re.compile(r"^\s*df\s*=\s*pd\.(DataFrame|read_csv)", re.I)
_READ_CSV = re.compile(r"pd\.read_csv", re.I)


def _sanear_codigo(codigo: str) -> str:
    """Descarta sentencias que recrean el DataFrame (df = pd.DataFrame/read_csv).

    Así el código siempre opera sobre el `df` real cargado, aunque el LLM intente
    redefinirlo con datos de ejemplo.
    """
    partes = re.split(r"[;\n]", codigo)
    conservadas = []
    for parte in partes:
        s = parte.strip()
        if not s or _REGROUND.match(s) or _READ_CSV.search(s):
            continue
        conservadas.append(s)
    return "; ".join(conservadas)


class AgenteDatos:
    """Responde preguntas sobre un DataFrame generando y ejecutando código pandas."""

    def _modelo_ids(self):
        ids = {
            "groq": settings.groq_model,
            "gemini": settings.chat_model,
            "cohere": settings.cohere_model,
        }
        keys = {
            "groq": settings.groq_api_key,
            "gemini": settings.gemini_api_key,
            "cohere": settings.cohere_api_key,
        }
        return [(p, ids[p]) for p in _CANDIDATOS_TOOLS if keys[p]]

    def _codigo_con_tools(self, df, pregunta: str) -> str:
        """Genera el código vía tool-calling (fiel al curso), con respaldo de proveedores."""
        from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_experimental.tools import PythonAstREPLTool

        disponibles = self._modelo_ids()
        if not disponibles:
            raise llm.SinCupoError("no hay proveedor con tool-calling configurado")

        herramienta = PythonAstREPLTool(locals={"df": df})
        atados = [
            llm._construir_modelo(p, m, temperature=0).bind_tools(
                [herramienta], tool_choice=herramienta.name
            )
            for p, m in disponibles
        ]
        modelo = atados[0].with_fallbacks(atados[1:]) if len(atados) > 1 else atados[0]

        parser = JsonOutputKeyToolsParser(key_name=herramienta.name, first_tool_only=True)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_CODIGO), ("human", "{pregunta}")]
        )
        args = (prompt | modelo | parser).invoke(
            {"columnas": df.columns.to_list(), "pregunta": pregunta}
        )
        return args.get("query", "") if isinstance(args, dict) else ""

    def _codigo_como_texto(self, df, pregunta: str) -> str:
        """Respaldo: genera el código como texto con la cadena con respaldo (sin tools)."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_CODIGO), ("human", "{pregunta}")]
        )
        texto = (
            prompt | llm.construir_chat_model(temperature=0) | StrOutputParser()
        ).invoke({"columnas": df.columns.to_list(), "pregunta": pregunta})
        return _limpiar_codigo(texto)

    def _redactar(self, instruccion: str) -> str:
        """Pasa una instrucción ya construida al LLM (con respaldo) y devuelve el texto."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([("human", "{contenido}")])
        texto = (
            prompt | llm.construir_chat_model(temperature=0.3) | StrOutputParser()
        ).invoke({"contenido": instruccion})
        return (texto or "").strip()

    def reporte_general(self, df) -> str:
        """Herramienta exploradora: panorama general del DataFrame (metadatos reales)."""
        shape = f"{df.shape[0]} filas x {df.shape[1]} columnas"
        columns = "\n".join(f"- {c}: {df[c].dtype}" for c in df.columns)
        nulos_serie = df.isnull().sum()
        nulos = "\n".join(f"- {c}: {int(n)}" for c, n in nulos_serie.items() if n) or "Ninguna"
        nans = {}
        for c in df.columns:
            if df[c].dtype == object:
                cnt = int(df[c].astype(str).str.fullmatch(r"(?i)nan").sum())
                if cnt:
                    nans[c] = cnt
        nans_str = "\n".join(f"- {c}: {n}" for c, n in nans.items()) or "Ninguna"
        duplicados = int(df.duplicated().sum())
        instruccion = PROMPT_EXPLORADORA.format(
            shape=shape, columns=columns, nulos=nulos, nans_str=nans_str, duplicados=duplicados
        )
        return self._redactar(instruccion)

    def reporte_estadistico(self, df) -> str:
        """Herramienta estadística: interpreta df.describe() de las columnas numéricas."""
        numericas = df.select_dtypes(include="number")
        if numericas.empty:
            return "El dataset no tiene columnas numéricas para un resumen estadístico."
        resumen = numericas.describe().to_string()
        return self._redactar(PROMPT_ESTADISTICA.format(resumen=resumen))

    def generar_grafico(self, df, pregunta: str) -> dict:
        """Herramienta visual: el LLM genera código matplotlib/seaborn y se ejecuta.

        Returns: {"codigo": str, "imagen_base64": str} (PNG). Ejecuta código de graficado
        generado por el LLM: el endpoint está protegido con ADMIN_KEY.
        """
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")  # backend sin ventana (servidor)
        import matplotlib.pyplot as plt
        import seaborn as sns

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {"codigo": "", "imagen_base64": "", "error": "Escribe qué gráfico quieres."}

        columnas = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        instruccion = PROMPT_VISUAL.format(
            pregunta=pregunta, columnas=columnas, muestra=df.head(3).to_string()
        )
        # Genera el código como texto (con respaldo de proveedores) y lo sanea.
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([("human", "{contenido}")])
        codigo_bruto = (
            prompt | llm.construir_chat_model(temperature=0) | StrOutputParser()
        ).invoke({"contenido": instruccion})
        codigo = _sanear_codigo(_limpiar_codigo(codigo_bruto))
        if not codigo:
            return {"codigo": "", "imagen_base64": "", "error": "No pude generar el gráfico."}

        # Ejecuta el código de graficado y captura la figura como PNG en base64.
        import pandas as pd

        plt.close("all")
        entorno = {"df": df, "plt": plt, "sns": sns, "pd": pd}
        try:
            exec(codigo, entorno)  # noqa: S102 - código de graficado; endpoint gated
            figura = plt.gcf()
            buffer = io.BytesIO()
            figura.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
            imagen = base64.b64encode(buffer.getvalue()).decode()
            return {"codigo": codigo, "imagen_base64": imagen}
        except Exception as exc:  # noqa: BLE001
            return {"codigo": codigo, "imagen_base64": "", "error": f"Error al graficar: {exc}"}
        finally:
            plt.close("all")

    def analizar(self, df, pregunta: str) -> dict:
        """Returns: {"codigo": str, "resultado": str, "respuesta": str}."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_experimental.tools import PythonAstREPLTool

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {"codigo": "", "resultado": "", "respuesta": "Escribe una pregunta sobre los datos."}

        # 1) Generar el código: primero tool-calling (curso); si falla, como texto.
        try:
            codigo = self._codigo_con_tools(df, pregunta)
        except Exception:  # noqa: BLE001 - degradamos a generación por texto
            codigo = self._codigo_como_texto(df, pregunta)
        # Descarta cualquier intento de recrear el df (analizar datos inventados).
        codigo = _sanear_codigo(codigo)
        if not codigo:
            return {"codigo": "", "resultado": "", "respuesta": "No pude generar el análisis."}

        # 2) Ejecución REAL del código sobre el DataFrame (no se alucina el resultado).
        herramienta = PythonAstREPLTool(locals={"df": df})
        resultado = str(herramienta.invoke(codigo)).strip()

        # 3) Respuesta en lenguaje natural con el resultado (con respaldo de proveedores).
        prompt_resp = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_RESPUESTA),
                (
                    "human",
                    "Pregunta: {pregunta}\n\nCódigo ejecutado:\n{codigo}\n\n"
                    "Resultado:\n{resultado}",
                ),
            ]
        )
        respuesta = (
            prompt_resp | llm.construir_chat_model(temperature=0.2) | StrOutputParser()
        ).invoke({"pregunta": pregunta, "codigo": codigo, "resultado": resultado})

        return {"codigo": codigo, "resultado": resultado, "respuesta": respuesta.strip()}


    # ------------------------------------------------------------------ #
    # Router / catálogo de herramientas (cierre del curso): el agente elige
    # sola qué herramienta usar según la solicitud. En vez del ReAct de
    # string-parsing (frágil), clasificamos con salida estructurada (Literal),
    # como el triaje del grafo — determinista y con respaldo de proveedores.
    # ------------------------------------------------------------------ #
    def _clasificar(self, pregunta: str) -> str:
        """Elige la herramienta: explorar | estadisticas | grafico | pregunta."""
        from pydantic import BaseModel, Field

        class RutaDatos(BaseModel):
            herramienta: Literal["explorar", "estadisticas", "grafico", "pregunta"] = Field(
                description="La herramienta que mejor atiende la solicitud del usuario"
            )

        disponibles = self._modelo_ids()
        if not disponibles:
            return "pregunta"
        modelos = [
            llm._construir_modelo(p, m, temperature=0).with_structured_output(RutaDatos)
            for p, m in disponibles
        ]
        router = modelos[0].with_fallbacks(modelos[1:]) if len(modelos) > 1 else modelos[0]
        try:
            ruta = router.invoke(
                [("system", SYSTEM_ROUTER), ("human", pregunta)]
            )
            return ruta.herramienta
        except Exception:  # noqa: BLE001 - ante cualquier fallo, cálculo puntual
            return "pregunta"

    def responder(self, df, pregunta: str) -> dict:
        """Punto de entrada del agente: enruta a la herramienta adecuada y ejecuta."""
        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {"herramienta": "pregunta", "respuesta": "Escribe una pregunta sobre los datos."}
        herramienta = self._clasificar(pregunta)
        if herramienta == "explorar":
            return {"herramienta": "explorar", "respuesta": self.reporte_general(df)}
        if herramienta == "estadisticas":
            return {"herramienta": "estadisticas", "respuesta": self.reporte_estadistico(df)}
        if herramienta == "grafico":
            return {"herramienta": "grafico", **self.generar_grafico(df, pregunta)}
        return {"herramienta": "pregunta", **self.analizar(df, pregunta)}


# Instancia única reutilizable (sin estado propio; el df se pasa en cada llamada).
agente_datos = AgenteDatos()
