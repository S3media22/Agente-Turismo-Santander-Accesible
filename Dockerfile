# Imagen para desplegar el Agente Turístico de Santander en OCI.
FROM python:3.11-slim

WORKDIR /app

# Dependencias primero (mejor cacheo de capas).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación.
COPY app ./app
COPY static ./static
COPY data ./data

EXPOSE 8000

# La API key del proveedor de LLM (GEMINI/GROQ/COHERE) se pasa como variable de entorno.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
