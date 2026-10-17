FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Tesseract reads scanned PDFs and photos of documents.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-bedrock.txt ./
ARG WITH_BEDROCK=false
RUN pip install -r requirements.txt \
 && if [ "$WITH_BEDROCK" = "true" ]; then pip install -r requirements-bedrock.txt; fi

COPY app ./app
COPY data ./data
COPY eval ./eval

RUN useradd --create-home nyaya && mkdir -p /app/var && chown nyaya /app/var
USER nyaya

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"
CMD ["uvicorn", "app.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
