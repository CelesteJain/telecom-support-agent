FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# CPU-only torch first: the default Linux wheel pulls about 2 GB of GPU libraries we cannot use
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
RUN python -m spacy download en_core_web_lg

COPY . .

# Build the search index, and download the two search models into the image
RUN python rag/ingest.py \
 && python -c "from rag.search import search; print(len(search('wifi keeps dropping')), 'results')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]