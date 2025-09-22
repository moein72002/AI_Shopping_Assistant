FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime dependency for faiss-cpu (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps separately for better cache utilization
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only required application files
COPY main.py ./
COPY tools ./tools
COPY torob.db ./torob.db

# Optional: uncomment if indexes are present in your workspace
# COPY vector_index.faiss ./vector_index.faiss
# COPY vector_index.faiss.json ./vector_index.faiss.json
# COPY vector_index_names.faiss ./vector_index_names.faiss
# COPY vector_index_names.faiss.json ./vector_index_names.faiss.json

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
