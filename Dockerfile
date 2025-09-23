FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime dependency for faiss-cpu (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps. Make sure requirements.txt includes "kaggle" and "python-dotenv"
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- MODIFIED SECTION ---
# Copy the download script from its subfolder into the image's WORKDIR
# Copy the Kaggle download script
COPY download_data_scripts/download_data_from_kaggle.py ./

# Attempt Kaggle download if build secret is provided; otherwise skip
RUN --mount=type=secret,id=kaggle_env \
    sh -c 'if [ -f /run/secrets/kaggle_env ]; then \
      echo "Kaggle secret detected. Downloading dataset..." && \
      python download_data_from_kaggle.py && \
      mv ./shopping_dataset/torob.db ./torob.db; \
    else \
      echo "ERROR: No kaggle_env secret found. Set build secret to download torob.db." 1>&2; \
      exit 1; \
    fi'
# --- END MODIFIED SECTION ---

# Copy the rest of the application files
COPY main.py ./
COPY tools ./tools
COPY tests ./tests

# --- Healthcheck & Key Validation ---
# Run a simple test to confirm the app starts and is responsive
RUN pip install pytest requests python-dotenv && pytest tests/test_api.py::test_sanity_check_ping

# Optional: To validate the OpenAI key during build, you would need to
# pass the key as a secret and run a test that makes a simple API call.
# RUN --mount=type=secret,id=openai_key,dst=/app/.env pytest tests/test_api.py::test_openai_key_is_valid

# Optional: uncomment if indexes are present in your workspace
# COPY vector_index.faiss ./vector_index.faiss
# COPY vector_index.faiss.json ./vector_index.faiss.json
# COPY vector_index_names.faiss ./vector_index_names.faiss
# COPY vector_index_names.faiss.json ./vector_index_names.faiss.json

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
