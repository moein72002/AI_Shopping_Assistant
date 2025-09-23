# --- Stage 1: DB Prepper ---
# This stage's only job is to provide a context for torob.db if it exists.
# If torob.db does not exist in the build context, the build will not fail.
FROM busybox:latest AS db_prepper
# COPY torob.db /data/

# --- Stage 2: Main Application ---
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime dependency for faiss-cpu (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Conditionally copy torob.db from the prepper stage.
# This command will not fail if /data/torob.db does not exist in the prepper stage.
COPY --from=db_prepper /data/torob.db ./torob.db

# Install deps. Make sure requirements.txt includes "kaggle" and "python-dotenv"
COPY requirements.txt ./
RUN pip install uv && uv pip sync --system requirements.txt

# --- MODIFIED SECTION ---
# Include the Kaggle downloader directory so startup can locate it at runtime
# torob.db is NOT copied; it will be downloaded at runtime if missing.
COPY download_data_scripts ./download_data_scripts


# Copy the rest of the application files
COPY main.py ./
COPY tools ./tools
COPY tests ./tests

# --- Healthcheck & Key Validation ---
# Run a simple test to confirm the app starts and is responsive
# All dependencies including pytest are already installed by the `uv pip sync` command above.
RUN pytest tests/test_api.py::test_sanity_check_ping

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

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
