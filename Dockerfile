FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime dependency for faiss-cpu (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps with uv (faster, deterministic). Ensure uv is present.
COPY requirements.txt ./
RUN pip install --no-cache-dir uv && uv pip sync --system requirements.txt

# --- MODIFIED SECTION ---
# Include the Kaggle downloader directory so startup can locate it at runtime
COPY download_data_scripts ./download_data_scripts
COPY run_test_scripts ./run_test_scripts
COPY utils ./utils


# Copy the rest of the application files
COPY main.py ./
COPY admin_ui.py ./
COPY start.sh ./
COPY tools ./tools
COPY tests ./tests

# --- Healthcheck & Key Validation ---
# Run a simple test to confirm the app starts and is responsive
RUN uv pip install --system pytest requests python-dotenv && pytest tests/test_api.py::test_sanity_check_ping

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

RUN chmod +x ./start.sh

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]