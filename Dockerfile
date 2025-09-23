FROM python:3.11-slim

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
COPY download_data_scripts/download_data_from_kaggle.py ./

# Run the script to download the data using a build secret.
# This command doesn't change because the script was copied into the WORKDIR.
RUN --mount=type=secret,id=kaggle_env \
    python download_data_from_kaggle.py

# Move the downloaded database to the app's root directory
RUN mv ./shopping_dataset/torob.db ./torob.db
# --- END MODIFIED SECTION ---

# Copy the rest of the application files
COPY main.py ./
COPY tools ./tools

# Optional: uncomment if indexes are present in your workspace
# COPY vector_index.faiss ./vector_index.faiss
# COPY vector_index.faiss.json ./vector_index.faiss.json
# COPY vector_index_names.faiss ./vector_index_names.faiss
# COPY vector_index_names.faiss.json ./vector_index_names.faiss.json

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
