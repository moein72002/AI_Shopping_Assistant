#!/bin/sh

# Ensure log file exists and has correct permissions
: > requests.log
chmod 666 requests.log

# Start FastAPI server in the background
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} &

# Start Streamlit UI in the foreground
# The server.address and server.port flags are important for Docker
streamlit run admin_ui.py --server.port 8501 --server.address 0.0.0.0
