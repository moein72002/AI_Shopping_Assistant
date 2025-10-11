#!/bin/sh

# Ensure log file exists and has correct permissions
: > requests.log
chmod 666 requests.log

# Start API server (gunicorn with uvicorn workers) in the background
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:${PORT:-8080} --timeout 60 main:app &

# Start Admin Logs UI in the background
# The server.address and server.port flags are important for Docker
streamlit run admin_ui.py --server.port 8501 --server.address 127.0.0.1 --server.headless true &

# Start Tester Streamlit UI in the foreground
streamlit run streamlit_app.py --server.port 8502 --server.address 127.0.0.1 --server.headless true