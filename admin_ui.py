import streamlit as st
import json
import pandas as pd
from datetime import datetime

LOG_FILE = "requests.log"

st.set_page_config(layout="wide", page_title="Request Logs")

st.title("API Request Logs")

def load_logs():
    logs = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    # Ignore malformed lines
                    pass
    except FileNotFoundError:
        st.warning(f"Log file not found at: {LOG_FILE}")
    return logs

logs = load_logs()

if not logs:
    st.info("No logs to display yet. Make some requests to the `/chat` endpoint.")
else:
    # Reverse logs to show most recent first
    logs.reverse()

    # Create a DataFrame for a high-level overview
    display_data = []
    for log in logs:
        req_body = log.get("request_body", {})
        messages = req_body.get("messages", [])
        last_message = messages[-1].get("content", "N/A") if messages else "N/A"
        
        display_data.append({
            "Timestamp": log.get("timestamp", "N/A"),
            "Chat ID": req_body.get("chat_id", "N/A"),
            "Path": log.get("path", "N/A"),
            "Status": log.get("status_code", "N/A"),
            "Last Query": last_message
        })
    
    df = pd.DataFrame(display_data)
    # Avoid Streamlit's Arrow dependency by rendering as HTML
    st.markdown(df.to_html(index=False, escape=False), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Log Details")

    for i, log in enumerate(logs):
        timestamp = log.get('timestamp', 'No Timestamp')
        try:
            # Prettier timestamp format
            ts_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            ts_pretty = ts_obj.strftime('%Y-%m-%d %H:%M:%S UTC')
        except (ValueError, TypeError):
            ts_pretty = timestamp

        status = log.get('status_code', 'N/A')
        chat_id = log.get('request_body', {}).get('chat_id', 'N/A')

        with st.expander(f"**{ts_pretty}** | **{status}** | Chat ID: `{chat_id}`"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Request")
                st.json(log.get("request_body", {}))
            with col2:
                st.markdown("#### Response")
                st.json(log.get("response_body", {}))
