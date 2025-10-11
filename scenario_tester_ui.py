import os
import json
import time
import random
import re
from typing import Any, Dict, List, Tuple

import requests
import pandas as pd
import streamlit as st


def find_scenario_files() -> List[str]:
    """Return a list of available scenario JSON files under server_tests/."""
    root = os.getcwd()
    scen_dir = os.path.join(root, "server_tests")
    if not os.path.isdir(scen_dir):
        return []
    files = []
    for name in sorted(os.listdir(scen_dir)):
        if name.endswith(".json"):
            files.append(os.path.join("server_tests", name))
    return files


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return []


def item_to_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    chat_id = item.get("Chat ID") or item.get("chat_id") or f"ui-{random.randrange(10**8):08d}"
    if "messages" in item and isinstance(item["messages"], list) and item["messages"]:
        messages = item["messages"]
    else:
        last = item.get("Last Query") or "ping"
        messages = [{"type": "text", "content": last}]
    return {"chat_id": chat_id, "messages": messages}


def post_chat(base_url: str, payload: Dict[str, Any], timeout: int = 30) -> Tuple[int, Any, float]:
    url = base_url.rstrip("/") + "/chat"
    started_at = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - started_at
    try:
        data = r.json()
    except ValueError:
        data = r.text
    return r.status_code, data, elapsed


def parse_expected_bases(expected: str) -> List[str]:
    """Parse patterns like "base=['abc','def']" from scenario logs."""
    if not isinstance(expected, str):
        return []
    m = re.search(r"base=\[(.*)\]", expected)
    if not m:
        return []
    inner = m.group(1)
    parts = [p.strip().strip("'") for p in inner.split(",") if p.strip()]
    return [p for p in parts if p]


st.set_page_config(layout="wide", page_title="Scenario Tester /chat")
st.title("/chat Scenario Tester")

with st.sidebar:
    st.header("Settings")
    default_base = os.environ.get("BASE_URL", "http://localhost:8080")
    base_url = st.text_input("Service Base URL", value=default_base)
    timeout = st.number_input("HTTP Timeout (s)", min_value=5, max_value=120, value=30, step=1)
    scenario_files = find_scenario_files()
    scenario_path = st.selectbox("Scenario file", options=scenario_files, index=0 if scenario_files else -1)
    mode = st.radio("Mode", options=["Single", "Batch"], index=0, horizontal=True)

if not scenario_files:
    st.warning("No scenario files found in `server_tests/`.")
    st.stop()

items = load_items(scenario_path)
st.caption(f"Loaded {len(items)} items from `{scenario_path}`")


def make_preview_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    display_rows: List[Dict[str, Any]] = []
    for it in records:
        req_body = it.get("request_body", {})
        messages = req_body.get("messages", [])
        last_message = messages[-1].get("content", "N/A") if messages else (it.get("Last Query") or "N/A")
        display_rows.append({
            "Timestamp": it.get("Timestamp") or it.get("timestamp", "N/A"),
            "Chat ID": it.get("Chat ID") or req_body.get("chat_id", "N/A"),
            "Status": it.get("Status") or it.get("status_code", "N/A"),
            "Last Query": last_message,
            "Response": it.get("Response", "")
        })
    return pd.DataFrame(display_rows)


if mode == "Single":
    preview_df = make_preview_table(items[:100])
    st.subheader("Preview (first 100)")
    st.markdown(preview_df.to_html(index=False, escape=False), unsafe_allow_html=True)

    sel_idx = st.number_input("Pick item index", min_value=0, max_value=max(0, len(items) - 1), value=0, step=1)
    it = items[sel_idx]

    with st.expander("Request Payload", expanded=True):
        payload = item_to_payload(it)
        st.json(payload)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        run_btn = st.button("Send to /chat", type="primary")
    with col_b:
        random_btn = st.button("Pick Random Item")
    with col_c:
        expected_str = it.get("Response", "")
        st.text_input("Expected (from file)", value=str(expected_str), key="expected_view", disabled=True)

    if random_btn:
        st.rerun()

    if run_btn:
        try:
            status, data, elapsed = post_chat(base_url, payload, timeout=int(timeout))
            st.success(f"Status: {status} | {elapsed:.2f}s")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Response Body")
                st.json(data)
            with col2:
                st.markdown("#### Quick Comparison")
                exp_bases = parse_expected_bases(expected_str)
                if exp_bases:
                    actual_bases = []
                    if isinstance(data, dict):
                        actual_bases = (data.get("base_random_keys") or [])
                    inter = set(exp_bases).intersection(set(actual_bases))
                    st.write({
                        "expected_bases": exp_bases,
                        "actual_bases": actual_bases,
                        "overlap_count": len(inter),
                    })
                else:
                    st.write("No structured expectation parsed. Showing raw expected text:")
                    st.code(str(expected_str))
        except Exception as e:
            st.error(str(e))

else:
    st.subheader("Batch Run (random sample)")
    batch_n = st.number_input("Number of items", min_value=1, max_value=min(200, len(items)), value=min(20, len(items)), step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=10**9, value=0, step=1)
    run_batch = st.button("Run Batch")

    if run_batch:
        random.seed(int(seed))
        sample = random.sample(items, k=min(int(batch_n), len(items)))
        rows = []
        successes = 0
        for idx, it in enumerate(sample, 1):
            payload = item_to_payload(it)
            expected = it.get("Response", "")
            try:
                status, data, elapsed = post_chat(base_url, payload, timeout=int(timeout))
                ok = (status == 200)
                if ok:
                    successes += 1
                exp_bases = parse_expected_bases(expected)
                actual_bases = (data.get("base_random_keys") if isinstance(data, dict) else None) or []
                overlap = len(set(exp_bases).intersection(set(actual_bases))) if exp_bases else None
                rows.append({
                    "idx": idx,
                    "chat_id": payload.get("chat_id"),
                    "status": status,
                    "elapsed_s": round(elapsed, 2),
                    "last_query": payload.get("messages", [{}])[-1].get("content", ""),
                    "expected": expected,
                    "actual_bases": actual_bases,
                    "overlap_with_expected_bases": overlap,
                })
            except Exception as e:
                rows.append({
                    "idx": idx,
                    "chat_id": payload.get("chat_id"),
                    "status": "error",
                    "elapsed_s": None,
                    "last_query": payload.get("messages", [{}])[-1].get("content", ""),
                    "expected": expected,
                    "actual_bases": None,
                    "overlap_with_expected_bases": None,
                    "error": str(e),
                })
        df = pd.DataFrame(rows)
        st.markdown(df.to_html(index=False, escape=False), unsafe_allow_html=True)
        st.info(f"HTTP 200 success rate: {successes}/{len(rows)} = {successes/len(rows):.0%}")


