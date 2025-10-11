import os
import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


# --- Config ---
DEFAULT_BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")


# --- DB helpers (mirror utils.get_datasets_dir/get_db_path logic) ---
def _get_datasets_dir() -> str:
    datasets_dir = os.environ.get("DATASETS_DIR") or "/datasets/"
    try:
        os.makedirs(datasets_dir, exist_ok=True)
        return datasets_dir
    except Exception:
        app_root = os.path.dirname(os.path.abspath(__file__))
        fallback_dir = os.path.join(app_root, "datasets")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


def _get_db_path() -> str:
    return os.path.join(_get_datasets_dir(), "torob.db")


def _fetch_base_product(cur: sqlite3.Cursor, random_key: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        "SELECT random_key, persian_name, english_name, category_id, brand_id, extra_features, image_url FROM base_products WHERE random_key = ?",
        (random_key,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return {cols[i]: row[i] for i in range(len(cols))}


def _member_count_and_min_price(cur: sqlite3.Cursor, base_key: str) -> Tuple[int, Optional[float]]:
    cur.execute(
        "SELECT COUNT(*), MIN(price) FROM members WHERE base_random_key = ?",
        (base_key,),
    )
    row = cur.fetchone()
    if not row:
        return 0, None
    cnt = int(row[0] or 0)
    minp = row[1]
    try:
        minp = float(minp) if minp is not None else None
    except Exception:
        minp = None
    return cnt, minp


def _render_base_product_info(base_key: str) -> None:
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        st.warning(f"Database not found at {_get_db_path()} — ensure startup downloaded/created it.")
        return
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        data = _fetch_base_product(cur, base_key)
        if not data:
            st.warning("No product found for this base_random_key.")
            return
        members_cnt, min_price = _member_count_and_min_price(cur, base_key)

        name = (data.get("persian_name") or "").strip() or (data.get("english_name") or "").strip()
        st.subheader(name or base_key)

        cols = st.columns(2)
        with cols[0]:
            # Image
            img_url = (data.get("image_url") or "").strip()
            if img_url:
                # Support various Streamlit versions
                try:
                    st.image(img_url, use_column_width=True)
                except TypeError:
                    try:
                        st.image(img_url, use_container_width=True)
                    except TypeError:
                        st.image(img_url)
            else:
                st.info("No image available.")
        with cols[1]:
            st.markdown("**Base ID**: `" + base_key + "`")
            st.markdown("**Brand ID**: " + str(data.get("brand_id")))
            st.markdown("**Category ID**: " + str(data.get("category_id")))
            st.markdown("**Sellers (members)**: " + str(members_cnt))
            if min_price is not None:
                st.markdown("**Min Price**: " + str(int(min_price)) if float(min_price).is_integer() else f"**Min Price**: {min_price}")

            # Extra features pretty JSON if parseable
            ef = data.get("extra_features")
            if ef:
                try:
                    obj = json.loads(str(ef))
                    st.markdown("**Extra Features**")
                    st.json(obj)
                except Exception:
                    st.markdown("**Extra Features (raw)**")
                    st.code(str(ef))
    finally:
        conn.close()


# --- UI ---
st.set_page_config(layout="wide", page_title="AI Shopping Assistant - Streamlit UI")
st.title("AI Shopping Assistant - Test UI")

with st.sidebar:
    base_url = st.text_input("Service Base URL", value=DEFAULT_BASE_URL)
    st.caption("The Streamlit app will send POST requests to `{base_url}/chat`.")
    language = st.selectbox("Language", ["Persian", "English"], index=0)

# Apply RTL for Persian UI
if language == "Persian":
    st.markdown(
        """
<style>
html, body, [data-testid="stAppViewContainer"] * { direction: rtl; text-align: right; }
</style>
""",
        unsafe_allow_html=True,
    )


def _post_chat(base_url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat"
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        return {"status": r.status_code, "data": r.json()}
    except Exception:
        return {"status": r.status_code, "data": {"message": r.text}}


label_text = "متن پرسش را وارد کنید" if language == "Persian" else "Type your query (text)"
default_query = "سلام، یک گوشی آیفون ۱۶ پرومکس میخوام" if language == "Persian" else "Hello, I want an iPhone 16 Pro Max"
query_text = st.text_input(label_text, value=default_query)
send_btn = st.button("Send to /chat")

payload: Optional[Dict[str, Any]] = None

if send_btn:
    if not payload:
        payload = {"chat_id": f"ui-{os.getpid()}-{os.urandom(3).hex()}", "messages": [{"type": "text", "content": query_text}]}
    resp = _post_chat(base_url, payload)

    data = resp.get("data") or {}
    brks: Optional[List[str]] = data.get("base_random_keys") if isinstance(data, dict) else None
    if brks:
        st.markdown("---")
        st.markdown("### Detected Base Product(s)")
        for rk in brks[:10]:
            with st.expander(f"Base `{rk}`", expanded=True):
                _render_base_product_info(str(rk))



