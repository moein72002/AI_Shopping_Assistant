from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import re
import json
from typing import Dict, List as _List
from datetime import datetime
import sqlite3
import subprocess as _subp
import logging
import subprocess
import sys

# --- Logging Setup ---
# Create a logger
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

# Create a file handler
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests.log")
handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

# Add the handlers to the logger
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == handler.baseFilename for h in logger.handlers):
    logger.addHandler(handler)
# --- End Logging Setup ---


load_dotenv()

app = FastAPI()

# --- Per-chat logging helpers ---
def _append_chat_log(chat_id: str, record: dict):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, f"{chat_id}.log")
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            **record,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never fail request due to logging issues
        pass

def _reset_chat_log(chat_id: str):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, f"{chat_id}.log")
        # Truncate (overwrite) the log file for this chat_id at the start of each request
        with open(path, "w", encoding="utf-8"):
            pass
    except Exception:
        pass


def _base_names_for_keys(keys: _List[str]) -> Dict[str, str]:
    """Return a mapping from `random_key` to product name.

    Preference: use `persian_name` if non-empty, else `english_name`.
    """
    if not keys:
        return {}

    db_path = os.path.join(os.path.dirname(__file__), "torob.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        # First fetch all unique ids -> name into a lookup
        lookup: Dict[str, str] = {}

        unique_keys = list(dict.fromkeys(keys))
        chunk_size = 900
        for i in range(0, len(unique_keys), chunk_size):
            chunk = unique_keys[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = (
                "SELECT random_key, persian_name, english_name "
                "FROM base_products WHERE random_key IN (" + placeholders + ")"
            )
            cur.execute(sql, chunk)
            for row in cur.fetchall():
                rid = str(row["random_key"])  # ensure str keys
                pn = (row["persian_name"] or "").strip()
                en = (row["english_name"] or "").strip()
                name = pn if pn else en
                lookup[rid] = name

        # Now build an ordered mapping following the original input order
        ordered: Dict[str, str] = {}
        for k in keys:
            ordered[str(k)] = lookup.get(str(k), "")

        return ordered
    finally:
        conn.close()

# --- Logging Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "client_ip": request.client.host,
        "method": request.method,
        "path": request.url.path,
    }

    # Only log request body for /chat endpoint
    if request.url.path == "/chat":
        try:
            body = await request.json()
            log_data["request_body"] = body
        except json.JSONDecodeError:
            log_data["request_body"] = "Invalid JSON"

    response = await call_next(request)

    # A bit of a workaround to get the response body from a StreamingResponse
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk
    
    log_data["status_code"] = response.status_code
    try:
        log_data["response_body"] = json.loads(response_body)
    except json.JSONDecodeError:
        log_data["response_body"] = response_body.decode('utf-8', errors='ignore')

    logger.info(json.dumps(log_data))
    
    # We need to create a new response because the body_iterator has been consumed
    return Response(content=response_body, status_code=response.status_code, headers=dict(response.headers))
# --- End Logging Middleware ---


# --- Simple in-memory state (sliding window) ---
chat_histories: Dict[str, _List[dict]] = {}

class Message(BaseModel):
    type: str
    content: str

class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]

class ChatResponse(BaseModel):
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    print("\n--- [ADMIN] Admin page requested ---")
    # Render a simple HTML page with recent logs so it works on the same port
    try:
        print(f"[ADMIN] Attempting to read log file at: {LOG_PATH}")
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[ADMIN] Successfully read {len(lines)} lines from log file.")
    except Exception as e:
        print(f"[ADMIN] ERROR reading log file: {e}")
        lines = []

    # Show most recent first
    lines = list(reversed(lines))

    rows_html = []
    for line in lines:
        try:
            entry = json.loads(line)
        except Exception:
            continue
        ts = entry.get("timestamp", "-")
        path = entry.get("path", "-")
        status = entry.get("status_code", "-")
        chat_id = (entry.get("request_body", {}) or {}).get("chat_id", "-")
        # Short preview of last message
        msgs = (entry.get("request_body", {}) or {}).get("messages", [])
        preview = msgs[-1].get("content", "-") if msgs else "-"
        # Short preview of response
        resp = entry.get("response_body", {})
        if isinstance(resp, dict):
            resp_msg = resp.get("message")
            brk = resp.get("base_random_keys") or []
            mrk = resp.get("member_random_keys") or []
            resp_preview = resp_msg or (f"base={brk[:3]}" if brk else "") or (f"member={mrk[:3]}" if mrk else "-")
        else:
            resp_preview = str(resp)[:120]

        rows_html.append(
            f"<tr><td>{ts}</td><td>{status}</td><td>{path}</td><td>{chat_id}</td><td>{preview}</td><td>{resp_preview}</td></tr>"
        )

    table_html = (
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<thead><tr><th>Timestamp</th><th>Status</th><th>Path</th><th>Chat ID</th><th>Last Query</th><th>Response</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>API Request Logs</title>"
        "<meta http-equiv='refresh' content='5'>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px} table{width:100%; border-collapse:collapse} th{background:#f5f5f5}</style>"
        "</head><body>"
        "<h2>API Request Logs</h2>"
        + (table_html if rows_html else "<p>No logs yet.</p>")
        + "</body></html>"
    )

    print("[ADMIN] Successfully generated HTML. Sending response.")
    return HTMLResponse(content=html)

@app.on_event("startup")
async def maybe_download_kaggle_dataset():
    """If torob.db is missing and Kaggle creds are present at runtime, download the dataset."""
    try:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torob.db")
        force = os.environ.get("FORCE_KAGGLE_DOWNLOAD") == "1"
        print(f"[startup] Kaggle download check: force={force}, db_exists={os.path.exists(db_path)}")
        if os.path.exists(db_path) and not force:
            print("[startup] Skipping Kaggle download (DB already exists and force is off)")
            return
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_data_scripts", "download_data_from_kaggle.py")
            if os.path.exists(script_path):
                # Run the downloader
                print("[startup] Running Kaggle download script...")
                subprocess.run([sys.executable, script_path], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
                # Move produced DB into root if created in shopping_dataset/
                produced = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shopping_dataset", "torob.db")
                if os.path.exists(produced):
                    try:
                        os.replace(produced, db_path)
                        print("[startup] Moved torob.db into app root")
                    except Exception:
                        pass
            else:
                print("[startup] Kaggle script not found; skipping")
        else:
            print("[startup] Kaggle env vars not present; skipping download")
    except Exception as _:
        # Non-fatal; app should still run
        print("[startup] Kaggle download failed (non-fatal)")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    chat_id = request.chat_id
    
    # Scenario 0: Content-based health checks (chat_id can be random)
    try:
        last_msg = request.messages[-1]
        if last_msg.type == "text":
            txt = (last_msg.content or "").strip()
            lowered = txt.lower()
            # Initialize per-request log by truncating previous content for this chat_id
            _reset_chat_log(request.chat_id)
            _append_chat_log(request.chat_id, {"stage": "sanity_check", "text": txt})
            if lowered == "ping":
                return ChatResponse(message="pong")
            if lowered.startswith("return base random key:"):
                base_key = txt.split(":", 1)[1].strip() if ":" in txt else txt[len("return base random key:"):].strip()
                return ChatResponse(base_random_keys=[base_key])
            if lowered.startswith("return member random key:"):
                member_key = txt.split(":", 1)[1].strip() if ":" in txt else txt[len("return member random key:"):].strip()
                return ChatResponse(member_random_keys=[member_key])
    except Exception:
        pass

    user_query = request.messages[-1].content
    
    # Import tools here to ensure dependencies are loaded only after sanity checks
    from tools.database_tool import (
        text_to_sql,
        get_product_feature,
        get_min_price_by_product_name,
        get_min_price_by_product_id,
    )
    from tools.bm25_tool import bm25_search
    from tools.comparison_extractor_tool import comparison_extract_products
    from tools.product_name_extractor_tool import extract_product_name
    from tools.product_id_lookup_tool import extract_product_id
    from tools.answer_question_about_a_product_tool import answer_question_about_a_product
    from tools.compare_two_products_tool import compare_two_products
    from tools.answer_question_about_a_product_sellers_tool import answer_question_about_a_product_sellers
    from tools.ask_clarifying_questions_tool import ask_clarifying_questions
    from tools.member_picker_tool import pick_best_member_for_base

    # No precheck heuristics; rely on router tool-calling entirely

    # Maintain sliding window history (keep last 10 messages total)
    history = chat_histories.get(request.chat_id, [])
    history.append({"role": "user", "content": user_query})
    chat_histories[request.chat_id] = history[-10:]
    user_queries_count = sum(1 for m in chat_histories[request.chat_id] if m.get("role") == "user")
    remaining_turns = max(0, 5 - user_queries_count)

    # No scenario-specific heuristics beyond sanity checks; rely on LLM tool-calling
    
    # --- Test-mode bypass: disable router LLM and run bm25_llm_search directly ---
    if os.environ.get("DISABLE_ROUTER_LLM") == "1":
        try:
            k = 10
            results = bm25_search(user_query.strip(), k=k)
            _append_chat_log(request.chat_id, {"stage": "bypass_router", "tool": "bm25_search", "query": user_query, "results": results[:10] if results else []})
            if not results:
                # Ensure k results via bm25 padding path
                results = bm25_search(user_query.strip(), k=5)
                return ChatResponse(base_random_keys=results or None)
            return ChatResponse(base_random_keys=results)
        except Exception:
            pass

    # --- LLM: Classify scenario first ---
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=20)

    # Helper to find a single base product id using id/name -> bm25
    def _resolve_base_id(q: str) -> Optional[str]:
        pid = extract_product_id(q or "")
        if pid:
            return pid
        name = extract_product_name(q or "")
        if name:
            # If the extractor accidentally returned an id-like token, validate via extract_product_id
            pid2 = extract_product_id(name)
            if pid2:
                return pid2
            # Else search by name
            results = bm25_search(name, k=5) or []
            return results[0] if results else None

        # TODO: Check below
        # Fallback: bm25 on full query
        results = bm25_search(q or "", k=5) or []
        return results[0] if results else None

    # Ask LLM to classify into scenario 1..5
    scenario_num: Optional[int] = None
    try:
        cls_system_message = {
            "role": "system",
            "content": (
                """
You are an expert AI assistant for Torob. Your **sole purpose** is to analyze the user's request (user requests are in Persian language), classify it into one of the five scenarios described below, and return **only the scenario number**.

## **Primary Rule**
Your response must be in the exact format: `scenario_number = X`, where `X` is the number of the matching scenario. **Do not provide any other text, explanation, or formatting.**

---

## **Scenarios**

### **Scenario 1: Direct Product Lookup**
-   **Description:** The user is looking for a specific product and provides a clear, unambiguous name or product code.
-   **Example Query:** `"لطفاً دراور چهار کشو (کد D14) را برای من تهیه کنید."`
-   **Your Output for this Example:** `scenario_number = 1`

---

### **Scenario 2: Product Feature Question**
-   **Description:** The user asks a factual question about the attributes or specifications of a specific, named product (e.g., dimensions, material, technical details).
-   **Example Query:** `"عرض پارچه تریکو جودون 1/30 لاکرا گردباف نوریس به رنگ زرد طلایی چقدر است؟"`
-   **Your Output for this Example:** `scenario_number = 2`

---

### **Scenario 3: Seller & Price Question**
-   **Description:** The user asks a question about the commercial aspects of a specific, named product, such as its **price**, availability, sellers, or warranty.
-   **Example Query:** `"کمترین قیمت در این پایه برای گیاه طبیعی بلک گلد بنسای نارگل کد ۰۱۰۸ چقدر است؟"`
-   **Your Output for this Example:** `scenario_number = 3`

---

### **Scenario 4: Guided Product Discovery**
-   **Description:** The user has a vague or general request for a product category and needs help narrowing down their options. They are not asking about a specific, named item.
-   **Example Query:** `"نبال یه میز تحریر هستم که برای کارهای روزمره و نوشتن مناسب باشه."`
-   **Your Output for this Example:** `scenario_number = 4`

---

### **Scenario 5: Product Comparison**
-   **Description:** The user explicitly asks to compare two or more specific, named products.
-   **Example Query:** `"کدام یک از این ماگ‌های خرید ماگ-لیوان هندوانه فانتزی و کارتونی کد 1375 یا ماگ لته خوری سرامیکی با زیره کد 741 مناسب‌تر است؟"`
-   **Your Output for this Example:** `scenario_number = 5`
                """
            ),
        }
        model_messages = [cls_system_message] + (history[-10:])
        cls_resp = client.chat.completions.create(
            model=os.environ.get("LLM_ROUTER_MODEL", "gpt-5-mini"),
            messages=model_messages,
            timeout=10,
        )
        cls_text = (cls_resp.choices[0].message.content or "").strip()
        _append_chat_log(request.chat_id, {"stage": "scenario_classification", "raw": cls_text})
        m = re.search(r"scenario_number\s*=\s*(\d)", cls_text)
        if m:
            scenario_num = int(m.group(1))
    except Exception as _:
        scenario_num = None

    # Orchestrate flows per scenario if classified
    if scenario_num in {1, 2, 3, 4, 5}:
        try:
            if scenario_num == 1:
                # Scenario 1: extract product id and return it
                pid = _resolve_base_id(user_query)
                _append_chat_log(request.chat_id, {"stage": "scenario1_resolve", "pid": pid})
                return ChatResponse(base_random_keys=[pid] if pid else None)

            if scenario_num == 2:
                # Scenario 2: resolve product id -> answer about product
                pid = _resolve_base_id(user_query)
                _append_chat_log(request.chat_id, {"stage": "scenario2_resolve", "pid": pid})

                ans = answer_question_about_a_product(product_id=pid, question=user_query)
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(ans)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                return ChatResponse(message=ans)

            if scenario_num == 3:
                # Scenario 3: resolve product id -> answer about sellers
                pid = _resolve_base_id(user_query)
                _append_chat_log(request.chat_id, {"stage": "scenario3_resolve", "pid": pid})
                ans = answer_question_about_a_product_sellers(product_id=pid, question=user_query)
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(ans)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                return ChatResponse(message=ans)

            if scenario_num == 4:
                # Scenario 4: conversational narrowing, then return member_random_key
                if remaining_turns <= 1:
                    name = extract_product_name(user_query or "")
                    # Last turn: pick a best candidate from the latest user text
                    cands = bm25_search(name or "", k=5) or []
                    _append_chat_log(request.chat_id, {"stage": "scenario4_candidates", "cands": cands[:5]})
                    if cands:
                        best_member = pick_best_member_for_base(cands[0])
                        if best_member:
                            return ChatResponse(member_random_keys=[best_member])
                    # Fallback no candidates: ask one final question
                    q = ask_clarifying_questions(user_query)
                    chat_histories[request.chat_id].append({"role": "assistant", "content": q})
                    chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                    return ChatResponse(message=q)

                # Not final turn: ask a clarifying question first if this is the first assistant turn
                if assistant_count == 0:
                    q = ask_clarifying_questions(user_query)
                    chat_histories[request.chat_id].append({"role": "assistant", "content": q})
                    chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                    _append_chat_log(request.chat_id, {"stage": "scenario4_clarify", "q": q})
                    return ChatResponse(message=q)

                # Subsequent turns: try searching based on latest constraints and pick a member
                cands = bm25_search(user_query.strip(), k=5) or []
                _append_chat_log(request.chat_id, {"stage": "scenario4_candidates", "cands": cands[:5]})
                if cands:
                    best_member = pick_best_member_for_base(cands[0])
                    if best_member:
                        return ChatResponse(member_random_keys=[best_member])
                # If still nothing, ask another clarifying question
                q = ask_clarifying_questions(user_query)
                chat_histories[request.chat_id].append({"role": "assistant", "content": q})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                return ChatResponse(message=q)

            if scenario_num == 5:
                # Scenario 5: extract products -> bm25 -> compare -> return winner id and reason
                pair = comparison_extract_products(user_query)
                _append_chat_log(request.chat_id, {"stage": "scenario5_pair", "pair": pair})
                ra_top: Optional[str] = None
                rb_top: Optional[str] = None
                if pair.get("product_a"):
                    ra = bm25_search(pair["product_a"], k=5) or []
                    if ra:
                        ra_top = ra[0]
                if pair.get("product_b"):
                    rb = bm25_search(pair["product_b"], k=5) or []
                    if rb:
                        rb_top = rb[0]
                if ra_top and rb_top:
                    winner_id, reason_fa = compare_two_products(ra_top, rb_top, user_query)
                    _append_chat_log(request.chat_id, {"stage": "scenario5_compare", "winner": winner_id, "reason": reason_fa})
                    if winner_id:
                        return ChatResponse(message=reason_fa, base_random_keys=[winner_id])
                # Fallback: return any found candidates
                merged = [rk for rk in [ra_top, rb_top] if rk]
                return ChatResponse(base_random_keys=merged or None)
        except Exception as e:
            _append_chat_log(request.chat_id, {"stage": "scenario_flow_error", "error": str(e)})
            # Graceful fallback to bm25
            results = bm25_search(user_query.strip(), k=5)
            return ChatResponse(base_random_keys=results)

    # --- Fallback: original tool-based router (kept for resilience) ---
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bm25_search",
                "description": "Lexical search over product names/features. Use when query includes exact names, codes, ids.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Exactish name/code/id text to match lexically."}
                    },
                    "required": ["query"],
                },
            },
        },
        # bm25_llm_search removed from router: use extract_product_name + bm25_search instead
        {
            "type": "function",
            "function": {
                "name": "comparison_extract_products",
                "description": "Extracts the two product names from comparison-style queries (e.g., 'A vs B'). Returns product_a and product_b for subsequent search calls.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Original user query (Persian/English)."}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_product_name",
                "description": "Extract a single product name (brand+model) from a non-comparison query; returns a concise lexical name to pass to bm25_llm_search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_product_id",
                "description": "Extract a 6-letter product id (base_random_key) from the query and validate against DB.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "answer_question_about_a_product",
                "description": "Answer a question about a specific product using its DB features and min price. Use when product id is known.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "base_random_key of the product"},
                        "question": {"type": "string", "description": "User question text"}
                    },
                    "required": ["product_id", "question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_two_products",
                "description": "Given two base product ids (random_key) and the user's query, compare using DB features/prices and return the better product id and a short Persian reason.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id_a": {"type": "string", "description": "First base product id (random_key)"},
                        "product_id_b": {"type": "string", "description": "Second base product id (random_key)"},
                        "user_query": {"type": "string", "description": "Original user query to guide comparison"}
                    },
                    "required": ["product_id_a", "product_id_b", "user_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product_feature",
                "description": "Gets a specific feature of a named product. Use when the user asks for a specific attribute of a product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string", "description": "The name of the product."},
                        "feature_name": {"type": "string", "description": "The name of the feature to retrieve, e.g., 'عرض' or 'قیمت'."},
                    },
                    "required": ["product_name", "feature_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_min_price_by_product_name",
                "description": "(Legacy) Gets the lowest price by name; prefer get_min_price_by_product_id after the product is found.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string", "description": "The product name to match in the base catalog."}
                    },
                    "required": ["product_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_min_price_by_product_id",
                "description": "Gets the absolute lowest price for a product by base_random_key. Call only AFTER the product id is found via bm25_llm_search or extract_product_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The 6-letter base_random_key."}
                    },
                    "required": ["product_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "text_to_sql",
                "description": "Answers complex questions by generating and running a SQL query against the database. Use as a last resort for questions that other tools cannot answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user's full natural language query."},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    try:
        system_message = {
            "role": "system",
            "content": (
                """
                You are an expert AI assistant for Torob. Your **sole purpose** is to analyze the user's request, classify it into one of the five scenarios described below, and return **only the scenario number**.

                ## **Primary Rule**
                Your response must be in the exact format: `scenario_number = X`, where `X` is the number of the matching scenario. **Do not provide any other text, explanation, or formatting.**

                ---

                ## **Scenarios**

                ### **Scenario 1: Direct Product Lookup**
                -   **Description:** The user is looking for a specific product and provides a clear, unambiguous name or product code.
                -   **Example Query:** `"لطفاً دراور چهار کشو (کد D14) را برای من تهیه کنید."`
                -   **Your Output for this Example:** `scenario_number = 1`

                ---

                ### **Scenario 2: Product Feature Question**
                -   **Description:** The user asks a factual question about the attributes or specifications of a specific, named product (e.g., dimensions, material, technical details).
                -   **Example Query:** `"عرض پارچه تریکو جودون 1/30 لاکرا گردباف نوریس به رنگ زرد طلایی چقدر است؟"`
                -   **Your Output for this Example:** `scenario_number = 2`

                ---

                ### **Scenario 3: Seller & Price Question**
                -   **Description:** The user asks a question about the commercial aspects of a specific, named product, such as its **price**, availability, sellers, or warranty.
                -   **Example Query:** `"کمترین قیمت در این پایه برای گیاه طبیعی بلک گلد بنسای نارگل کد ۰۱۰۸ چقدر است؟"`
                -   **Your Output for this Example:** `scenario_number = 3`

                ---

                ### **Scenario 4: Guided Product Discovery**
                -   **Description:** The user has a vague or general request for a product category and needs help narrowing down their options. They are not asking about a specific, named item.
                -   **Example Query:** `"نبال یه میز تحریر هستم که برای کارهای روزمره و نوشتن مناسب باشه."`
                -   **Your Output for this Example:** `scenario_number = 4`

                ---

                ### **Scenario 5: Product Comparison**
                -   **Description:** The user explicitly asks to compare two or more specific, named products.
                -   **Example Query:** `"کدام یک از این ماگ‌های خرید ماگ-لیوان هندوانه فانتزی و کارتونی کد 1375 یا ماگ لته خوری سرامیکی با زیره کد 741 مناسب‌تر است؟"`
                -   **Your Output for this Example:** `scenario_number = 5`
                """
            ),
        }
        model_messages = [system_message] + (history[-10:])
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=model_messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        _append_chat_log(request.chat_id, {"stage": "router_response", "message": getattr(response_message, "content", None), "tool_calls": [tc.function.name for tc in (tool_calls or [])]})

        if tool_calls:
            available_functions = {
                "bm25_search": bm25_search,
                "comparison_extract_products": comparison_extract_products,
                "extract_product_name": extract_product_name,
                "extract_product_id": extract_product_id,
                "get_product_feature": get_product_feature,
                "get_min_price_by_product_name": get_min_price_by_product_name,
                "get_min_price_by_product_id": get_min_price_by_product_id,
                "text_to_sql": text_to_sql,
                "answer_question_about_a_product": answer_question_about_a_product,
                "compare_two_products": compare_two_products,
            }
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            _append_chat_log(request.chat_id, {"stage": "tool_call", "tool": function_name, "args": function_args})

            if function_name == "bm25_search":
                results = function_to_call(query=function_args.get("query"))
                name_map = _base_names_for_keys(results or [])
                print(f"[main] bm25_search name_map={name_map} results={results[:10]}")
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "results": results[:10] if results else [], "names": name_map})
                return ChatResponse(base_random_keys=results)
            # bm25_llm_search branch removed
            elif function_name == "comparison_extract_products":
                pair = function_to_call(query=function_args.get("query"))
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "pair": pair})
                # If extracted, run bm25 for each name, select top candidate for each, then compare
                ra_top: Optional[str] = None
                rb_top: Optional[str] = None
                merged: _List[str] = []
                if pair.get("product_a"):
                    ra = bm25_search(pair["product_a"], k=5) or []
                    if ra:
                        ra_top = ra[0]
                        merged.extend([rk for rk in ra if rk not in merged])
                if pair.get("product_b"):
                    rb = bm25_search(pair["product_b"], k=5) or []
                    if rb:
                        rb_top = rb[0]
                        merged.extend([rk for rk in rb if rk not in merged])
                name_map = _base_names_for_keys(merged)
                print(f"[main] comparison_extract_products name_map={name_map} merged={merged[:10]}")
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": "comparison_candidates", "a_top": ra_top, "b_top": rb_top, "all": merged[:10], "names": name_map})
                # If we have both top candidates, run compare tool
                if ra_top and rb_top:
                    winner_id, reason_fa = compare_two_products(ra_top, rb_top, user_query)
                    _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": "compare_two_products", "winner": winner_id, "reason": reason_fa})
                    if winner_id:
                        return ChatResponse(message=reason_fa, base_random_keys=[winner_id])
                # Fallback: return merged candidates
                return ChatResponse(base_random_keys=merged[:10] or None)
            elif function_name == "compare_two_products":
                winner_id, reason = function_to_call(
                    product_id_a=function_args.get("product_id_a"),
                    product_id_b=function_args.get("product_id_b"),
                    user_query=function_args.get("user_query") or user_query,
                )
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "winner": winner_id, "reason": reason})
                if winner_id:
                    return ChatResponse(message=reason, base_random_keys=[winner_id])
                return ChatResponse(base_random_keys=None)
            elif function_name == "extract_product_name":
                name = function_to_call(query=function_args.get("query"))
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "name": name})
                # If we got a name, run bm25 name search and then let the agent decide next tool
                print(f"[main] bm25_search name={name}")
                if name:
                    results = bm25_search(name, k=5) or []
                    print(f"[main] bm25_search name={name} results={results}")
                    name_map = _base_names_for_keys(results)
                    print(f"[main] bm25_search name_map={name_map} results={results}")
                    _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": "bm25_search", "query": name, "results": results[:10], "names": name_map})
                    return ChatResponse(base_random_keys=results)
                # fallback to bm25 on full query
                results = bm25_search(function_args.get("query"), k=5) or []
                return ChatResponse(base_random_keys=results)
            elif function_name == "extract_product_id":
                pid = function_to_call(query=function_args.get("query"))
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "product_id": pid})
                return ChatResponse(base_random_keys=[pid] if pid else None)
            elif function_name == "get_product_feature":
                result = function_to_call(
                    product_name=function_args.get("product_name"),
                    feature_name=function_args.get("feature_name"),
                )
                # Append assistant message to history
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(result)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "message": result})
                return ChatResponse(message=result)
            elif function_name == "get_min_price_by_product_name":
                result = function_to_call(
                    product_name=function_args.get("product_name")
                )
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(result)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "message": result})
                return ChatResponse(message=result)
            elif function_name == "get_min_price_by_product_id":
                result = function_to_call(
                    product_id=function_args.get("product_id")
                )
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(result)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "message": result})
                return ChatResponse(message=result)
            elif function_name == "answer_question_about_a_product":
                result = function_to_call(
                    product_id=function_args.get("product_id"),
                    question=function_args.get("question") or user_query,
                )
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(result)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "message": result})
                return ChatResponse(message=result)
            elif function_name == "text_to_sql":
                result = function_to_call(query=function_args.get("query"))
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(result)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                _append_chat_log(request.chat_id, {"stage": "tool_result", "tool": function_name, "message": result})
                return ChatResponse(message=result)

        # No tool call: return assistant's content if present (clarifying question or direct answer)
        if getattr(response_message, "content", None):
            content_text = response_message.content
            chat_histories[request.chat_id].append({"role": "assistant", "content": content_text})
            chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
            _append_chat_log(request.chat_id, {"stage": "final_message", "message": content_text})
            return ChatResponse(message=content_text)

    except Exception as e:
        # Fallback to bm25_search on any failure
        print(f"Tool-based router failed: {e}. Defaulting to bm25_search.")
        _append_chat_log(request.chat_id, {"stage": "router_error", "error": str(e)})
        results = bm25_search(user_query.strip(), k=5)
        _append_chat_log(request.chat_id, {"stage": "fallback_bm25", "results": results[:10] if results else []})
        return ChatResponse(base_random_keys=results)
    
    # Default fallback if no tool is called
        results = bm25_search(user_query.strip(), k=5)
        return ChatResponse(base_random_keys=results)
