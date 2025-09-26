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
from utils.utils import _append_chat_log, _reset_chat_log, _base_names_for_keys
import sqlite3
import subprocess as _subp
import logging
import subprocess
import sys
from tools.image_search import find_most_similar_product, warm_up_image_search
from tools.bm25_tool import bm25_search

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
    """Ensure required Kaggle artifacts exist in project root: torob.db, products.index, image_paths.json."""
    try:
        datasets_dir = os.path.dirname("/datasets/")
        db_path = os.path.join(datasets_dir, "torob.db")
        product_index_path = os.path.join(datasets_dir, "products.index")
        image_paths_path = os.path.join(datasets_dir, "image_paths.json")

        force = os.environ.get("FORCE_KAGGLE_DOWNLOAD") == "1"
        have_kaggle = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

        print(
            f"[startup] Kaggle check: force={force}, have_kaggle={have_kaggle}, "
            f"db_exists={os.path.exists(db_path)}, product_index_exists={os.path.exists(product_index_path)}, image_paths_exists={os.path.exists(image_paths_path)}"
        )

        # --- torob.db (database) ---
        if (not os.path.exists(db_path) or force) and have_kaggle:
            script_path = os.path.join(root_dir, "download_data_scripts", "download_data_from_kaggle.py")
            if os.path.exists(script_path):
                try:
                    print("[startup] Downloading torob.db from Kaggle...")
                    subprocess.run([sys.executable, script_path], check=True, cwd=root_dir)
                    produced = os.path.join("/datasets/", "torob.db")
                    if os.path.exists(produced):
                        print("[startup] torob.db downloaded successfully")
                    # if os.path.exists(produced):
                    #     try:
                    #         os.replace(produced, db_path)
                    #         print("[startup] Moved torob.db into app root")
                    #     except Exception:
                    #         pass
                except Exception as e:
                    print(f"[startup] Failed to download torob.db: {e}")
            else:
                print("[startup] Kaggle DB script not found; skipping torob.db download")

        # --- products.index and image_paths.json (image search index) ---
        needs_product_assets = (not os.path.exists(product_index_path) or not os.path.exists(image_paths_path) or force)
        if needs_product_assets and have_kaggle:
            script2_path = os.path.join(root_dir, "download_data_scripts", "download_product_index_from_kaggle.py")
            if os.path.exists(script2_path):
                try:
                    print("[startup] Downloading product index assets from Kaggle...")
                    subprocess.run([sys.executable, script2_path], check=True, cwd=root_dir)
                    print("[startup] product index assets downloaded successfully")
                    # # Search for the files under /datasets/ and move them to root
                    # shopping_dir = os.path.join("/datasets/")
                    # for dirpath, _, filenames in os.walk(shopping_dir):
                    #     for fname in filenames:
                    #         if fname in {"products.index", "image_paths.json"}:
                    #             src = os.path.join(dirpath, fname)
                    #             dst = os.path.join(root_dir, fname)
                    #             try:
                    #                 os.replace(src, dst)
                    #                 print(f"[startup] Moved {fname} into app root")
                    #             except Exception:
                    #                 pass
                except Exception as e:
                    print(f"[startup] Failed to download product index assets: {e}")
            else:
                print("[startup] Kaggle product index script not found; skipping products.index download")

        if not have_kaggle:
            print("[startup] Kaggle env vars not present; skipping all downloads")

        _append_chat_log("startup", {"stage": "warm_up_image_search"})
        warm_up_product_id = warm_up_image_search()
        print(f"[startup] Warm up image search: {warm_up_product_id}")

        _append_chat_log("startup", {"stage": "warm_up_bm25"})
        bm25_warmup_query = "رولت خوری دست ساز سرامیکی رودخانه"
        results = bm25_search("startup", bm25_warmup_query, k=5)
        print(f"[startup] Warm up bm25: bm25_warmup_query: {bm25_warmup_query}, results: {results}, names: {_base_names_for_keys(results)}")

    except Exception as _:
        # Non-fatal; app should still run
        print("[startup] Kaggle download failed (non-fatal)")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    chat_id = request.chat_id
    _reset_chat_log(request.chat_id)
    _append_chat_log(request.chat_id, {"stage": "chat_start", "request": request})
    
    # Scenario 0: Content-based health checks (chat_id can be random)
    try:
        last_msg = request.messages[-1]
        if last_msg.type == "text":
            txt = (last_msg.content or "").strip()
            lowered = txt.lower()
            # Initialize per-request log by truncating previous content for this chat_id
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

    # --- Early handling for image-based scenarios (6 & 7) ---
    try:
        last_msg = request.messages[-1]
        if last_msg.type == "image":
            _append_chat_log(request.chat_id, {"stage": "image_search_start_1", "type": last_msg.type})
            # Find most similar product by image
            img_b64 = last_msg.content
            _append_chat_log(request.chat_id, {"stage": "image_search_start_2"})
            product_id = find_most_similar_product(request.chat_id, img_b64)
            _append_chat_log(request.chat_id, {"stage": "image_search", "result": str(product_id)})

            # If image matching failed, return a graceful message
            if not product_id or isinstance(product_id, str) and product_id.lower().startswith("error:"):
                return ChatResponse(message="خطا در پردازش تصویر. لطفاً دوباره تلاش کنید.")

            # Use previous text (if any) to determine Scenario 6 vs 7
            prior_texts = [m.content for m in request.messages if getattr(m, "type", "text") == "text"]
            prior_text = (prior_texts[-1] if prior_texts else "").strip()

            # LLM: classify as scenario 6 or 7 from the user's accompanying text
            try:
                from openai import OpenAI
                client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=15)
                cls_msg = [
                    {
                        "role": "system",
                        "content": (
                            "You are a Persian assistant. Decide which image scenario applies based on the user's Persian text that came with an image.\n"
                            "Respond ONLY as: scenario_number = 6 or scenario_number = 7.\n\n"
                            "Scenario 6: The user asks what the main object in the image is (e.g., 'what is in this image?').\n"
                            "Scenario 7: The user asks for a related product suggestion based on the image (e.g., 'give me a product related to this image')."
                        ),
                    },
                    {"role": "user", "content": prior_text or ""},
                ]
                resp = client.chat.completions.create(
                    model=os.environ.get("LLM_ROUTER_MODEL", "gpt-5-mini"),
                    messages=cls_msg,
                    timeout=10,
                )
                cls_text = (resp.choices[0].message.content or "").strip()
            except Exception:
                cls_text = "scenario_number = 7"  # default to returning a product if classifier fails

            if "= 7" in cls_text:
                # Scenario 7: return the most similar base id
                return ChatResponse(base_random_keys=[product_id])

            # Scenario 6: return a concise Persian noun for the main object
            try:
                # Get product display name for guidance
                names = _base_names_for_keys([product_id])
                display_name = names.get(str(product_id), "")
                from openai import OpenAI
                client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=15)
                gen_msg = [
                    {
                        "role": "system",
                        "content": (
                            "You are a Persian assistant. Given a product name, output ONLY the core object noun in Persian (1-5 words).\n"
                            "Examples: 'پتو نرم مسافرتی' -> 'پتو' ; 'ماگ سرامیکی فانتزی' -> 'ماگ' ; 'گوشی موبایل آیفون 15' -> 'گوشی موبایل'"
                        ),
                    },
                    {"role": "user", "content": display_name or str(product_id)},
                ]
                gen_resp = client.chat.completions.create(
                    model=os.environ.get("LLM_ROUTER_MODEL", "gpt-5-mini"),
                    messages=gen_msg,
                    timeout=10,
                )
                noun = (gen_resp.choices[0].message.content or "").strip()
                noun = noun.split("\n")[0][:50]
                return ChatResponse(message=noun or display_name or "")
            except Exception:
                return ChatResponse(message="")
    except Exception:
        return ChatResponse(message="")

    user_query = request.messages[-1].content
    
    # Import tools here to ensure dependencies are loaded only after sanity checks
    from tools.database_tool import (
        text_to_sql,
        get_product_feature,
        get_min_price_by_product_name,
        get_min_price_by_product_id,
    )
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
            results = bm25_search(request.chat_id, user_query.strip(), k=k)
            _append_chat_log(request.chat_id, {"stage": "bypass_router", "tool": "bm25_search", "query": user_query, "results": results[:10] if results else []})
            if not results:
                # Ensure k results via bm25 padding path
                results = bm25_search(request.chat_id, user_query.strip(), k=5)
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
        name = extract_product_name(request.chat_id, q or "")
        if name:
            # If the extractor accidentally returned an id-like token, validate via extract_product_id
            pid2 = extract_product_id(name)
            if pid2:
                return pid2
            # Else search by name
            results = bm25_search(request.chat_id, name, k=5) or []
            return results[0] if results else None

        # TODO: Check below
        # Fallback: bm25 on full query
        results = bm25_search(request.chat_id, q or "", k=5) or []
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
-   **Description:** The user asks a question about the sellers of a specific, named product, such as its **price**. The output (answer of the user's question) must be parsable as an `int` or `float`
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

                ans = answer_question_about_a_product(request.chat_id, product_id=pid, question=user_query)
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(ans)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                return ChatResponse(message=ans)

            if scenario_num == 3:
                # Scenario 3: resolve product id -> answer about sellers
                _append_chat_log(request.chat_id, {"stage": "scenario3_start", "user_query": user_query})
                pid = _resolve_base_id(user_query)
                _append_chat_log(request.chat_id, {"stage": "scenario3_resolve", "pid": pid})
                ans = answer_question_about_a_product_sellers(request.chat_id, product_id=pid, question=user_query)
                _append_chat_log(request.chat_id, {"stage": "scenario3_answer", "answer": ans})
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(ans)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                return ChatResponse(message=ans)

            if scenario_num == 4:
                # Scenario 4: conversational narrowing, then return member_random_key
                if remaining_turns <= 1:
                    name = extract_product_name(request.chat_id, user_query or "")
                    # Last turn: pick a best candidate from the latest user text
                    cands = bm25_search(request.chat_id, name or "", k=5) or []
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
                if user_queries_count == 1:
                    q = ask_clarifying_questions(user_query)
                    chat_histories[request.chat_id].append({"role": "assistant", "content": q})
                    chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                    _append_chat_log(request.chat_id, {"stage": "scenario4_clarify", "q": q})
                    return ChatResponse(message=q)

                # Subsequent turns: try searching based on latest constraints and pick a member
                cands = bm25_search(request.chat_id, user_query.strip(), k=5) or []
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
                    ra = bm25_search(request.chat_id, pair["product_a"], k=5) or []
                    if ra:
                        ra_top = ra[0]
                if pair.get("product_b"):
                    rb = bm25_search(request.chat_id, pair["product_b"], k=5) or []
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
            results = bm25_search(request.chat_id, user_query.strip(), k=5)
            return ChatResponse(base_random_keys=results)
