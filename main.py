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
    # Import tools here to ensure dependencies are loaded on-demand
    from tools.database_tool import (
        text_to_sql, 
        get_product_by_code, 
        get_product_feature,
        get_min_price_by_product_name
    )
    from tools.vector_search_tool import vector_search
    from tools.bm25_tool import bm25_search
    from tools.bm25_llm_tool import bm25_llm_search
    from openai import OpenAI

    chat_id = request.chat_id
    
    # Scenario 0: Content-based health checks (chat_id can be random)
    try:
        last_msg = request.messages[-1]
        if last_msg.type == "text":
            txt = (last_msg.content or "").strip()
            lowered = txt.lower()
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

    # Maintain sliding window history (keep last 10 messages total)
    history = chat_histories.get(request.chat_id, [])
    history.append({"role": "user", "content": user_query})
    chat_histories[request.chat_id] = history[-10:]
    assistant_count = sum(1 for m in chat_histories[request.chat_id] if m.get("role") == "assistant")
    remaining_turns = max(0, 5 - assistant_count)

    # No scenario-specific heuristics beyond sanity checks; rely on LLM tool-calling
    
    # --- New Tool-Based Intent Router ---
    client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=20)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "vector_search",
                "description": "Searches for products based on a descriptive query. Use for general or semantic searches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user's search query, e.g., 'a red shirt for summer'"},
                    },
                    "required": ["query"],
                },
            },
        },
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
        {
            "type": "function",
            "function": {
                "name": "bm25_llm_search",
                "description": "LLM-augmented BM25S search. The LLM extracts brand/model/codes and creates 1-3 concise sub-queries, then BM25S retrieves matching products. Prefer this when queries contain numbers, codes, or more than one product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Original user query (Persian/English)."},
                        "k": {"type": "integer", "description": "Max results to return (<=10).", "default": 5}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product_by_code",
                "description": "Returns the base_random_key by matching a product code present in the product name, e.g., '(کد D14)'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_code": {"type": "string", "description": "The product code text, e.g., 'D14' or '(کد D14)'."}
                    },
                    "required": ["product_code"],
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
                "description": "Gets the absolute lowest price for a product from all sellers. Use this when the user asks for the 'minimum price', 'lowest price', or 'cheapest price'.",
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
                "You are a friendly AI Shopping Assistant. \n"
                "Rule Priority: Your main goal is to answer the user's specific question. If a query contains both a product identifier (like a code) and a question about a feature (like price or width), you MUST prioritize answering the feature question. \n"
                "--- \n"
                "1. **Product Code Rule**: If the user's query ONLY asks to find a product and provides a code like '(کد D14)', call `get_product_by_code`. \n"
                "Example: User says '... (کد D14) ...', you MUST call `get_product_by_code(product_code='(کد D14)')`. \n"
                "--- \n"
                "2. **Price/Feature Rule**: If the user asks for minimum price, lowest price, or a specific feature, you MUST use the appropriate tool (`get_min_price_by_product_name` or `get_product_feature`). This rule takes priority over the product code rule if both are present. \n"
                "FEW-SHOT EXAMPLE: \n"
                "User Query: 'کمترین قیمت ... محصول X ... چقدر است؟' \n"
                "Your action: Call `get_min_price_by_product_name(product_name='محصول X')`. \n"
                "--- \n"
                "Other Rules: \n"
                "- Otherwise use vector_search for discovery or ranking. \n"
                "- Prefer bm25_llm_search when the query contains explicit numbers/codes (e.g., model codes, sizes) or mentions multiple specific products to compare; this tool will extract concise lexical sub-queries and retrieve matching products via BM25S. \n"
                "- If the request is ambiguous, respond with one short clarifying question. \n"
                f"- You have at most 5 assistant messages per chat. Remaining assistant turns: {remaining_turns}. If you have no turns left, produce your best final answer using an appropriate tool."
            ),
        }
        model_messages = [system_message] + (history[-10:])
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=model_messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {
                "vector_search": vector_search,
                "bm25_search": bm25_search,
                "bm25_llm_search": bm25_llm_search,
                "get_product_by_code": get_product_by_code,
                "get_product_feature": get_product_feature,
                "get_min_price_by_product_name": get_min_price_by_product_name,
                "text_to_sql": text_to_sql,
            }
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "vector_search":
                results = function_to_call(query=function_args.get("query"))
                return ChatResponse(base_random_keys=results)
            elif function_name == "bm25_search":
                results = function_to_call(query=function_args.get("query"))
                return ChatResponse(base_random_keys=results)
            elif function_name == "bm25_llm_search":
                k = function_args.get("k", 5)
                if not isinstance(k, int) or k <= 0:
                    k = 5
                k = min(k, 10)
                results = function_to_call(query=function_args.get("query"), k=k)
                return ChatResponse(base_random_keys=results)
            elif function_name == "get_product_by_code":
                result = function_to_call(product_code=function_args.get("product_code"))
                return ChatResponse(base_random_keys=[result] if result else [])
            elif function_name == "get_product_feature":
                result = function_to_call(
                    product_name=function_args.get("product_name"),
                    feature_name=function_args.get("feature_name"),
                )
                # Append assistant message to history
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(result)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                return ChatResponse(message=result)
            elif function_name == "get_min_price_by_product_name":
                result = function_to_call(
                    product_name=function_args.get("product_name")
                )
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(result)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                return ChatResponse(message=result)
            elif function_name == "text_to_sql":
                result = function_to_call(query=function_args.get("query"))
                chat_histories[request.chat_id].append({"role": "assistant", "content": str(result)})
                chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
                return ChatResponse(message=result)

        # No tool call: return assistant's content if present (clarifying question or direct answer)
        if getattr(response_message, "content", None):
            content_text = response_message.content
            chat_histories[request.chat_id].append({"role": "assistant", "content": content_text})
            chat_histories[request.chat_id] = chat_histories[request.chat_id][-10:]
            return ChatResponse(message=content_text)

    except Exception as e:
        # Fallback to vector search on any failure
        print(f"Tool-based router failed: {e}. Defaulting to vector search.")
        results = vector_search(user_query.strip(), k=5)
        return ChatResponse(base_random_keys=results)
    
    # Default fallback if no tool is called
        results = vector_search(user_query.strip(), k=5)
        return ChatResponse(base_random_keys=results)
