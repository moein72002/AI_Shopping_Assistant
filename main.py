from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import re
import json
from typing import Dict, List as _List
# Defer tool imports to avoid startup errors
# from tools.database_tool import text_to_sql
# from tools.vector_search_tool import vector_search

load_dotenv()

app = FastAPI()

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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Import tools here to ensure dependencies are loaded on-demand
    from tools.database_tool import text_to_sql, get_product_by_code, get_product_feature
    from tools.vector_search_tool import vector_search
    from openai import OpenAI

    chat_id = request.chat_id
    
    if chat_id == "sanity-check-ping":
        return ChatResponse(message="pong")
    
    if chat_id == "sanity-check-base-key":
        content = request.messages[0].content
        base_key = content.replace("return base random key: ", "")
        return ChatResponse(base_random_keys=[base_key])
        
    if chat_id == "sanity-check-member-key":
        content = request.messages[0].content
        member_key = content.replace("return member random key: ", "")
        return ChatResponse(member_random_keys=[member_key])

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
                "description": "Returns the minimum price among member products for a base matched by product name.",
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
                "Rules: \n"
                "- Always call exactly one tool to act; do not answer without a tool unless asking a clarifying question. \n"
                "- If the query mentions a product code like '(کد D14)' or similar, call get_product_by_code with the code. \n"
                "- If the user asks for a specific feature value (e.g., 'عرض ... چقدر است؟'), call get_product_feature with product_name and feature_name. \n"
                "- If the user asks for minimum price (e.g., 'کمترین قیمت ... چقدر است؟'), call get_min_price_by_product_name. \n"
                "- Otherwise use vector_search for discovery or ranking. \n"
                "- If the request is ambiguous, respond with one short clarifying question. \n"
                f"- You have at most 5 assistant messages per chat. Remaining assistant turns: {remaining_turns}. If you have no turns left, produce your best final answer using an appropriate tool."
            ),
        }
        model_messages = [system_message] + (history[-10:])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            top_p=0,
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {
                "vector_search": vector_search,
                "get_product_by_code": get_product_by_code,
                "get_product_feature": get_product_feature,
                "get_min_price_by_product_name": lambda **kwargs: __import__("tools.database_tool", fromlist=["get_min_price_by_product_name"]).get_min_price_by_product_name(**kwargs),
                "text_to_sql": text_to_sql,
            }
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "vector_search":
                results = function_to_call(query=function_args.get("query"))
                return ChatResponse(base_random_keys=results)
            elif function_name == "get_product_feature":
                result = function_to_call(
                    product_name=function_args.get("product_name"),
                    feature_name=function_args.get("feature_name"),
                )
                # Append assistant message to history
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
