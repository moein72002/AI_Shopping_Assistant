import os
import json
import sqlite3
from typing import Dict, Optional, Tuple
from tools.database_tool import get_db_schema
from utils.utils import _append_chat_log


DB_PATH = os.path.join("/datasets/", "torob.db")


def _fetch_base_product(cur: sqlite3.Cursor, base_random_key: str) -> Optional[Dict]:
    try:
        cur.execute(
            """
            SELECT random_key, persian_name, english_name, category_id, brand_id, extra_features, image_url
            FROM base_products
            WHERE random_key = ?
            LIMIT 1
            """,
            (base_random_key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        extra = row[5]
        try:
            features = json.loads(extra) if extra else {}
        except Exception:
            features = {}
        return {
            "random_key": row[0],
            "persian_name": row[1],
            "english_name": row[2],
            "category_id": row[3],
            "brand_id": row[4],
            "extra_features": features,
            "image_url": row[6],
        }
    except Exception:
        return None


def _fetch_min_price(cur: sqlite3.Cursor, base_random_key: str) -> Optional[float]:
    try:
        cur.execute(
            "SELECT MIN(price) FROM members WHERE base_random_key = ?",
            (base_random_key,),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            try:
                return float(row[0])
            except Exception:
                return None
        return None
    except Exception:
        return None


def _compose_llm_prompt(product_a: Dict, product_b: Dict, user_query: str, sql_result_a: Optional[str], sql_result_b: Optional[str]) -> Tuple[str, str]:
    # Build a compact bilingual system prompt and a user message that aggregates
    # product features and the SQL results for each product.
    a_name = product_a.get("persian_name") or product_a.get("english_name") or product_a.get("random_key")
    b_name = product_b.get("persian_name") or product_b.get("english_name") or product_b.get("random_key")
    a_feat = json.dumps(product_a.get("extra_features") or {}, ensure_ascii=False)
    b_feat = json.dumps(product_b.get("extra_features") or {}, ensure_ascii=False)
    a_price = product_a.get("min_price")
    b_price = product_b.get("min_price")
    system_prompt = (
        "English: Choose the better product strictly among the two base products based on the user's goal.\n"
        "Return strict JSON: {\\\"winner_id\\\": string, \\\"reason_fa\\\": string}.\n"
        "Prefer features relevant to the request; break ties with lower price. Keep the reason short in Persian.\n\n"
        "Persian: از بین دو محصول زیر، بر اساس هدف کاربر فقط یک مورد را انتخاب کن.\n"
        "خروجی فقط JSON باشد: {\\\"winner_id\\\": string, \\\"reason_fa\\\": string}.\n"
        "ویژگی‌های مرتبط با درخواست کاربر را اولویت بده؛ در حالت برابر، قیمت کمتر برنده است. دلیل کوتاه و فارسی باشد.\n"
    )

    user_msg = (
        f"User Query:\n{user_query}\n\n"
        f"Product A:\n"
        f"- id: {product_a.get('random_key')}\n"
        f"- name: {a_name}\n"
        f"- min_price: {a_price}\n"
        f"- features: {a_feat}\n"
        f"- sql_result: {(sql_result_a or '').strip()}\n\n"
        f"Product B:\n"
        f"- id: {product_b.get('random_key')}\n"
        f"- name: {b_name}\n"
        f"- min_price: {b_price}\n"
        f"- features: {b_feat}\n"
        f"- sql_result: {(sql_result_b or '').strip()}\n"
    )

    return (system_prompt, user_msg)


def _generate_sql_for_compare(chat_id: str, product_id: str, user_query: str) -> Optional[str]:
    try:
        schema = get_db_schema(DB_PATH)
        from openai import OpenAI  # type: ignore
        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
        sys_prompt = (
"""
You are an expert SQLite query generator. Your task is to generate a single, read-only sql query to answer a user's question about comparison of two products. The query's final result **must be a single numerical value** that can be parsed as an `int` or `float`.

You will be given the database schema, a specific product ID, and a user's question in Persian.

### Rules
1.  **Analyze Intent:** Carefully analyze the user's Persian question to determine the correct calculation needed (e.g., counting sellers, finding the minimum price, calculating an average).
2.  **Single Numeric Result:** The query **MUST** return a single numerical value. You may use aggregate functions like `COUNT()`, `AVG()`, `SUM()`, `MIN()`, or `MAX()` to achieve this.
3.  **Mandatory Filter:** The query **MUST** filter results for the specified product using a condition like `WHERE base_random_key = '{product_id}'`.
4.  **Table Usage:** Prioritize querying from the `members` (offers) table. `JOIN` with the `shops` table for seller information as needed for filtering or aggregation.
5.  **Output Format:** Your response **MUST** contain only the SQL query. Enclose the entire query in a `sql` markdown block. Do not add any text, comments, or explanations before or after the code block.

**Example Response (for a question like "What is the lowest price?"):**
```sql
SELECT
  MIN(m.price)
FROM members AS m
WHERE
  m.base_random_key = '{product_id}';
"""


# """
# ### 🎯 **Task**

# Your goal is to generate a single SQLite `SELECT` query. This query should extract a specific **numeric value** from the database that is relevant to the user's comparison goal for a given product.

# ### 📝 **Inputs**

# You will be provided with:

# 1.  **User Query:** The user's request in Persian (e.g., "باتری این گوشی چطوره؟").
# 2.  **Product ID:** A unique identifier for the product, which you should use in place of `<PRODUCT_ID>`.

# ### 📜 **Rules**

# 1.  **Single Query:** You must generate only **one** `SELECT` statement.
# 2.  **Single Numeric Value:** The query's output must be containing only one numeric value (e.g., an integer or a float).
# 3.  **Filter by Product ID:** The query **must** include a `WHERE` clause: `base_random_key = '<PRODUCT_ID>'`.
# 4.  **SQLite Syntax:** The query must use valid SQLite syntax.
# 5.  **Output Format:** Return **only** the raw SQL query inside a markdown code block. Do not add any explanation, greeting, or other text.

# """

            # "English: Generate ONE SELECT SQLite query that produces a single numeric value relevant to the user's comparison goal for the given product.\n"
            # "Rules: The query MUST filter by base_random_key = '<PRODUCT_ID>'. Return ONLY SQL (code block allowed).\n\n"
            # "Persian: یک کوئری SELECT واحد تولید کن که یک مقدار عددی مرتبط با هدفِ مقایسه‌ای بر اساس شناسه محصول بدهد.\n"
            # "نتایج را به product_id محدود کن (base_random_key = '<PRODUCT_ID>'). فقط SQL برگردان.\n"
        )
        user_msg = (
            "Schema:\n" + json.dumps(schema, ensure_ascii=False) +
            f"\n\nPRODUCT_ID: {product_id}\n\nUSER_QUERY:\n{user_query}\n\nReturn SQL only."
        )
        _append_chat_log(chat_id, {"stage": "compare_two_products_end_generate_sql_for_compare", "product_id": product_id, "user_query": user_query, "user_msg": user_msg})
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_SQL_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=10,
        )
        _append_chat_log(chat_id, {"stage": "compare_two_products_end_generate_sql_for_compare", "product_id": product_id, "user_query": user_query, "resp": resp})
        sql = (resp.choices[0].message.content or "").strip()
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()
        return sql or None
    except Exception:
        return None


def _execute_sql(sql_query: Optional[str]) -> Optional[str]:
    if not sql_query:
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(sql_query)
        if sql_query.strip().lower().startswith("select"):
            rows = cur.fetchall()
            # Minimal formatting: last value or header+first row
            if not rows:
                result_text = "No rows returned."
            else:
                header = [c[0] for c in cur.description]
                result_text = ", ".join(header) + "\n" + ", ".join(str(v) for v in rows[0])
        else:
            conn.commit()
            result_text = f"Query executed successfully. Rows affected: {cur.rowcount}"
        conn.close()
        return result_text
    except Exception:
        return None


def compare_two_products(chat_id: str, product_id_a: str, product_id_b: str, user_query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Compare two base products (by random_key) using DB facts and a small LLM call.

    Returns: (winner_base_random_key, reason_fa)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        a = _fetch_base_product(cur, product_id_a)
        b = _fetch_base_product(cur, product_id_b)
        if not a or not b:
            conn.close()
            return (None, None)
        a_price = _fetch_min_price(cur, product_id_a)
        b_price = _fetch_min_price(cur, product_id_b)
        a["min_price"] = a_price
        b["min_price"] = b_price
        conn.close()

        # If both have identical features and prices missing, fallback to cheaper known price or None
        if (a_price is not None) and (b_price is not None) and abs(a_price - b_price) < 0.005:
            # Tie on price: prefer richer features length if any
            len_a = len(a.get("extra_features") or {})
            len_b = len(b.get("extra_features") or {})
            winner = product_id_a if len_a >= len_b else product_id_b
            reason = "به‌دلیل ویژگی‌های بیشتر/مشابه با قیمت برابر انتخاب شد."
            return (winner, reason)

        # Generate per-product SQL signals guided by the user request
        _append_chat_log(chat_id, {"stage": "compare_two_products_start_generate_sql_for_compare", "product_id_a": product_id_a, "product_id_b": product_id_b})
        sql_a = _generate_sql_for_compare(chat_id, product_id_a, user_query)
        _append_chat_log(chat_id, {"stage": "compare_two_products_end_generate_sql_for_compare", "product_id_a": product_id_a, "sql_a": sql_a})
        sql_b = _generate_sql_for_compare(chat_id, product_id_b, user_query)
        _append_chat_log(chat_id, {"stage": "compare_two_products_end_generate_sql_for_compare", "product_id_b": product_id_b, "sql_b": sql_b})
        sql_res_a = _execute_sql(sql_a)
        sql_res_b = _execute_sql(sql_b)
        _append_chat_log(chat_id, {"stage": "compare_two_products_end_generate_sql_for_compare", "product_id_a": product_id_a, "product_id_b": product_id_b, "sql_a": sql_a, "sql_res_a": sql_res_a, "sql_b": sql_b, "sql_res_b": sql_res_b})

        # Call LLM for decision with SQL results included
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=12)
            sys_prompt, user_msg = _compose_llm_prompt(a, b, user_query, sql_res_a, sql_res_b)
            resp = client.chat.completions.create(
                model=os.environ.get("LLM_COMPARE_MODEL", "gpt-4o-mini"),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                timeout=12,
            )
            content = resp.choices[0].message.content
            data = json.loads(content or "{}")
            winner_id = (data.get("winner_id") or "").strip()
            reason_fa = (data.get("reason_fa") or "").strip()
            if winner_id in {product_id_a, product_id_b}:
                return (winner_id, reason_fa or None)
        except Exception:
            pass

        # Cheap deterministic fallback if LLM fails: prefer lower min price, else A
        if a_price is not None and b_price is not None:
            return (product_id_a if a_price <= b_price else product_id_b, None)
        if a_price is not None:
            return (product_id_a, None)
        if b_price is not None:
            return (product_id_b, None)
        return (product_id_a, None)
    except Exception:
        return (None, None)


