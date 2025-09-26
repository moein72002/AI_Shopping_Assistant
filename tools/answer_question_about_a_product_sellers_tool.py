import os
import sqlite3
from typing import Optional, Dict, Any
import json
from utils.utils import _append_chat_log
from tools.database_tool import get_db_schema

DATASETS_PATH = os.path.abspath("/datasets/")
DB_PATH = os.path.join(DATASETS_PATH, "torob.db")


def _aggregate_seller_facts(cur: sqlite3.Cursor, product_id: str) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    # Count sellers
    cur.execute("SELECT COUNT(*) FROM members WHERE base_random_key = ?", (product_id,))
    row = cur.fetchone()
    facts["seller_count"] = int(row[0] or 0)

    # Min and max price
    cur.execute("SELECT MIN(price), MAX(price) FROM members WHERE base_random_key = ?", (product_id,))
    row = cur.fetchone() or (None, None)
    facts["min_price"] = float(row[0]) if row and (row[0] is not None) else None
    facts["max_price"] = float(row[1]) if row and (row[1] is not None) else None

    # Any Torob warranty sellers and how many
    cur.execute(
        """
        SELECT COUNT(*)
        FROM members m
        JOIN shops s ON m.shop_id = s.id
        WHERE m.base_random_key = ? AND s.has_warranty = 1
        """,
        (product_id,),
    )
    row = cur.fetchone()
    facts["warranty_seller_count"] = int(row[0] or 0)

    # Best seller by score (with tie-breaker price asc)
    cur.execute(
        """
        SELECT m.random_key, m.price, s.id, s.score, s.has_warranty
        FROM members m
        JOIN shops s ON m.shop_id = s.id
        WHERE m.base_random_key = ?
        ORDER BY s.score DESC, m.price ASC
        LIMIT 1
        """,
        (product_id,),
    )
    row = cur.fetchone()
    if row:
        facts["best_member_random_key"] = row[0]
        facts["best_member_price"] = float(row[1]) if row[1] is not None else None
        facts["best_shop_id"] = int(row[2]) if row[2] is not None else None
        facts["best_shop_score"] = float(row[3]) if row[3] is not None else None
        facts["best_shop_has_warranty"] = bool(row[4])
    else:
        facts["best_member_random_key"] = None
        facts["best_member_price"] = None
        facts["best_shop_id"] = None
        facts["best_shop_score"] = None
        facts["best_shop_has_warranty"] = None

    return facts


def _format_rows_as_text(description, rows, max_rows: int = 10) -> str:
    try:
        if not rows:
            return "No rows returned."
        col_names = [col[0] for col in description]
        lines = [", ".join(col_names)]
        for row in rows[:max_rows]:
            lines.append(", ".join(str(v) for v in row))
        if len(rows) > max_rows:
            lines.append(f"... ({len(rows) - max_rows} more rows)")
        return "\n".join(lines)
    except Exception:
        return "No rows returned."


def _generate_sql_for_sellers(product_id: str, question: str) -> Optional[str]:
    """
    Use an LLM to generate a single SQLite SELECT query that answers the seller-related
    aspect of the user's question for the given base product id.
    The SQL MUST constrain results to the specific product via base_random_key/product_id.
    Return the SQL string or None.
    """
    try:
        schema = get_db_schema(DB_PATH)
        sys_prompt = (
            """
You are an expert SQLite query generator. Your task is to generate a single, read-only sql query to answer a user's question about product sellers. The query's final result **must be a single numerical value** that can be parsed as an `int` or `float`.

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
        )
        from openai import OpenAI  # type: ignore
        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
        user_msg = (
            "### Database Schema:\n" + json.dumps(schema, ensure_ascii=False) +
            f"\n\n### PRODUCT_ID: {product_id}\n\n### QUESTION:\n{question or ''}\n\nReturn ONLY SQL."
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_SQL_MODEL", "gpt-5-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=10,
        )
        sql_query = (resp.choices[0].message.content or "").strip()
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
        return sql_query or None
    except Exception:
        return None


def answer_question_about_a_product_sellers(chat_id: str, product_id: str, question: str) -> str:
    """
    Answer questions about sellers (price, availability, Torob warranty, shop score) for a base product.
    Returns a concise Persian string. For purely numeric requests (e.g., min price), returns just the number.
    """
    try:
        _append_chat_log(chat_id, {"stage": "tool_start", "function_name": "answer_question_about_a_product_sellers", "product_id": product_id, "question": question})
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        facts = _aggregate_seller_facts(cur, product_id)
        conn.close()

        # 1) Generate SQL specifically constrained to this product
        sql_query = _generate_sql_for_sellers(product_id, question)
        sql_result_text: Optional[str] = None
        if sql_query:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(sql_query)
                if sql_query.strip().lower().startswith("select"):
                    rows = cursor.fetchall()
                    sql_result_text = _format_rows_as_text(cursor.description, rows)
                else:
                    conn.commit()
                    sql_result_text = f"Query executed successfully. Rows affected: {cursor.rowcount}"
                conn.close()
            except Exception as _:
                sql_result_text = None

        # 2) Merge SQL result alongside product seller facts
        enriched = dict(facts)
        enriched["generated_sql"] = sql_query
        enriched["generated_sql_result"] = sql_result_text
        _append_chat_log(chat_id, {"stage": "tool_result", "function_name": "answer_question_about_a_product_sellers_sql_generated_result", "product_id": product_id, "question": question, "facts": facts, "sql": sql_query, "sql_result": sql_result_text})
        content = sql_result_text.split("\n")[-1]
        return content
#         # 3) Ask LLM to produce the final concise Persian answer using both
#         try:
#             from openai import OpenAI  # type: ignore

#             client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
#             sys_prompt = (
#                 """
#                 ```markdown
# You are a data extraction bot. Your sole purpose is to output a single numerical value based on the provided inputs.

# ### Context
# - **User's Original Question (Persian):** `{question}`
# - **Result from SQL Query:** `{sql_result}`

# ### Your Task
# Your task is to return the raw numerical value from the **Result from SQL Query**. The user's question is provided only for context.

# ### Rules for Output
# 1.  Your response **MUST** be a single number (integer or float).
# 2.  Do **NOT** include any text, sentences, explanations, currency units (like تومان), or any characters other than digits and a potential decimal point.
# 3.  The output must be directly parsable by a program as an `int` or `float`.

# ---

# ### Example 1
# **Inputs:**
# - **User's Original Question (Persian):** `ارزان‌ترین قیمت چند است؟`
# - **Result from SQL Query:** `2550000`

# **Required Output:**
# ```

# 2550000

# ```

# ---

# ### Example 2
# **Inputs:**
# - **User's Original Question (Persian):** `چند فروشنده این کالا را دارند؟`
# - **Result from SQL Query:** `14`

# **Required Output:**
# ```

# 14

# ```
# ```
#                 """
#             )
#             resp = client.chat.completions.create(
#                 model=os.environ.get("LLM_SELLERS_QA_MODEL", "gpt-5-mini"),
#                 messages=[
#                     {"role": "system", "content": sys_prompt},
#                     {
#                         "role": "user",
#                         "content": "Facts:\n"
#                         + json.dumps(enriched, ensure_ascii=False)
#                         + "\n\nQuestion (سؤال):\n"
#                         + (question or ""),
#                     },
#                 ],
#                 timeout=10,
#             )
#             content = (resp.choices[0].message.content or "").strip()
#             if content:
#                 if chat_id:
#                     _append_chat_log(chat_id, {"stage": "tool_result", "function_name": "answer_question_about_a_product_sellers", "product_id": product_id, "question": question, "facts": facts, "sql": sql_query, "sql_result": sql_result_text, "answer": content})
#                 return content
        # except Exception:
        #     pass

        # Fallback: if we got any SQL result text, return a compact form, else min_price, else unknown
        if sql_result_text:
            return sql_result_text
        if facts.get("min_price") is not None:
            return str(facts["min_price"])
        return "نامشخص"
    except Exception:
        return "نامشخص"


