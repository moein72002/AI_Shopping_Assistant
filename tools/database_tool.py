import sqlite3
# Removed top-level OpenAI import to defer heavy deps
import os
import re
from typing import Optional
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")
# get_product_by_code removed per new design

# Mapping for common Persian feature names to English keys in the JSON
FEATURE_NAME_MAP = {
    "عرض": "width",
    "قیمت": "price",
    # Add other mappings as needed
}

def get_product_feature(product_name: str, feature_name: str) -> Optional[str]:
    """
    Gets a specific feature for a product given its name.
    Searches for the feature key in the extra_features JSON.
    """
    english_feature_name = FEATURE_NAME_MAP.get(feature_name, feature_name)
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        search_pattern = f"%{product_name}%"
        cur.execute(
            "SELECT extra_features FROM base_products WHERE persian_name LIKE ?",
            (search_pattern,),
        )
        result = cur.fetchone()
        if not result:
            return "Product not found."
        
        features_obj = json.loads(result[0])
        feature_value = features_obj.get(english_feature_name)

        conn.close()

        if feature_value is not None:
            return str(feature_value)
        else:
            return f"Feature '{english_feature_name}' not found for this product."

    except Exception as e:
        return f"Error accessing database: {e}"


def get_db_schema(db_path):
    """Gets the database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        schema[table_name] = [f"{col[1]} ({col[2]})" for col in columns]
    conn.close()
    return schema


def _format_rows_as_text(description, rows, max_rows=10):
    if not rows:
        return "No rows returned."
    col_names = [col[0] for col in description]
    lines = [", ".join(col_names)]
    for row in rows[:max_rows]:
        lines.append(", ".join(str(v) for v in row))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def _fallback_answer(query: str) -> str | None:
    ql = query.lower()
    if "price" in ql and "random_key" in ql:
        m = re.search(r"random_key[^'\"]*['\"]([^'\"]+)['\"]", query)
        rk = m.group(1) if m else None
        if rk:
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("SELECT MIN(price) FROM members WHERE base_random_key = ?", (rk,))
                row = cur.fetchone()
                conn.close()
                if row and row[0] is not None:
                    return str(row[0])
                return "Not found"
            except Exception as e:
                return f"Error executing fallback SQL: {e}"
    return None


def get_min_price_by_product_name(product_name: str) -> str | None:
    """Returns the minimum member price for a base product matched by name.
    Returns a numeric string if found; otherwise None.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT random_key FROM base_products WHERE persian_name LIKE ? LIMIT 1",
            (f"%{product_name}%",),
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            return None

        base_rk = row[0]
        cur.execute(
            "SELECT MIN(price) FROM members WHERE base_random_key = ?",
            (base_rk,),
        )
        price_row = cur.fetchone()
        conn.close()
        if price_row and price_row[0] is not None:
            return str(price_row[0])
        return None
    except Exception:
        return None


def get_min_price_by_product_id(product_id: str) -> str | None:
    """Return the minimum price across member offers for a given base product id (random_key)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT MIN(price) FROM members WHERE base_random_key=?", (product_id,))
        row = cur.fetchone()
        conn.close()
        if row and row[0] is not None:
            return str(row[0])
        return None
    except Exception:
        return None


def text_to_sql(query: str):
    """
    Converts a natural language query to a SQL query using an LLM,
    executes it, and returns the result as text.
    """
    # Prefer fast deterministic fallback to avoid network latency
    fb_pre = _fallback_answer(query)
    if fb_pre is not None:
        return fb_pre

    schema = get_db_schema(DB_PATH)

    prompt = f"""
    Given the following SQLite database schema:
    {schema}

    Generate a SQL query to answer the following question:
    "{query}"

    Only return the SQL query. Do not include any explanations.
    """

    sql_query = None
    try:
        # Defer openai import to avoid loading for unrelated routes
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            timeout=10,
        )
        sql_query = response.choices[0].message.content.strip()
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
    except Exception as e:
        fb = _fallback_answer(query)
        if fb is not None:
            return fb
        return f"LLM unavailable: {e}"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        if sql_query.strip().lower().startswith("select"):
            rows = cursor.fetchall()
            result_text = _format_rows_as_text(cursor.description, rows)
        else:
            conn.commit()
            result_text = f"Query executed successfully. Rows affected: {cursor.rowcount}"
        conn.close()
        return result_text
    except Exception as e:
        fb = _fallback_answer(query)
        if fb is not None:
            return fb
        return f"Error executing SQL query: {e}\nSQL: {sql_query}"


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    test_query = "What is the price of the product with random_key 'bmubxu'?"
    print(text_to_sql(test_query))
