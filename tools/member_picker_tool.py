import os
import sqlite3
from typing import Optional, Dict, Any
import json


DB_PATH = os.path.join("/datasets/", "torob.db")


def pick_best_member_for_base(
    base_random_key: str,
    *,
    max_price: Optional[float] = None,
    require_warranty: Optional[bool] = None,
    min_shop_score: Optional[float] = None,
    preferred_shop_id: Optional[int] = None,
    user_context: Optional[str] = None,
) -> Optional[str]:
    """
    Return the best member_random_key (shop offer) for a base product.

    Selection respects user constraints when provided:
    - max_price: filter offers with price <= max_price
    - require_warranty: if True, only sellers with Torob warranty
    - min_shop_score: filter sellers with shop score >= min_shop_score
    - preferred_shop_id: if provided, try to choose this seller if it offers the product (still constrained by other filters)

    Ordering heuristic:
    - Prefer preferred_shop_id if available (after filters), else order by price ASC, then shop score DESC, then has_warranty DESC.
    """
    try:
        # 0) If we have user context, attempt LLM-guided SQL generation first
        if user_context:
            try:
                # Extract structured prefs to guide SQL generation
                from tools.shop_preference_extractor_tool import extract_shop_preferences  # type: ignore
                prefs: Dict[str, Any] = extract_shop_preferences("", user_context)
            except Exception:
                prefs = {}
            try:
                sql = _generate_sql_for_member_selection(base_random_key, user_context, prefs or None)
            except Exception:
                sql = None
            if sql:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute(sql)
                    row = cur.fetchone()
                    conn.close()
                    if row and len(row) >= 1 and row[0]:
                        return str(row[0])
                except Exception:
                    pass

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        where_clauses = ["m.base_random_key = ?"]
        params: list = [base_random_key]
        if isinstance(max_price, (int, float)):
            where_clauses.append("m.price <= ?")
            params.append(float(max_price))
        if require_warranty is True:
            where_clauses.append("s.has_warranty = 1")
        if isinstance(min_shop_score, (int, float)):
            where_clauses.append("s.score >= ?")
            params.append(float(min_shop_score))

        where_sql = " AND ".join(where_clauses)

        # If user prefers a specific shop, try that first respecting filters
        if isinstance(preferred_shop_id, int):
            cur.execute(
                f"""
                SELECT m.random_key
                FROM members m
                JOIN shops s ON m.shop_id = s.id
                WHERE {where_sql} AND s.id = ?
                ORDER BY m.price ASC
                LIMIT 1
                """,
                params + [preferred_shop_id],
            )
            row = cur.fetchone()
            if row:
                conn.close()
                return row[0]

        # Otherwise choose by heuristic
        cur.execute(
            f"""
            SELECT m.random_key
            FROM members m
            JOIN shops s ON m.shop_id = s.id
            WHERE {where_sql}
            ORDER BY m.price ASC, s.score DESC, s.has_warranty DESC
            LIMIT 1
            """,
            params,
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
        return None
    except Exception:
        return None


def _generate_sql_for_member_selection(product_id: str, user_context: str, extracted_prefs: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Use an LLM to generate a single SQLite SELECT query that chooses ONE best member offer (member_random_key)
    for the given base product, based on the user's context and preferences.

    Requirements for the SQL:
    - MUST filter by the specific product: WHERE m.base_random_key = '{product_id}'
    - Return exactly one column as the first column: m.random_key (member_random_key)
    - Respect preferences if present (max price ceiling, Torob warranty requirement, minimum shop score, preferred shop id)
    - Use ORDER BY to encode preference trade-offs and LIMIT 1
    - ONLY read data from tables: members (alias m) and shops (alias s)
    """
    try:
        from tools.database_tool import get_db_schema  # type: ignore
        schema = get_db_schema(DB_PATH)
        # Defer OpenAI import
        from openai import OpenAI  # type: ignore
        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
        sys_prompt = (
            "You are an expert SQLite query generator for e-commerce offers.\n"
            "Your goal is to pick ONE best seller offer (member) for a specific base product.\n\n"
            "Rules:\n"
            "1) Return ONLY a SQL query in a code block.\n"
            "2) The SQL MUST filter by the specific product id using: WHERE m.base_random_key = '{product_id}'.\n"
            "3) The SELECT list's first column MUST be m.random_key.\n"
            "4) Join members m with shops s ON m.shop_id = s.id when shop attributes are needed.\n"
            "5) Respect preferences when given (max_price ceiling, require_warranty boolean, min_shop_score, preferred_shop_id).\n"
            "6) Include ORDER BY to encode trade-offs (e.g., price ASC, score DESC, has_warranty DESC) and LIMIT 1.\n"
            "7) Use ONLY read-only SQL.\n"
        )
        user_payload = {
            "schema": schema,
            "PRODUCT_ID": product_id,
            "USER_CONTEXT": user_context or "",
            "EXTRACTED_PREFERENCES": extracted_prefs or {},
        }
        user_msg = (
            "### Database Schema (JSON):\n" + json.dumps(schema, ensure_ascii=False) +
            f"\n\n### PRODUCT_ID: {product_id}" +
            "\n\n### USER_CONTEXT (Persian):\n" + (user_context or "") +
            "\n\n### EXTRACTED_PREFERENCES (JSON):\n" + json.dumps(extracted_prefs or {}, ensure_ascii=False) +
            "\n\nReturn ONLY SQL."
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_SQL_MODEL", "gpt-4o-mini"),
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


