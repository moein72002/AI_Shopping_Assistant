import os
import json
import sqlite3
from typing import Dict, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")


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


def _compose_llm_prompt(product_a: Dict, product_b: Dict, user_query: str) -> str:
    # Bilingual system guidance; short and cost-aware
    a_name = product_a.get("persian_name") or product_a.get("english_name") or product_a.get("random_key")
    b_name = product_b.get("persian_name") or product_b.get("english_name") or product_b.get("random_key")
    a_feat = json.dumps(product_a.get("extra_features") or {}, ensure_ascii=False)
    b_feat = json.dumps(product_b.get("extra_features") or {}, ensure_ascii=False)
    a_price = product_a.get("min_price")
    b_price = product_b.get("min_price")

    return (
        "English System Prompt:\n"
        "You must choose the better product strictly among the two provided base products, based on the user's goal.\n"
        "Return strict JSON: {\"winner_id\": string, \"reason_fa\": string}.\n"
        "Rules: Prefer features relevant to the user's request; break ties with lower price. Keep the reason short in Persian.\n\n"
        "Persian System Prompt:\n"
        "می‌بایست بین دو محصول ارائه‌شده صرفاً یکی را بر اساس هدف کاربر انتخاب کنی.\n"
        "خروجی فقط JSON باشد: {\"winner_id\": string, \"reason_fa\": string}.\n"
        "قواعد: ویژگی‌های مرتبط با درخواست کاربر را اولویت بده؛ در حالت برابر، قیمت کمتر برنده است. دلیل کوتاه و فارسی باشد.\n\n"
        f"User Query:\n{user_query}\n\n"
        f"Product A: name={a_name}, id={product_a.get('random_key')}, min_price={a_price}, features={a_feat}\n"
        f"Product B: name={b_name}, id={product_b.get('random_key')}, min_price={b_price}, features={b_feat}\n"
        "Return only JSON."
    )


def compare_two_products(product_id_a: str, product_id_b: str, user_query: str) -> Tuple[Optional[str], Optional[str]]:
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

        # Call LLM for decision
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=12)
            prompt = _compose_llm_prompt(a, b, user_query)
            resp = client.chat.completions.create(
                model=os.environ.get("LLM_COMPARE_MODEL", "gpt-5-mini"),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
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


