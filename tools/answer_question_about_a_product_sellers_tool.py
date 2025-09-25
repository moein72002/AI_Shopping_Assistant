import os
import sqlite3
from typing import Optional, Dict, Any
import json
from utils.utils import _append_chat_log

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")


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


def _rule_based_answer(question_lower: str, facts: Dict[str, Any]) -> Optional[str]:
    # Prefer numeric one-liners when clearly requested
    if ("کمترین" in question_lower) or ("min" in question_lower and "price" in question_lower) or ("lowest" in question_lower):
        if facts.get("min_price") is not None:
            return str(facts["min_price"])  # numeric as string
    if ("بیشترین" in question_lower) or ("max" in question_lower and "price" in question_lower) or ("highest" in question_lower):
        if facts.get("max_price") is not None:
            return str(facts["max_price"])  # numeric as string
    if ("چند" in question_lower and "فروشنده" in question_lower) or ("تعداد فروشنده" in question_lower) or ("how many sellers" in question_lower):
        return str(facts.get("seller_count", 0))
    if ("گارانتی" in question_lower) or ("warranty" in question_lower):
        count = int(facts.get("warranty_seller_count", 0))
        if count > 0:
            return f"تعداد فروشندگان دارای ضمانت توروب: {count}"
        return "فروشنده با ضمانت توروب یافت نشد."
    return None


def answer_question_about_a_product_sellers(chat_id: str, product_id: str, question: str) -> str:
    """
    Answer questions about sellers (price, availability, Torob warranty, shop score) for a base product.
    Returns a concise Persian string. For purely numeric requests (e.g., min price), returns just the number.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        facts = _aggregate_seller_facts(cur, product_id)
        conn.close()

        ql = (question or "").strip().lower()
        rb = _rule_based_answer(ql, facts)
        if rb is not None:
            return rb

        # Lightweight LLM formatting for non-trivial queries
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
            sys_prompt = (
                "English: Using provided facts about sellers for a product, answer in ONE short Persian sentence.\n"
                "If the answer is a number (e.g., a price or a count), reply with just the number.\n\n"
                "فارسی: با تکیه بر داده‌های فروشندگان، یک پاسخ کوتاه فارسی بده.\n"
                "اگر پاسخ عددی است (مثلاً قیمت یا تعداد)، فقط عدد را برگردان.\n"
            )
            resp = client.chat.completions.create(
                model=os.environ.get("LLM_SELLERS_QA_MODEL", "gpt-5-mini"),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": "Facts:\n"
                        + json.dumps(facts, ensure_ascii=False)
                        + "\n\nQuestion (سؤال):\n"
                        + (question or ""),
                    },
                ],
                timeout=10,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                _append_chat_log(chat_id, {"stage": "tool_result", "function_name": "answer_question_about_a_product_sellers", "product_id": product_id, "question": question, "product_facts": facts, "answer": content})
                return content
        except Exception:
            pass

        # Fallback
        if facts.get("min_price") is not None:
            return str(facts["min_price"])
        return "نامشخص"
    except Exception:
        return "نامشخص"


