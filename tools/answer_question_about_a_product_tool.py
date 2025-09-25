import os
import json
import sqlite3
from typing import Dict, Optional
from utils.utils import _append_chat_log


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")


def _fetch_base(cur: sqlite3.Cursor, base_id: str) -> Optional[Dict]:
    try:
        cur.execute(
            """
            SELECT random_key, persian_name, english_name, category_id, brand_id, extra_features
            FROM base_products
            WHERE random_key = ?
            LIMIT 1
            """,
            (base_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        features_raw = row[5]
        try:
            features = json.loads(features_raw) if features_raw else {}
        except Exception:
            features = {}
        return {
            "random_key": row[0],
            "persian_name": row[1],
            "english_name": row[2],
            "category_id": row[3],
            "brand_id": row[4],
            "extra_features": features,
        }
    except Exception:
        return None


def _fetch_min_price(cur: sqlite3.Cursor, base_id: str) -> Optional[float]:
    try:
        cur.execute("SELECT MIN(price) FROM members WHERE base_random_key = ?", (base_id,))
        row = cur.fetchone()
        if row and row[0] is not None:
            try:
                return float(row[0])
            except Exception:
                return None
        return None
    except Exception:
        return None


def _truncate_values(d: Dict, max_len: int = 160) -> Dict:
    out: Dict = {}
    for k, v in (d or {}).items():
        try:
            s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        except Exception:
            s = str(v)
        if len(s) > max_len:
            s = s[: max_len - 1] + "…"
        out[k] = s
    return out


def answer_question_about_a_product(chat_id: str, product_id: str, question: str) -> str:
    """
    Retrieve product data by base `product_id` and ask a small LLM to answer the user's
    question strictly based on that data. Returns a concise Persian answer.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        base = _fetch_base(cur, product_id)
        if not base:
            conn.close()
            return "نامشخص"
        min_price = _fetch_min_price(cur, product_id)
        conn.close()

        data = {
            "id": base.get("random_key"),
            "name": base.get("persian_name") or base.get("english_name"),
            "min_price": min_price,
            "features": _truncate_values(base.get("extra_features") or {}),
        }

        # Compose bilingual system prompt; request one short Persian answer
        sys_prompt = (
            "English: You answer questions about a product strictly using the provided data.\n"
            "Return ONE short Persian answer. If unknown, reply 'نامشخص'.\n\n"
            "فارسی: فقط با تکیه بر داده‌های محصول زیر پاسخ بده.\n"
            "پاسخ کوتاه و فارسی باشد و اگر قابل استخراج نیست، 'نامشخص' برگردان.\n"
        )

        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
            resp = client.chat.completions.create(
                model=os.environ.get("LLM_PRODUCT_QA_MODEL", "gpt-5-mini"),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Data:\n" + json.dumps(data, ensure_ascii=False) + "\n\nQuestion (سؤال):\n" + (question or "")
                        ),
                    },
                ],
                timeout=10,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                _append_chat_log(chat_id, {"stage": "tool_result", "function_name": "answer_question_about_a_product", "product_id": product_id, "question": question, "product_data": data, "answer": content})
                return content
        except Exception:
            pass

        # Simple fallback: if user asks price, return min_price; else unknown
        ql = (question or "").lower()
        if ("قیمت" in ql) or ("price" in ql):
            return str(min_price) if (min_price is not None) else "نامشخص"
        return "نامشخص"
    except Exception:
        return "نامشخص"


