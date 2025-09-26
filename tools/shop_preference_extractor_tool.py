import os
import json
import re
from typing import Optional, Dict, Any


def _normalize_digits(text: str) -> str:
    trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    return (text or "").translate(trans)


def _heuristic_extract(text: str) -> Dict[str, Any]:
    """
    Lightweight heuristic to extract shop preferences.
    Returns keys: max_price (float|None), require_warranty (bool|None), min_shop_score (float|None), preferred_shop_id (int|None)
    """
    s = _normalize_digits(text or "")
    s_low = s.lower()
    max_price: Optional[float] = None
    min_shop_score: Optional[float] = None
    preferred_shop_id: Optional[int] = None
    require_warranty: Optional[bool] = None

    # max price: find the largest integer near currency words
    price_patterns = [r"(?:(?:قیمت|بودجه|تا|حداکثر)\s*)(\d{4,})(?:\s*(?:تومان|ریال)?)",
                      r"(\d{6,})(?:\s*(?:تومان|ریال))"]
    for pat in price_patterns:
        m = re.search(pat, s)
        if m:
            try:
                val = float(m.group(1))
                max_price = val if (max_price is None or val > max_price) else max_price
            except Exception:
                pass

    # warranty requirement
    if re.search(r"گارانتی|وارانتی|warranty", s_low):
        # if negation present
        if re.search(r"بدون\s*گارانتی|گارانتی\s*نمی\s*خواهم", s_low):
            require_warranty = False
        else:
            require_warranty = True

    # shop score
    m_score = re.search(r"امتیاز\s*(\d(?:\.\d)?)", s)
    if m_score:
        try:
            min_shop_score = float(m_score.group(1))
        except Exception:
            pass

    # preferred shop id
    m_sid = re.search(r"(?:shop[_\s-]?id|شناسه\s*فروشنده|شناسه\s*فروشگاه)\s*[:=\s]*([0-9]{1,6})", s_low)
    if m_sid:
        try:
            preferred_shop_id = int(m_sid.group(1))
        except Exception:
            pass

    return {
        "max_price": max_price,
        "require_warranty": require_warranty,
        "min_shop_score": min_shop_score,
        "preferred_shop_id": preferred_shop_id,
    }


def extract_shop_preferences(chat_id: str, text: str) -> Dict[str, Any]:
    """
    Extract shop/seller preferences from user text.
    Returns dict with keys: max_price, require_warranty, min_shop_score, preferred_shop_id.

    Uses LLM JSON extraction when available; falls back to heuristics.
    """
    try:
        if os.environ.get("DISABLE_LLM_FOR_TESTS") == "1":
            return _heuristic_extract(text)

        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
        sys_prompt = (
            "English: Extract shop preferences from Persian or English text. Return strict JSON with keys: "
            "max_price (number or null), require_warranty (true/false/null), min_shop_score (number or null), preferred_shop_id (integer or null).\n"
            "Do NOT translate tokens. If no info, use null.\n\n"
            "فارسی: از متن فارسی/انگلیسی ترجیحات فروشنده را استخراج کن و فقط JSON بازگردان: "
            "max_price (عدد یا null)، require_warranty (true/false/null)، min_shop_score (عدد یا null)، preferred_shop_id (عدد صحیح یا null)."
        )
        user_msg = f"Text to analyze:\n{text or ''}\nReturn ONLY JSON."
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_CLARIFY_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=10,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        # Normalize types
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None
        def _to_int(x):
            try:
                return int(x)
            except Exception:
                return None
        out = {
            "max_price": _to_float(data.get("max_price")),
            "require_warranty": data.get("require_warranty") if isinstance(data.get("require_warranty"), bool) else None,
            "min_shop_score": _to_float(data.get("min_shop_score")),
            "preferred_shop_id": _to_int(data.get("preferred_shop_id")),
        }
        # Fallback fill from heuristics if all null
        if all(v is None for v in out.values()):
            return _heuristic_extract(text)
        return out
    except Exception:
        return _heuristic_extract(text)


