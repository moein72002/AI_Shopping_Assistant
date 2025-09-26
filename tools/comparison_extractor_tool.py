import json
import os
import re
from typing import Dict
from utils.utils import _append_chat_log


def comparison_extract_products(chat_id: str, query: str) -> Dict[str, str]:
    """
    Extract the two product names from a comparison-style user query.

    Returns a dict with keys: {"product_a": str, "product_b": str} when found, else {}.

    The prompt is bilingual (English + Persian) and enforces short, lexical product names
    with preserved codes/numbers (e.g., model tokens like 50RU766S or M 051).
    """

    _append_chat_log(chat_id, {"stage": "comparison_extract_products_start", "query": query})
    try:
        # 1) Prefer explicit Torob base ids (6 lowercase letters) if present
        ids = re.findall(r"\b[a-z]{6}\b", query or "")
        ids = [tok.strip() for tok in ids if tok.strip()]
        # Deduplicate while preserving order
        seen = set()
        ids = [x for x in ids if not (x in seen or seen.add(x))]

        # If we already have two ids, skip LLM entirely
        if len(ids) >= 2:
            _append_chat_log(chat_id, {"stage": "comparison_extract_products_two_ids_found", "product_a": ids[0], "product_b": ids[1]})
            return {"product_a": ids[0], "product_b": ids[1]}

        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"))

        sys_prompt = (
            """
```markdown
# English Prompt: Note that the user query is in Persian language.
You extract exactly two product names from a user's comparison request. 
Output must be strict JSON with product_a and product_b. 

## Rules: 
- If the user compares A vs B (explicitly or implicitly), return one concise lexical name for each product.
- KEEP ALPHANUMERIC CODES AND NUMBERS VERBATIM (e.g., NANO75, A3891, 50RU766S, M 051).
- Keep 1-10 words per product; remove filler words.
- If you cannot find two products, return an empty JSON object {}.
- Strict JSON only, no prose. Example: {"product_a": "product_a_name", "product_b": "product_b_name"}.
            """
            # "English Prompt: \n"
            # "You extract exactly two product names from a user's comparison request. "
            # "Output must be strict JSON with product_a and product_b. \n"
            # "Rules: \n"
            # "- If the user compares A vs B (explicitly or implicitly), return one concise lexical name for each product.\n"
            # "- KEEP ALPHANUMERIC CODES AND NUMBERS VERBATIM (e.g., NANO75, A3891, 50RU766S, M 051).\n"
            # "- Keep 1-5 words per product; remove filler words.\n"
            # "- If you cannot find two products, return an empty JSON object {}.\n"
            # "- Strict JSON only, no prose. Example: {\"product_a\": \"product_a_name\", \"product_b\": \"product_b_name\"}.\n"
            # "\n"
            # "Persian Prompt: \n"
            # "از یک درخواست مقایسه، دقیقاً دو نام محصول استخراج کن. \n"
            # "خروجی فقط JSON با فیلدهای product_a و product_b باشد.\n"
            # "قوانین: \n"
            # "- اگر کاربر بین دو محصول مقایسه می‌کند (صریح یا ضمنی)، برای هر کدام یک نام کوتاه و واژگانی برگردان.\n"
            # "- کدها و اعداد را دقیق حفظ کن (مانند NANO75، A3891، 50RU766S، M 051).\n"
            # "- برای هر محصول ۱ تا ۵ کلمه؛ کلمات اضافه حذف شوند.\n"
            # "- اگر دو محصول پیدا نشد، یک JSON خالی {} برگردان.\n"
            # "- فقط JSON دقیق؛ بدون متن اضافی. مثال: {\"product_a\": \"product_a_name\", \"product_b\": \"product_b_name\"}.\n"
        )

        user_msg = (
            "User query:"\
            f"\n{query}\n"\
            "Return only JSON."
        )
        _append_chat_log(chat_id, {"stage": "comparison_extract_products_user_msg", "user_msg": user_msg})

        resp = client.chat.completions.create(
            model=os.environ.get("LLM_BM25_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=12,
        )
        content = resp.choices[0].message.content
        _append_chat_log(chat_id, {"stage": "comparison_extract_products_llm_response", "content": content})
        data = json.loads(content or "{}")
        a = (data.get("product_a") or "").strip()
        b = (data.get("product_b") or "").strip()

        if len(ids) == 1:
            # Use the known id for one side; keep the other side as extracted name
            # Prefer filling A with the known id, and ensure the other side is not the same id
            other = b if b and b != ids[0] else a
            _append_chat_log(chat_id, {"stage": "comparison_extract_products_known_id_used", "product_a": ids[0], "product_b": (other or b or a)})
            return {"product_a": ids[0], "product_b": (other or b or a)}

        if a and b and a != b:
            _append_chat_log(chat_id, {"stage": "comparison_extract_products_two_names_found", "product_a": a, "product_b": b})
            return {"product_a": a, "product_b": b}
        return {}
    except Exception:
        return {}


