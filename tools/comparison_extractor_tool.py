import json
import os
from typing import Dict


def comparison_extract_products(query: str) -> Dict[str, str]:
    """
    Extract the two product names from a comparison-style user query.

    Returns a dict with keys: {"product_a": str, "product_b": str} when found, else {}.

    The prompt is bilingual (English + Persian) and enforces short, lexical product names
    with preserved codes/numbers (e.g., model tokens like 50RU766S or M 051).
    """
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"))

        sys_prompt = (
            "English Prompt: \n"
            "You extract exactly two product names from a user's comparison request. "
            "Output must be strict JSON with product_a and product_b. \n"
            "Rules: \n"
            "- If the user compares A vs B (explicitly or implicitly), return one concise lexical name for each product.\n"
            "- KEEP ALPHANUMERIC CODES AND NUMBERS VERBATIM (e.g., NANO75, A3891, 50RU766S, M 051).\n"
            "- Keep 1-5 words per product; remove filler words.\n"
            "- If you cannot find two products, return an empty JSON object {}.\n"
            "- Strict JSON only, no prose. Example: {\"product_a\": \"LG NANO75 50 inch\", \"product_b\": \"LG UT80006 50 inch\"}.\n"
            "\n"
            "Persian Prompt: \n"
            "از یک درخواست مقایسه، دقیقاً دو نام محصول استخراج کن. \n"
            "خروجی فقط JSON با فیلدهای product_a و product_b باشد.\n"
            "قوانین: \n"
            "- اگر کاربر بین دو محصول مقایسه می‌کند (صریح یا ضمنی)، برای هر کدام یک نام کوتاه و واژگانی برگردان.\n"
            "- کدها و اعداد را دقیق حفظ کن (مانند NANO75، A3891، 50RU766S، M 051).\n"
            "- برای هر محصول ۱ تا ۵ کلمه؛ کلمات اضافه حذف شوند.\n"
            "- اگر دو محصول پیدا نشد، یک JSON خالی {} برگردان.\n"
            "- فقط JSON دقیق؛ بدون متن اضافی. مثال: {\"product_a\": \"ال جی NANO75 50 اینچ\", \"product_b\": \"ال جی UT80006 50 اینچ\"}.\n"
        )

        user_msg = (
            "User query:"\
            f"\n{query}\n"\
            "Return only JSON."
        )

        resp = client.chat.completions.create(
            model=os.environ.get("LLM_BM25_MODEL", "gpt-5"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=12,
        )
        content = resp.choices[0].message.content
        data = json.loads(content or "{}")
        a = (data.get("product_a") or "").strip()
        b = (data.get("product_b") or "").strip()
        if a and b and a != b:
            return {"product_a": a, "product_b": b}
        return {}
    except Exception:
        return {}


