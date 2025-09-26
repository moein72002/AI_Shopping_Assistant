import json
import os
from typing import Optional
from utils.utils import _append_chat_log


def extract_product_name(chat_id: str, query: str) -> Optional[str]:
    """
    English Prompt:
    Extract a single product name mentioned in the user query (non-comparison).
    Returns a concise lexical product name (1-6 words), preserving alphanumeric
    codes and numbers.
    Warnings:
    You should not translate anything from query in the product name.
    Persian Prompt:
    استخراج نام یک محصول واحد ذکر شده در درخواست کاربر (از درخواست غیرمقایسه‌ای).
یک نام محصول واژگانی مختصر (۱ تا ۶ کلمه) را با حفظ کدهای الفبایی-عددی و اعداد برمی‌گرداند.
    هشدارها:
شما نباید هیچ بخشی از درخواست را در نام محصول ترجمه کنید.
    """
    try:
        # If LLM is disabled in tests, return None to let bm25_llm_search handle raw query
        if os.environ.get("DISABLE_LLM_FOR_TESTS") == "1":
            return None

        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"))
        sys_prompt = (
"""
You are an AI that refines search queries for a Persian e-commerce lexical search system. Your task is to extract a single, concise product name, including its most critical specifications, from the user's query (Persian or English).

### Rules

1.  **Extract Core Product:** Identify the primary product. Preserve essential tokens: brand, model, alphanumeric codes (e.g., `NANO75`, `GTV-50RU766S`), English phrases, color, size, and numbers.
2.  **Be Concise:** Remove all filler words (e.g., "best," "price," "خرید," "قیمت"). The ideal output is 1-10 words most of the time.
3.  **No Translation:** Retain all tokens in their original language.
4.  **Handle Special Keys:** If the query contains a 6-character lowercase key (e.g., `bmubxu`) or a UUID, your sole output must be that key.
5.  **Strict JSON Output:** The final output must be in the following format:
    ```json
    {"name": ["<product_name>"]}
    ```

### Examples

  * **Input**: "بهترین قیمت گوشی ایفون 13 پرو مکس 256 گیگ سبز"
      * **Output**: `{"name": ["ایفون 13 پرو مکس 256 گیگ سبز"]}`
  * **Input**: "buy cheap samsung TV model GTV-50RU766S"
      * **Output**: `{"name": ["samsung TV GTV-50RU766S"]}`
  * **Input**: "bmubxu فرش ۶ متری"
      * **Output**: `{"name": ["bmubxu"]}`
"""

#             """
# "English Prompt: \n"
# "You are an expert in refining shopping search queries for a BM25 lexical retrieval system that works with Persian product names. "
# "Your task is to extract exactly one single, concise product name from the user's query, which can be in either Persian or English. This product name should include its most critical specifications."
# "Rules: \n"
# "- Identify and extract the primary product mentioned in the query. Your output must be a single product name. \n"
# "- Focus on preserving crucial tokens that define the product: brand, model, specific alphanumeric codes (e.g., NANO75, A3891, GTV-50RU766S, S25), English expressions, color, size, and quantity numbers.\n"
# "- Do not translate any part of the query. Names, expressions, sizes, and all other tokens must be preserved in their original language.\n"
# "- If an alphanumeric code (e.g., 50RU766S) is present, it is critical and must be included in the output.\n"
# "- ALWAYS preserve alphanumeric codes and numbers exactly as they appear. Do not translate, spell out, or alter them.\n"
# "- If you find a Torob `base_random_key` (6 lower-case English letters, e.g., 'bmubxu') or a member key (UUID), your sole output should be the Torob `base_random_key`.\n"
# "- Remove all non-essential filler words (e.g., articles, prepositions, adjectives like 'best' or 'cheap') to keep the query concise (ideally 1-5 words).\n"
# "- The final output must be in a strict JSON format: {\"name\": [\"product_name_here\"]}"
# "Persian Prompt: \n"
# "شما یک متخصص در پالایش جستارهای جستجوی خرید برای یک سیستم بازیابی واژگانی BM25 هستید که بر روی نام‌های محصولات فارسی کار می‌کند."
# "وظیفه شما این است که دقیقاً یک نام محصول واحد و مختصر را از جستار کاربر استخراج کنید. این جستار می‌تواند به زبان فارسی یا انگلیسی باشد. نام محصول باید شامل حیاتی‌ترین مشخصات آن باشد."
# "قوانین: \n"
# "- محصول اصلی ذکر شده در جستار را شناسایی و استخراج کنید. خروجی شما باید یک نام محصول واحد باشد.\n"
# "- بر روی حفظ توکن‌های حیاتی که محصول را تعریف می‌کنند تمرکز کنید: برند، مدل، کدهای حروفی-عددی خاص (مانند NANO75، A3891، GTV-50RU766S، S25)، عبارات انگلیسی، رنگ، اندازه و اعداد مربوط به تعداد.\n"
# "- هیچ بخشی از جستار را ترجمه نکنید. نام‌ها، عبارات، اندازه‌ها و سایر توکن‌ها باید به زبان اصلی خود حفظ شوند.\n"
# "- اگر یک کد حروفی-عددی (مانند 50RU766S) وجود دارد، این بخش حیاتی است و باید در خروجی گنجانده شود.\n"
# "- همیشه کدهای حروفی-عددی و اعداد را دقیقاً همانطور که هستند حفظ کنید. آن‌ها را ترجمه، به حروف یا تغییر ندهید.\n"
# "- اگر یک `base_random_key` ترب (۶ حرف کوچک انگلیسی، مانند 'bmubxu') یا یک member key (UUID) پیدا کردید، تنها خروجی شما باید همان `base_random_key` ترب باشد.\n"
# "- تمام کلمات پرکننده و غیرضروری (مانند حروف اضافه، صفاتی مانند 'بهترین' یا 'ارزان') را حذف کنید تا جستار مختصر باقی بماند (ایده‌آل بین ۱ تا ۵ کلمه).\n"
# "- خروجی نهایی باید در قالب JSON دقیق باشد: {\"name\": [\"product_name_here\"]}"
# """
            # "English Prompt:\n"
            # "You extract exactly one product name from a user's request (not a comparison). "
            # "Return strict JSON with a field product_name only.\n"
            # "Rules:\n"
            # "- If the user mentions a specific product (brand+model), output a short lexical name (1-6 words).\n"
            # "- KEEP ALPHANUMERIC CODES AND NUMBERS VERBATIM (e.g., 50RU766S, M 051).\n"
            # "- Remove filler words.\n"
            # "- If no specific product found, return an empty JSON {}.\n"
            # "Warnings: You should not translate anything from query in the product name."
            # "\n"
            # "Persian Prompt:\n"
            # "از درخواست کاربر (بدون مقایسه) دقیقاً یک نام محصول استخراج کن.\n"
            # "خروجی فقط JSON با فیلد product_name باشد.\n"
            # "قوانین:\n"
            # "- اگر یک محصول مشخص (برند+مدل) ذکر شده، یک نام واژگانی کوتاه (۱ تا ۶ کلمه) برگردان.\n"
            # "- کدها و اعداد را دقیق حفظ کن (مانند 50RU766S، M 051).\n"
            # "- کلمات اضافه حذف شوند.\n"
            # "- اگر محصول مشخصی یافت نشد، یک JSON خالی {} برگردان.\n"
            # "هشدارها: شما نباید هیچ بخشی از درخواست را در نام محصول ترجمه کنید."
        )

        user_msg = f"User query:\n{query}\nReturn only JSON."

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
        name = content
        data = json.loads(content or "{}")
        name = (data.get("name") or [])[0] or ""
        _append_chat_log(chat_id, {"stage": "tool_result", "function_name": "extract_product_name", "query": query, "name": name})
        return name or None
    except Exception:
        return None


