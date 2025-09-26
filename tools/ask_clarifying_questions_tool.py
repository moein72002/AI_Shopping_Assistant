import os
from typing import Optional, List


def ask_clarifying_questions(context: str, candidate_names: Optional[List[str]] = None, force_shop_question: bool = False) -> str:
    """
    Generate ONE short Persian clarifying question to narrow down the user's needs.
    - If `candidate_names` (list of product names from BM25 top-k) is provided, the question should help disambiguate among them.
    - If `force_shop_question` is True, ask specifically about seller/shop preferences (price vs warranty vs score, or explicit shop_id) to finalize member selection.
    The `context` can include the user's last message or a brief history summary.
    """
    try:
        if os.environ.get("DISABLE_LLM_FOR_TESTS") == "1":
            return (
                "برای انتخاب فروشنده، قیمت پایین‌تر مهم‌تر است یا گارانتی و امتیاز فروشگاه؟ اگر فروشنده خاصی (shop_id) مدنظر دارید بفرمایید."
                if force_shop_question
                else "دقیق‌تر بفرمایید چه برندی/مدلی و حدود بودجه‌ای مدنظر شماست؟"
            )

        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=10)
        names_blob = "\n".join(f"- {n}" for n in (candidate_names or [])[:50])
        sys_prompt = (
            "English: Ask ONE short clarifying question in Persian. Keep it focused (<= 20 words).\n"
            "If candidate names are provided, ask about a key attribute (brand, size, color, model code, variant) that best disambiguates among them.\n"
            "If shop-stage is active, ask ONLY about seller preferences (budget/price ceiling, Torob warranty, high shop score, or a specific shop_id).\n\n"
            "فارسی: یک سوال کوتاه (<=۲۰ کلمه) برای شفاف‌سازی بپرس.\n"
            "اگر فهرست کاندیداها داده شد، درباره ویژگی تمایزبخش (مثلاً برند/مدل/کد/رنگ/سایز) سوال کن.\n"
            "اگر مرحله فروشنده است، فقط درباره ترجیحات فروشنده (بودجه/گارانتی/امتیاز/شناسه فروشنده) سوال کن."
        )
        user_sections = [f"Context:\n{context or ''}"]
        if candidate_names:
            user_sections.append("Top candidates (names):\n" + names_blob)
        if force_shop_question:
            user_sections.append("Stage: shop-selection")
        user_msg = "\n\n".join(user_sections) + "\n\nAsk one Persian question only."
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_CLARIFY_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=10,
        )
        content = (resp.choices[0].message.content or "").strip()
        default_q = (
            "برای فروشنده، قیمت پایین‌تر مهم‌تر است یا گارانتی و امتیاز؟ اگر shop_id خاصی مدنظرتان است بفرمایید."
            if force_shop_question
            else "چه برند/مدل یا ویژگی مشخصی (اندازه/رنگ/کد مدل) مدنظرتان است؟"
        )
        return content or default_q
    except Exception:
        return (
            "برای فروشنده، قیمت پایین‌تر مهم‌تر است یا گارانتی و امتیاز؟ اگر shop_id خاصی مدنظرتان است بفرمایید."
            if force_shop_question
            else "چه برند/مدل یا ویژگی مشخصی (اندازه/رنگ/کد مدل) مدنظرتان است؟"
        )


