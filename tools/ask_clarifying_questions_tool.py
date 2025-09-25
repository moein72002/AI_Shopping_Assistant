import os
from typing import Optional


def ask_clarifying_questions(context: str) -> str:
    """
    Generate ONE short Persian clarifying question to narrow down the user's needs.
    The `context` can include the user's last message or a brief history summary.
    """
    try:
        if os.environ.get("DISABLE_LLM_FOR_TESTS") == "1":
            return "دقیق‌تر می‌فرمایید چه برندی و حدود بودجه‌ای مدنظر شماست؟"

        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"), timeout=8)
        sys_prompt = (
            "English: Ask ONE short clarifying question in Persian to help select a product.\n"
            "Keep it focused (<= 20 words).\n\n"
            "فارسی: برای انتخاب محصول، فقط یک سوال کوتاه و شفاف بپرس (حداکثر ۲۰ کلمه).\n"
        )
        user_msg = f"Context:\n{context or ''}\n\nAsk one Persian question only."
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_CLARIFY_MODEL", "gpt-5-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=8,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or "بودجه و برند مدنظر شما چیست؟"
    except Exception:
        return "بودجه و برند مدنظر شما چیست؟"


