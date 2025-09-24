import json
import os
from typing import List

from .bm25_tool import bm25_search


def _llm_expand_queries(user_query: str) -> List[str]:
    """Generate up to 3 short lexical sub-queries via LLM; fallback to heuristics on error.

    Bilingual system prompt (English + Persian) emphasizes preserving alphanumeric codes
    and numbers verbatim, and splitting multi-product queries.
    """
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=os.environ.get("TOROB_PROXY_URL"))
        sys_prompt = (
            "English Prompt: \n"
            "You refine shopping search queries for BM25 lexical retrieval on Persian product names. "
            "Given a Persian or English user query, output up to 3 short, highly lexical sub-queries. "
            "Make sure each sub-query is likely to be a product name with some of its specs.\n"
            "Rules: \n"
            "- Sometimes multiple products are mentioned (comparisons). Output separate sub-queries capturing each.\n"
            "- Keep only crucial tokens: brand, model, EXPLICIT ALPHANUMERIC CODES and NUMBERS (e.g., NANO75, A3891, GTV-50RU766S, S25, 50, 1200), sizes, colors, key descriptors.\n"
            "- ALWAYS preserve alphanumeric codes and pure numbers verbatim; do not translate, spell out, or remove them.\n"
            "- If a Torob base_random_key (6 lower-case letters, e.g., 'bmubxu') or a member key (UUID) appears, return that base_random_key.\n"
            "- Remove filler words. Keep queries concise (1-5 words).\n"
            "- Output strict JSON: {\"queries\": [\"...\", ...]} with 1-3 items.\n"
            "\n"
            "Persian Prompt: \n"
            "شما جستارهای جستجوی خرید را برای بازیابی واژگانی BM25 بر روی نام محصولات فارسی پالایش می‌کنید.\n"
            "با دریافت یک جستار کاربر به زبان فارسی یا انگلیسی، حداکثر ۳ زیرجستار کوتاه و کاملاً واژگانی خروجی دهید. اطمینان حاصل کنید که هر زیرجستار احتمالاً نام یک محصول به همراه برخی از مشخصات آن باشد.\n"
            "قوانین: \n"
            "- اگر به چندین محصول اشاره شده (مقایسه)، زیرجستارهای جداگانه‌ای برای هر کدام خروجی دهید.\n"
            "- فقط توکن‌های حیاتی را نگه دارید: برند، مدل، کدهای حروفی-عددی صریح، رنگ، اندازه، تعداد و اعداد (مانند NANO75، A3891، GTV-50RU766S، S25، 50، 1200)، توصیف‌گرهای کلیدی.\n"
            "- همیشه کدهای حروفی-عددی و اعداد خالص را به صورت دقیق حفظ کنید؛ آن‌ها را ترجمه یا حذف نکنید.\n"
            "- اگر base_random_key ترب (۶ حرف کوچک) یا UUID ظاهر شد، همان base_random_key را برگردانید.\n"
            "- کلمات پرکننده را حذف کنید. زیرجستارها را کوتاه (۱ تا ۵ کلمه) نگه دارید.\n"
            "- خروجی JSON دقیق باشد: {\"queries\": [\"...\", ...]} با ۱ تا ۳ آیتم."
        )
        user_msg = (
            "User query:"\
            f"\n{user_query}\n"\
            "Return only JSON."
        )
        model = os.environ.get("LLM_BM25_MODEL", "gpt-5-nano")
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=12,
        )
        content = resp.choices[0].message.content
        data = json.loads(content or "{}")
        q = data.get("queries") or []
        if isinstance(q, list):
            clean: List[str] = []
            seen = set()
            for s in q:
                if not isinstance(s, str):
                    continue
                s2 = s.strip()
                if s2 and s2 not in seen:
                    clean.append(s2)
                    seen.add(s2)
            if clean:
                return clean[:3]
        return []
    except Exception:
        # Heuristic fallback
        import re

        tokens = user_query.split()
        keep: List[str] = []
        for t in tokens:
            t2 = t.strip().strip("،,.؛:()\"'")
            if not t2:
                continue
            if re.fullmatch(r"[a-z]{6}", t2):  # base_random_key
                keep.append(t2)
            elif re.fullmatch(r"[0-9a-fA-F-]{12,}", t2):  # uuid-like
                keep.append(t2)
            elif re.search(r"[A-Za-z]{2,}\d+|\d+[A-Za-z]{2,}", t2):  # model-like
                keep.append(t2)
            elif re.search(r"[A-Za-z]+-\w+", t2):  # hyphenated code
                keep.append(t2)
            elif re.fullmatch(r"\d+", t2):  # pure numbers
                keep.append(t2)
            elif re.fullmatch(r"[A-Za-zآ-ی]+", t2) and len(t2) >= 3:
                keep.append(t2)
        if keep:
            return [" ".join(keep[:12])]
        return [user_query]


def bm25_llm_search(query: str, k: int = 10) -> List[str]:
    """LLM-augmented BM25S search. Returns base_random_keys.

    - Expands the user's query into up to 3 concise sub-queries (codes/numbers preserved)
    - Runs bm25s for each sub-query
    - Merges and deduplicates results, preserving order
    """
    query = (query or "").strip()
    if not query:
        return []

    sub_qs = _llm_expand_queries(query) or [query]

    merged: List[str] = []
    seen = set()
    per_q_k = min(k, 10)
    for sq in sub_qs:
        try:
            hits = bm25_search(sq, k=per_q_k) or []
            for rk in hits:
                if rk not in seen:
                    merged.append(rk)
                    seen.add(rk)
        except Exception:
            continue
        if len(merged) >= k:
            break
    return merged[:k]


