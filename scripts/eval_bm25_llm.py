import json
import os
import random
import sqlite3
import sys
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.bm25_tool import bm25_search  # type: ignore


def _fetch_names(db_path: str, keys: List[str]) -> Dict[str, str]:
    if not keys:
        return {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(keys))
    q = (
        f"SELECT random_key, COALESCE(persian_name,'') as pn, COALESCE(english_name,'') as en "
        f"FROM base_products WHERE random_key IN ({placeholders})"
    )
    cur.execute(q, keys)
    rows = cur.fetchall()
    conn.close()
    out: Dict[str, str] = {}
    for rk, pn, en in rows:
        name = pn if pn else en
        out[rk] = name
    return out


def _llm_expand_queries(user_query: str) -> List[str]:
    try:
        # Lightweight LLM call; expected to use proxy via env.
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        sys_prompt = (
            "English Prompt: \n"
            "You refine shopping search queries for BM25 lexical retrieval on Persian product names. "
            "Given a Persian or English user query, output up to 3 short, highly lexical sub-queries. Make sure each sub-query is likely to be a product name with some of its specs."
            "Rules: \n"
            "- Handle Comparisons by Generating Multiple Queries: If the user's query compares two or more products (e.g., \"iphone 13 vs s22\" or \"which is better, A or B\"), you must generate a separate, distinct sub-query for each product mentioned to enable individual retrieval and comparison.\n"
            "- Sometime multiple products are mentioned, and the query wants to compare them, so you should output separate sub-queries capturing each to help the BM25 retrieve the products.\n"
            "- If multiple products are mentioned, output separate sub-queries capturing each.\n"
            "- Always when there is a comparison query (which one is better, etc.), make sure to output separate sub-queries capturing each to help the BM25 retrieve the products.\n"
            "- Keep only crucial tokens: brand, model, EXPLICIT ALPHANUMERIC CODES, English Expressions, color, size, quantity and NUMBERS (e.g., NANO75, A3891, GTV-50RU766S, S25, 50, 1200), sizes, colors, key descriptors.\n"
            "- If there exists an English expression (code) like 50RU766S, include it verbatim in one sub-query with only this one word.\n"
            "- ALWAYS preserve alphanumeric codes and pure numbers verbatim; do not translate, spell out, or remove them.\n"
            "- If a Torob base_random_key (6 lower-case letters, e.g., 'bmubxu') or a member key (UUID) appears, return the Torob base_random_key.\n"
            "- Remove filler words. Keep queries concise (1-5 words).\n"
            "- Output strict JSON: {\"queries\": [\"...\", \"...\"]} with 1 or 2 (when comparison is mentioned) items."
            "Persian Prompt: \n"
            "شما جستارهای جستجوی خرید را برای بازیابی واژگانی BM25 بر روی نام محصولات فارسی پالایش می‌کنید."
            "با دریافت یک جستار کاربر به زبان فارسی یا انگلیسی، حداکثر ۳ زیرجستار کوتاه و کاملاً واژگانی خروجی دهید. اطمینان حاصل کنید که هر زیرجستار احتمالاً نام یک محصول به همراه برخی از مشخصات آن باشد."
            "قوانین: \n"
            "- برای مقایسه‌ها، چند جستار تولید کنید: اگر جستار کاربر در حال مقایسه دو یا چند محصول است (مثلاً \"آیفون ۱۳ در مقابل اس ۲۲\" یا \"کدام بهتر است، الف یا ب\")، شما باید برای هر محصول یک زیرجستار مجزا و مشخص تولید کنید تا امکان بازیابی و مقایسه فراهم شود."
            "- گاهی اوقات به چندین محصول اشاره می‌شود و هدف از جستار مقایسه آنهاست، بنابراین باید برای هر کدام زیرجستارهای جداگانه‌ای خروجی دهید تا به BM25 در بازیابی محصولات کمک کنید.\n"
            "- اگر به چندین محصول اشاره شده است، زیرجستارهای جداگانه‌ای برای هر کدام خروجی دهید.\n"
            "- همیشه در جستارهای مقایسه‌ای (کدام یک بهتر است یا ...)، مطمئن شوید که برای هر محصول، زیرجستارهای جداگانه خروجی می‌دهید تا به BM25 در بازیابی آن‌ها کمک کنید.\n"
            "- فقط توکن‌های حیاتی را نگه دارید: برند، مدل، کدهای حروفی-عددی صریح، عبارات انگلیسی، رنگ، اندازه، تعداد و اعداد (مانند NANO75، A3891، GTV-50RU766S، S25، 50، 1200)، سایزها، رنگ‌ها، توصیف‌گرهای کلیدی.\n"
            "- اگر یک عبارت انگلیسی (کد) مانند 50RU766S وجود دارد، آن را به صورت دقیق در یک زیرجستار که فقط شامل همین یک کلمه است، قرار دهید.\n"
            "- همیشه کدهای حروفی-عددی و اعداد خالص را به صورت دقیق حفظ کنید؛ آن‌ها را ترجمه، به حروف یا حذف نکنید.\n"
            "- اگر یک base_random_key ترب (۶ حرف کوچک انگلیسی، مانند 'bmubxu') یا یک member key (UUID) ظاهر شد، base_random_key ترب را برگردانید.\n"
            "- کلمات پرکننده را حذف کنید. جستارها را مختصر نگه دارید (۱-۵ کلمه).\n"
            "- خروجی به صورت JSON دقیق باشد: {\"queries\": [\"...\", \"...\"]} با ۱ یا ۲ (وقتی مقایسه مورد نظر است) آیتم."
        )
        user = (
            "User query:"\
            f"\n{user_query}\n"\
            "Return only JSON."
        )
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content or "{}")
        q = data.get("queries") or []
        if isinstance(q, list):
            # Normalize and dedupe, keep non-empty
            clean = []
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
        # Fallback: naive heuristic – keep uppercase/lowercase code-like tokens and brand-like words
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
            elif re.search(r"[A-Za-z]{2,}\d+|\d+[A-Za-z]{2,}", t2):  # model-like (case-insensitive)
                keep.append(t2)
            elif re.search(r"[A-Za-z]+-\w+", t2):  # hyphenated code like GTV-50RU766S
                keep.append(t2)
            elif re.fullmatch(r"\d+", t2):  # pure numbers (sizes, counts)
                keep.append(t2)
            elif re.fullmatch(r"[A-Za-zآ-ی]+", t2) and len(t2) >= 3:
                keep.append(t2)
        if keep:
            return [" ".join(keep[:12])]
        return [user_query]


def main():
    datasets_path = os.path.dirname("/datasets/")
    datasets_path = os.path.dirname(datasets_path)
    req_path = os.path.join(datasets_path, "server_tests", "scenarios_1_to_5_requests.json")
    db_path = os.path.join(datasets_path, "torob.db")

    with open(req_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Sample 5 random queries for quick eval
    sample = random.sample(items, k=min(1000, len(items)))

    for it in sample:
        msg = it.get("Last Query") or (it.get("messages", [{}])[-1].get("content"))
        if not msg:
            continue
        print(f"Q: {msg[:200]}\n")
        sub_queries = _llm_expand_queries(msg)
        if not sub_queries:
            sub_queries = [msg]
        print("LLM queries:", sub_queries)

        # Run BM25S per subquery
        for i, sq in enumerate(sub_queries, 1):
            hits = bm25_search(sq, k=5)
            print(f"  SQ{i}: {sq}")
            print(f"  Hits: {hits}")
            if hits:
                name_map = _fetch_names(db_path, hits)
                for rk in hits:
                    nm = name_map.get(rk, "<name not found>")
                    print(f"    {rk} -> {nm[:140]}")
            print()
        print("-" * 80)


if __name__ == "__main__":
    main()


