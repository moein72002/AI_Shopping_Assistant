import json
import os
import re
import sqlite3
from typing import List

from .bm25_tool import bm25_search
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")


def _find_bases_by_inline_keys(text: str) -> List[str]:
    """Detect inline Torob base_random_key tokens (six lowercase letters) and validate against DB."""
    try:
        candidates = list({m.group(0) for m in re.finditer(r"\b[a-z]{6}\b", text or "")})
        if not candidates:
            return []
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        placeholders = ",".join(["?"] * len(candidates))
        cur.execute(
            f"SELECT random_key FROM base_products WHERE random_key IN ({placeholders})",
            candidates,
        )
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _simple_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t]


def _extract_keyword_tokens(text: str) -> List[str]:
    tokens = text.split()
    stop = {
        "این","از","برای","چند","را","است","به","در","یا","و","می","یک","که","های","ها","چقدر","کجا","چگونه","تا","با","بدون","هستند","شود","من"
    }
    codes: List[str] = []
    numbers: List[str] = []
    words: List[str] = []
    # Capture letter-number pairs (e.g., M 051, M051, سری-051)
    pairs = list({m.group(0) for m in re.finditer(r"[A-Za-z]{1,5}\s*[-/ ]?\s*[0-9۰-۹]{1,6}", text or "")})
    for p in pairs:
        codes.append(p.strip())
    for t in tokens:
        t2 = t.strip().strip("،,.؛:()\"'")
        if not t2:
            continue
        if re.fullmatch(r"[a-z]{6}", t2):
            codes.append(t2)
        elif re.fullmatch(r"[0-9a-fA-F-]{12,}", t2):
            codes.append(t2)
        elif re.search(r"[A-Za-z]{2,}\d+|\d+[A-Za-z]{2,}", t2):
            codes.append(t2)
        elif re.search(r"[A-Za-z]+-\w+", t2):
            codes.append(t2)
        elif re.fullmatch(r"\d+", t2):
            numbers.append(t2)
        elif re.fullmatch(r"[A-Za-zآ-ی]+", t2) and len(t2) >= 3 and t2 not in stop:
            words.append(t2)
    # Prioritize codes, then numbers, then words
    # Weight longer words first to prefer brand/model over generic nouns
    ordered_words = sorted(words, key=lambda s: (-len(s), s))
    ordered = codes + numbers + ordered_words
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for t in ordered:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _select_candidate_rows_by_keywords(keywords: List[str], strict_first_n: int = 3, cap: int = 50000) -> List[tuple]:
    if not keywords:
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Build strict AND conditions for first N keywords
        use = keywords[:strict_first_n]
        where_parts = []
        params: List[str] = []
        for kw in use:
            like = f"%{kw}%"
            where_parts.append("(persian_name LIKE ? OR english_name LIKE ? OR extra_features LIKE ?)")
            params.extend([like, like, like])
        where_sql = " AND ".join(where_parts) if where_parts else "1=1"
        q = (
            "SELECT random_key, COALESCE(persian_name,'') || ' ' || COALESCE(english_name,'') || ' ' || COALESCE(extra_features,'') AS txt "
            f"FROM base_products WHERE {where_sql} LIMIT ?"
        )
        params.append(cap)
        cur.execute(q, params)
        rows = cur.fetchall()
        conn.close()
        if rows:
            return rows
        # Fallback: OR across all keywords if strict AND returned nothing
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        where_parts = []
        params = []
        for kw in keywords:
            like = f"%{kw}%"
            where_parts.append("(persian_name LIKE ? OR english_name LIKE ? OR extra_features LIKE ?)")
            params.extend([like, like, like])
        where_sql = " OR ".join(where_parts) if where_parts else "1=1"
        q = (
            "SELECT random_key, COALESCE(persian_name,'') || ' ' || COALESCE(english_name,'') || ' ' || COALESCE(extra_features,'') AS txt "
            f"FROM base_products WHERE {where_sql} LIMIT ?"
        )
        params.append(cap)
        cur.execute(q, params)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _normalize_digits(text: str) -> str:
    # Map Eastern Arabic digits to Western
    trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    return (text or "").translate(trans)


def _select_exact_match_by_and(keywords: List[str], min_terms: int = 3, cap: int = 50) -> List[str]:
    """Try exact AND matching on persian_name/english_name using top keywords.
    Returns a list of random_keys sorted by shorter name length first.
    """
    if not keywords:
        return []
    kw_norm = []
    for kw in keywords:
        k = _normalize_digits(kw.strip())
        if not k:
            continue
        kw_norm.append(k)
    # Prefer codes/numbers and longest words already ordered by _extract_keyword_tokens
    use = kw_norm[:5]
    if len(use) < min_terms:
        min_terms = len(use)
    if not use:
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Build AND clauses for first N terms
        where_parts = []
        params: List[str] = []
        for k in use[:min_terms]:
            like = f"%{k}%"
            where_parts.append("(persian_name LIKE ? OR english_name LIKE ?)" )
            params.extend([like, like])
        where_sql = " AND ".join(where_parts) if where_parts else "1=1"
        q = (
            "SELECT random_key, LENGTH(COALESCE(persian_name,'')) AS ln "
            f"FROM base_products WHERE {where_sql} ORDER BY ln ASC LIMIT ?"
        )
        params.append(cap)
        cur.execute(q, params)
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []



def _llm_expand_queries(user_query: str) -> List[str]:
    """Generate up to 3 short lexical sub-queries via LLM; fallback to heuristics on error.

    Bilingual system prompt (English + Persian) emphasizes preserving alphanumeric codes
    and numbers verbatim, and splitting multi-product queries.
    """
    def _heuristic_fallback(text: str) -> List[str]:
        import re
        tokens = text.split()
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
        return [text]

    # Test-mode bypass to avoid LLM calls when DISABLE_LLM_FOR_TESTS=1
    if os.environ.get("DISABLE_LLM_FOR_TESTS") == "1":
        return _heuristic_fallback(user_query)

    # Try to reuse the exact implementation from scripts/eval_bm25_llm.py to avoid divergence
    try:
        import importlib.util, sys
        eval_path = os.path.join(PROJECT_ROOT, "scripts", "eval_bm25_llm.py")
        if os.path.exists(eval_path):
            spec = importlib.util.spec_from_file_location("eval_bm25_llm_runtime", eval_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["eval_bm25_llm_runtime"] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "_llm_expand_queries"):
                    return mod._llm_expand_queries(user_query)  # type: ignore[attr-defined]
    except Exception:
        # Fall back silently to local implementation below
        pass

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
        model = os.environ.get("LLM_BM25_MODEL", "gpt-5-mini")
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
        return _heuristic_fallback(user_query)


def bm25_llm_search(query: str, k: int = 10) -> List[str]:
    """LLM-augmented BM25S search. Returns base_random_keys.

    - Expands the user's query into up to 3 concise sub-queries (codes/numbers preserved)
    - Runs bm25s for each sub-query
    - Merges and deduplicates results, preserving order
    """
    query = (query or "").strip()
    if not query:
        return []

    # 1) Fast path: if inline base_random_key(s) present, return them directly
    inline_hits = _find_bases_by_inline_keys(query)
    merged: List[str] = []
    seen = set()
    for rk in inline_hits:
        if rk not in seen:
            merged.append(rk)
            seen.add(rk)
            if len(merged) >= k:
                return merged[:k]

    # 1b) Exact AND DB match using extracted keywords (helps cases like "M 051 راحتیران سناتور")
    kw = _extract_keyword_tokens(query)
    exact_ids = _select_exact_match_by_and(kw)
    for rk in exact_ids:
        if rk not in seen:
            merged.append(rk)
            seen.add(rk)
            if len(merged) >= k:
                return merged[:k]

    # Always include the raw query as the first sub-query to avoid over-pruning
    sub_qs = _llm_expand_queries(query) or []
    merged_subs: List[str] = []
    seen_subs = set()
    for q in [query] + sub_qs:
        qn = (q or "").strip()
        if not qn:
            continue
        if qn not in seen_subs:
            merged_subs.append(qn)
            seen_subs.add(qn)

    # 2) Run BM25S over sub-queries and merge results
    per_q_k = min(k, 10)
    for sq in merged_subs:
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
    if merged:
        return merged[:k]

    # 3) If BM25S returns empty, do a targeted SQL keyword prefilter then retry BM25 on focused query
    keywords = _extract_keyword_tokens(query)
    rows = _select_candidate_rows_by_keywords(keywords)
    if rows:
        focused = " ".join(keywords[:12]) if keywords else query
        hits = bm25_search(focused, k=k) or []
        for rk in hits:
            if rk not in seen:
                merged.append(rk)
                seen.add(rk)
                if len(merged) >= k:
                    break
    return merged[:k]


