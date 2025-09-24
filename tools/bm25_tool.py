import os
import re
import sqlite3
from typing import List

_bm25_cache = {
    "ready": False,
    "doc_ids": None,
    "bm25": None,
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")


def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t]


def _load_bm25_assets(limit: int | None = None):
    if _bm25_cache["ready"]:
        return
    import bm25s

    # Allow limiting docs via env for quicker cold-start/eval
    if limit is None:
        try:
            env_limit = os.getenv("BM25_LIMIT_DOCS")
            if env_limit:
                limit = int(env_limit)
        except Exception:
            pass

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    q = (
        "SELECT random_key, "
        "COALESCE(persian_name,'') || ' ' || COALESCE(english_name,'') || ' ' || COALESCE(extra_features,'') "
        "FROM base_products"
    )
    if limit and limit > 0:
        q += " LIMIT ?"
        cur.execute(q, (limit,))
    else:
        cur.execute(q)
    rows = cur.fetchall()
    conn.close()

    doc_ids = [r[0] for r in rows]
    # Include the random_key in the text to catch code/id queries directly
    corpus = [f"{rk} {txt}" for rk, txt in rows]
    # Use a Unicode-friendly tokenizer for Persian/Arabic text
    corpus_tokens = [_simple_tokenize(text) for text in corpus]
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)

    _bm25_cache.update({
        "ready": True,
        "doc_ids": doc_ids,
        "bm25": retriever,
    })
    if os.getenv("BM25_DEBUG"):
        try:
            print(f"[bm25] indexed_docs={len(doc_ids)} limit={limit}")
            print(f"[bm25] sample_doc_id={doc_ids[0] if doc_ids else 'NA'}")
        except Exception:
            pass


def bm25_search(query: str, k: int = 5) -> List[str]:
    try:
        if not _bm25_cache["ready"]:
            # Build once, lazily
            _load_bm25_assets()

        # Tokenize query with the same Unicode-friendly tokenizer
        query_tokens = [_simple_tokenize(query)]  # bm25s expects List[List[str]]
        if os.getenv("BM25_DEBUG"):
            qt = query_tokens[0] if query_tokens else []
            print(f"[bm25] q_tokens={len(qt)} example={(qt[:5]) if qt else []}")
        docs, scores = _bm25_cache["bm25"].retrieve(query_tokens, k=k)

        # docs may be indices or strings depending on version/config
        first = None
        try:
            first = docs[0][0]
        except Exception:
            pass

        if isinstance(first, int):
            idxs = list(docs[0])
            out = [_bm25_cache["doc_ids"][i] for i in idxs if 0 <= i < len(_bm25_cache["doc_ids"])]
            if os.getenv("BM25_DEBUG"):
                print(f"[bm25] hits={len(out)} ids={out[:5]}")
            return out
        else:
            # Assume strings
            id_map = {text: _bm25_cache["doc_ids"][i] for i, text in enumerate(_bm25_cache["bm25"].corpus)}
            out = []
            for d in docs[0]:
                rk = id_map.get(d)
                if rk:
                    out.append(rk)
            if os.getenv("BM25_DEBUG"):
                print(f"[bm25] hits={len(out)} ids={out[:5]}")
            return out
    except Exception as e:
        if os.getenv("BM25_DEBUG"):
            try:
                print(f"[bm25] exception: {e}")
            except Exception:
                pass
        return []


