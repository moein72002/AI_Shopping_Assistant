import argparse
import os
import sqlite3
from typing import List, Tuple

import numpy as np

import bm25s
from utils.utils import _append_chat_log, _base_names_for_keys


def _compose_text(persian: str | None, english: str | None) -> str:
    parts = []
    if english and english.strip():
        parts.append(str(english))
    if persian and persian.strip():
        parts.append(str(persian))
    return " ".join(parts).strip()


def load_products(db_path: str, limit: int | None = None) -> Tuple[np.ndarray, List[str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        sql = "SELECT random_key, persian_name, english_name FROM base_products"
        if isinstance(limit, int) and limit > 0:
            sql += f" LIMIT {int(limit)}"
        cur.execute(sql)
        ids: List[str] = []
        texts: List[str] = []
        for row in cur.fetchall():
            pid = row["random_key"]
            text = _compose_text(row["persian_name"], row["english_name"]) or ""
            ids.append(str(pid))
            texts.append(text)
        return np.asarray(ids, dtype=str), texts
    finally:
        conn.close()


class BM25ProductSearcher:
    def __init__(self, ids: np.ndarray, texts: List[str]) -> None:
        self.ids = np.asarray(ids)
        self.texts = texts
        # Tokenize and index
        corpus_tokens = bm25s.tokenize(self.texts, stopwords=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def top_k_ids(self, query: str, k: int = 5) -> List[int]:
        # Tokenize query (compatible with bm25s.retrieve)
        query_tokens = bm25s.tokenize(query, stopwords=False)
        docs, scores = self.retriever.retrieve(
            query_tokens, corpus=self.ids, k=k, sorted=True
        )
        # docs shape: (num_queries, k) -> here num_queries = 1
        return [str(x) for x in docs[0].tolist()]


# --- Simple module-level API ---
_GLOBAL_SEARCHER: BM25ProductSearcher | None = None


def _ensure_searcher() -> BM25ProductSearcher:
    global _GLOBAL_SEARCHER
    if _GLOBAL_SEARCHER is None:
        # Optional environment override to limit rows during initialization for faster startup
        limit_env = os.environ.get("BM25_LIMIT")
        try:
            limit_val = int(limit_env) if limit_env is not None else None
        except ValueError:
            limit_val = None

        ids, texts = load_products("torob.db", limit=limit_val)
        if len(ids) == 0:
            raise RuntimeError("No products loaded from database.")
        _GLOBAL_SEARCHER = BM25ProductSearcher(ids, texts)
    return _GLOBAL_SEARCHER


def bm25_search(chat_id: str, query: str, k: int = 5) -> List[str]:
    """Return the `random_key` values of the top 5 products for the given query."""
    searcher = _ensure_searcher()
    results = searcher.top_k_ids(query, k=k)
    name_map = _base_names_for_keys(results or [])
    _append_chat_log(chat_id, {"stage": "tool_result", "results": results[:10] if results else [], "names": name_map})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25S search over base_products names")
    parser.add_argument("--db", default="torob.db", help="Path to SQLite database file")
    parser.add_argument("--query", required=True, help="Search query (Persian or English)")
    parser.add_argument("--k", type=int, default=5, help="Top-K products to return (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to index (for quick tests)")
    args = parser.parse_args()

    ids, texts = load_products(args.db, limit=args.limit)
    if len(ids) == 0:
        raise SystemExit("No products loaded from database.")

    searcher = BM25ProductSearcher(ids, texts)
    top_ids = searcher.top_k_ids(args.query, k=args.k)


if __name__ == "__main__":
    main()
