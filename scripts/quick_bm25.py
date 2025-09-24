import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.bm25_tool import bm25_search, _bm25_cache  # type: ignore


def main():
    query = os.environ.get("Q", "سلام گوشی آیفون ۱۶ پرومکس")
    k = int(os.environ.get("K", "5"))
    os.environ.setdefault("BM25_LIMIT_DOCS", os.environ.get("LIMIT", "20000"))
    os.environ.setdefault("BM25_DEBUG", "1")
    res = bm25_search(query, k=k)
    print("BM25 result:", res)
    bm = _bm25_cache.get("bm25")
    if bm is not None:
        try:
            corpus = getattr(bm, "corpus", None)
            print("indexed_docs:", len(corpus) if corpus is not None else "NA")
        except Exception:
            pass


if __name__ == "__main__":
    main()


