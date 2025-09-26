import os
import sys
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.bm25_tool import bm25_search  # type: ignore


def pick_sample_query(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    row = c.execute(
        "SELECT COALESCE(persian_name,'') || ' ' || COALESCE(english_name,'') FROM base_products WHERE persian_name IS NOT NULL AND length(persian_name)>0 LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else "سلام گوشی آیفون ۱۶ پرومکس"


def main():
    os.environ.setdefault("BM25_LIMIT_DOCS", os.environ.get("LIMIT", "15000"))
    os.environ.setdefault("BM25_DEBUG", "1")
    q = os.environ.get("Q") or pick_sample_query("/datasets/", "torob.db"))
    print("Query:", q[:200])
    res = bm25_search(q, k=5)
    print("BM25 result:", res)


if __name__ == "__main__":
    main()


