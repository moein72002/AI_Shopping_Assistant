import os
import re
import sqlite3
from typing import Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")


def extract_product_id(query: str) -> Optional[str]:
    """
    Extract a 6-letter Torob base_random_key (lowercase a-z) from the query and
    validate it against the database.
    """
    try:
        m = re.search(r"\b[a-z]{6}\b", query or "")
        if not m:
            return None
        cand = m.group(0)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM base_products WHERE random_key=? LIMIT 1", (cand,))
        ok = cur.fetchone() is not None
        conn.close()
        return cand if ok else None
    except Exception:
        return None


