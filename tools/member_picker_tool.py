import os
import sqlite3
from typing import Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")


def pick_best_member_for_base(base_random_key: str) -> Optional[str]:
    """
    Return the best member_random_key (shop offer) for a base product.
    Heuristic: prioritize lower price; break ties by higher shop score; then warranty.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.random_key, m.price, s.score, s.has_warranty
            FROM members m
            JOIN shops s ON m.shop_id = s.id
            WHERE m.base_random_key = ?
            ORDER BY m.price ASC, s.score DESC, s.has_warranty DESC
            LIMIT 1
            """,
            (base_random_key,),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
        return None
    except Exception:
        return None


