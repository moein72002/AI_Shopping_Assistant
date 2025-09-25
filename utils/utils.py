import os
import json
from datetime import datetime
from typing import Dict, List as _List
import sqlite3

# --- Per-chat logging helpers ---
def _append_chat_log(chat_id: str, record: dict):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "..", "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, f"{chat_id}.log")
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            **record,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never fail request due to logging issues
        pass

def _reset_chat_log(chat_id: str):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "..", "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, f"{chat_id}.log")
        # Truncate (overwrite) the log file for this chat_id at the start of each request
        with open(path, "w", encoding="utf-8"):
            pass
    except Exception:
        pass


def _base_names_for_keys(keys: _List[str]) -> Dict[str, str]:
    """Return a mapping from `random_key` to product name.

    Preference: use `persian_name` if non-empty, else `english_name`.
    """
    if not keys:
        return {}

    db_path = os.path.join(os.path.dirname(__file__), "..", "torob.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        # First fetch all unique ids -> name into a lookup
        lookup: Dict[str, str] = {}

        unique_keys = list(dict.fromkeys(keys))
        chunk_size = 900
        for i in range(0, len(unique_keys), chunk_size):
            chunk = unique_keys[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = (
                "SELECT random_key, persian_name, english_name "
                "FROM base_products WHERE random_key IN (" + placeholders + ")"
            )
            cur.execute(sql, chunk)
            for row in cur.fetchall():
                rid = str(row["random_key"])  # ensure str keys
                pn = (row["persian_name"] or "").strip()
                en = (row["english_name"] or "").strip()
                name = pn if pn else en
                lookup[rid] = name

        # Now build an ordered mapping following the original input order
        ordered: Dict[str, str] = {}
        for k in keys:
            ordered[str(k)] = lookup.get(str(k), "")

        return ordered
    finally:
        conn.close()
