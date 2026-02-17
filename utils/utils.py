import os
import json
from datetime import datetime
from typing import Any, Dict, List as _List
import sqlite3

# --- Dataset path helpers ---
def get_datasets_dir() -> str:
    """Return a writable datasets directory consistent with app startup logic.

    Preference order:
    1) DATASETS_DIR env var (ensure directory exists)
    2) Fallback to ./datasets under app root
    """
    datasets_dir = os.environ.get("DATASETS_DIR") or "/datasets/"
    try:
        os.makedirs(datasets_dir, exist_ok=True)
        return datasets_dir
    except Exception:
        # Fallback to project-local datasets directory
        app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fallback_dir = os.path.join(app_root, "datasets")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


def get_db_path() -> str:
    """Return absolute path to torob.db inside the resolved datasets directory."""
    return os.path.join(get_datasets_dir(), "torob.db")

def _chat_logs_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _log_json_default(value: Any):
    """Best-effort serializer for non-JSON objects in chat logs."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, set):
        return list(value)

    # Pydantic v2
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass

    # Pydantic v1
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass

    # sqlite3.Row and row-like structures
    try:
        return dict(value)
    except Exception:
        return str(value)


# --- Per-chat logging helpers ---
def _append_chat_log(chat_id: str, record: dict):
    try:
        logs_dir = _chat_logs_dir()
        path = os.path.join(logs_dir, f"{chat_id}.log")
        global_path = os.path.join(logs_dir, "chat_debug.jsonl")
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            **record,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(payload, ensure_ascii=False, default=_log_json_default)
                + "\n"
            )
        with open(global_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"chat_id": chat_id, **payload},
                    ensure_ascii=False,
                    default=_log_json_default,
                )
                + "\n"
            )
    except Exception:
        # Never fail request due to logging issues
        pass

def _reset_chat_log(chat_id: str):
    try:
        path = os.path.join(_chat_logs_dir(), f"{chat_id}.log")
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

    db_path = get_db_path()
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
