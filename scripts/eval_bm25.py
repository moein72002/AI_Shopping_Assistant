import json
import os
import sys
import sqlite3
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.bm25_tool import bm25_search


def _fetch_names(db_path: str, keys: List[str]) -> Dict[str, str]:
    if not keys:
        return {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(keys))
    q = f"SELECT random_key, COALESCE(persian_name,'') as pn, COALESCE(english_name,'') as en FROM base_products WHERE random_key IN ({placeholders})"
    cur.execute(q, keys)
    rows = cur.fetchall()
    conn.close()
    out: Dict[str, str] = {}
    for rk, pn, en in rows:
        name = pn if pn else en
        out[rk] = name
    return out


def main():
    datasets_path = os.path.dirname("/datasets/")
    datasets_path = os.path.dirname(datasets_path)
    req_path = os.path.join(datasets_path, "server_tests", "scenarios_1_to_5_requests.json")
    db_path = os.path.join(datasets_path, "torob.db")
    with open(req_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    total = 0
    printed = 0
    for it in items:
        total += 1
        try:
            msg = it.get("Last Query") or it.get("messages", [{}])[-1].get("content")
            if not msg:
                continue
            res = bm25_search(msg, k=5)
            if printed < 20:
                print(f"Q: {msg[:120]}...")
                print(f"BM25 -> {res}\n")
                if res:
                    name_map = _fetch_names(db_path, res)
                    for rk in res:
                        nm = name_map.get(rk, "<name not found>")
                        print(f"  {rk} -> {nm[:120]}")
                    print()
                printed += 1
        except Exception as e:
            print(f"Error: {e}")

    print(f"Processed {total} items.")


if __name__ == "__main__":
    main()



