import argparse
import json
import os
import sqlite3
import sys
from typing import Any, List, Tuple


def detect_db_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "torob.db")


def get_columns(cur: sqlite3.Cursor, table: str) -> List[str]:
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return [r[1] for r in rows]


def row_to_dict(row: Tuple[Any, ...], cols: List[str]) -> dict:
    return {col: row[i] for i, col in enumerate(cols)}


def pretty(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    # Try JSON parse for fields like extra_features
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return text


def show_related_for_base(cur: sqlite3.Cursor, base_random_key: str, limit: int = 10) -> None:
    try:
        print("\nMembers (first {}):".format(limit))
        cur.execute(
            "SELECT random_key, shop_id, price FROM members WHERE base_random_key = ? LIMIT ?",
            (base_random_key, limit),
        )
        rows = cur.fetchall()
        for rk, shop_id, price in rows:
            print(f"  - member.random_key={rk} shop_id={shop_id} price={price}")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Show all columns of a row by id from torob.db")
    parser.add_argument("id", help="Identifier value (default assumes base_products.random_key)")
    parser.add_argument("--table", default="base_products", help="Table name (default: base_products)")
    parser.add_argument("--key-col", default="random_key", help="Key column name (default: random_key)")
    parser.add_argument("--db", default=detect_db_path(), help="Path to torob.db (auto-detected)")
    parser.add_argument("--limit-related", type=int, default=10, help="Limit for related rows display")
    parser.add_argument("--name-only-json", action="store_true", help="Print {random_key,name} JSON and exit")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: Database not found at {args.db}")
        return 1

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Validate table exists
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (args.table,),
    )
    if cur.fetchone() is None:
        print(f"ERROR: Table '{args.table}' not found in {args.db}")
        conn.close()
        return 1

    cols = get_columns(cur, args.table)
    if args.key_col not in cols:
        print(f"ERROR: Column '{args.key_col}' not found in table '{args.table}'")
        conn.close()
        return 1

    cur.execute(
        f"SELECT {', '.join(cols)} FROM {args.table} WHERE {args.key_col} = ?",
        (args.id,),
    )
    row = cur.fetchone()
    if row is None:
        print(f"No row found in '{args.table}' where {args.key_col} = {args.id}")
        conn.close()
        return 2

    data = row_to_dict(row, cols)

    # Fast path: just print JSON name and exit
    if args.name_only_json:
        name = data.get("persian_name") or data.get("english_name") or ""
        print(json.dumps({"random_key": args.id, "name": name}, ensure_ascii=False))
        conn.close()
        return 0
    print(f"Table: {args.table}")
    print(f"Key: {args.key_col} = {args.id}")
    print("\nColumns:")
    for k in cols:
        val = data.get(k)
        rendered = pretty(val)
        # Indent multi-line values
        if "\n" in rendered:
            print(f"- {k}:")
            for line in rendered.splitlines():
                print(f"    {line}")
        else:
            print(f"- {k}: {rendered}")

    # If base_products, show related members
    if args.table == "base_products" and args.key_col == "random_key":
        show_related_for_base(cur, args.id, limit=args.limit_related)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())


