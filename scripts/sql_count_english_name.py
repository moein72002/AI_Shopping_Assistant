import os
import sqlite3
import sys


def get_database_path() -> str:
    datasets_path = os.path.dirname("/datasets/")
    return os.path.join(datasets_path, "torob.db")


def count_non_empty_english_name(database_path: str) -> int:
    query = (
        "SELECT COUNT(*) FROM base_products "
        "WHERE english_name IS NOT NULL AND LENGTH(TRIM(english_name)) > 0"
    )
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        if result is None:
            return 0
        return int(result[0])
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        db_path = get_database_path()
        count = count_non_empty_english_name(db_path)
        print(f"COUNT={count}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
