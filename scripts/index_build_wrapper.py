import argparse
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATUS_PATH = os.path.join(PROJECT_ROOT, "index_build_status.json")


def write_status(**fields):
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(fields, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names-only", action="store_true", help="Also build names-only index")
    args = parser.parse_args()

    write_status(status="running", started_at=now_iso(), names_only=args.names_only, pid=os.getpid())

    try:
        # Import lazily so wrapper can run even if builder has transient deps
        from scripts.create_vector_index import create_vector_index
        create_vector_index(build_names_only=args.names_only)
        write_status(status="completed", completed_at=now_iso())
        return 0
    except KeyboardInterrupt:
        write_status(status="cancelled", cancelled_at=now_iso())
        return 130
    except Exception as e:
        write_status(status="error", error=str(e), failed_at=now_iso())
        return 1


if __name__ == "__main__":
    sys.exit(main())
