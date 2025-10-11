import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def item_to_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    chat_id = item.get("Chat ID") or item.get("chat_id") or f"cli-{random.randrange(10**8):08d}"
    if "messages" in item and isinstance(item["messages"], list) and item["messages"]:
        messages = item["messages"]
    else:
        last = item.get("Last Query") or "ping"
        messages = [{"type": "text", "content": last}]
    return {"chat_id": chat_id, "messages": messages}


def post_chat(base_url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat"
    r = requests.post(url, json=payload, timeout=timeout)
    return {"status": r.status_code, "data": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 5 random /chat requests from the scenarios file")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"), help="Service base URL (default: http://localhost:8000)")
    # parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://moein-ai-shopping-assistant.darkube.app"), help="Service base URL (default: http://localhost:8000)")
    parser.add_argument("--file", default=os.path.join("server_tests", "scenario1.json"), help="Path to scenarios JSON file")
    parser.add_argument("--limit", type=int, default=20, help="Max number of samples to run (default: 20)")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30)")
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count(), 4)), help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    items = load_items(args.file)
    if not isinstance(items, list) or len(items) == 0:
        print("No items found in scenarios file.")
        return 1

    # Limit deterministically (first N)
    limit = max(1, int(args.limit or 1))
    items = items[: min(limit, len(items))]

    # Prepare tasks
    tasks: List[Tuple[int, Dict[str, Any]]] = []
    for i, it in enumerate(items, 1):
        tasks.append((i, item_to_payload(it)))

    print(f"Base URL: {args.base_url}")
    print(f"Total samples: {len(tasks)} | Workers: {args.workers}")

    def _worker(idx: int, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Dict[str, Any] | None, str | None]:
        try:
            resp = post_chat(args.base_url, payload, timeout=args.timeout)
            return idx, payload, resp, None
        except Exception as e:
            return idx, payload, None, str(e)

    results: List[Tuple[int, Dict[str, Any], Dict[str, Any] | None, str | None]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        future_map = {ex.submit(_worker, idx, payload): (idx, payload) for idx, payload in tasks}
        for fut in as_completed(future_map):
            idx, payload = future_map[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = (idx, payload, None, str(e))
            results.append(res)

    # Print results in original input order
    results.sort(key=lambda t: t[0])
    for idx, payload, resp, err in results:
        last = (payload.get("messages") or [{}])[-1].get("content", "")
        print(f"\n[{idx}] Chat ID: {payload['chat_id']}")
        print(f"> {str(last)[:200]}")
        if err is not None:
            print(json.dumps({"status": "error", "error": err}, ensure_ascii=False))
        else:
            print(json.dumps(resp or {}, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())


