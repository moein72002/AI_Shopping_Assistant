import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

import requests


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
    parser.add_argument("--file", default=os.path.join("server_tests", "scenarios_1_to_5_requests.json"), help="Path to scenarios JSON file")
    parser.add_argument("--limit", type=int, default=20, help="Number of random samples to run (default: 5)")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30)")
    args = parser.parse_args()

    items = load_items(args.file)
    if not isinstance(items, list) or len(items) == 0:
        print("No items found in scenarios file.")
        return 1

    # random.seed(args.seed)
    sample = random.sample(items, k=min(args.limit, len(items)))

    print(f"Base URL: {args.base_url}")
    for i, it in enumerate(sample, 1):
        payload = item_to_payload(it)
        last = (payload.get("messages") or [{}])[-1].get("content", "")
        print(f"\n[{i}] Chat ID: {payload['chat_id']}")
        print(f"> {str(last)[:200]}")
        try:
            resp = post_chat(args.base_url, payload, timeout=args.timeout)
            print(json.dumps(resp, ensure_ascii=False, indent=2))
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())


