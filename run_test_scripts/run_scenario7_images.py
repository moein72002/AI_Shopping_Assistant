import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

import requests


DEF_PROMPT_FA = "لطفا محصول موجود در تصویر را پیدا کن"


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_payload(item: Dict[str, Any], text_prompt: str) -> Dict[str, Any]:
    chat_id = item.get("Chat ID") or item.get("chat_id") or f"cli-{random.randrange(10**8):08d}"
    image_b64 = item.get("Last Query") or ""
    messages = [
        {"type": "text", "content": text_prompt},
        {"type": "image", "content": image_b64},
    ]
    return {"chat_id": chat_id, "messages": messages}


def post_chat(base_url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat"
    r = requests.post(url, json=payload, timeout=timeout)
    return {
        "status": r.status_code,
        "data": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Scenario 7 image-based requests")
    # parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"), help="Service base URL")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://moein-ai-shopping-assistant.darkube.app"), help="Service base URL")
    parser.add_argument("--file", default=os.path.join("server_tests", "scenario7.json"), help="Path to scenario7.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0 = all)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--prompt", default=DEF_PROMPT_FA, help="First text message to send before the image")
    args = parser.parse_args()

    items = load_items(args.file)
    if not isinstance(items, list) or len(items) == 0:
        print("No items found in scenarios file.")
        return 1

    sample = items if args.limit in (0, None) else items[: max(0, min(args.limit, len(items)))]

    print(f"Base URL: {args.base_url}")
    for i, it in enumerate(sample, 1):
        payload = build_payload(it, args.prompt)
        print(f"\n[{i}] Chat ID: {payload['chat_id']}")
        try:
            resp = post_chat(args.base_url, payload, timeout=args.timeout)
            print(json.dumps(resp, ensure_ascii=False, indent=2))
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())


