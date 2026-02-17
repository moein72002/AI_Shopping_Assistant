import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests


DEF_PROMPT_FA = "این تصویر چیه؟"


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
    started_at = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "status": r.status_code,
        "elapsed_ms": elapsed_ms,
        "request_id": r.headers.get("x-request-id"),
        "data": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Scenario 7 image-based requests")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"), help="Service base URL")
    # parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://moein-ai-shopping-assistant.darkube.app"), help="Service base URL")
    parser.add_argument("--file", default=os.path.join("server_tests", "scenario7.json"), help="Path to scenario7.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0 = all)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--prompt", default=DEF_PROMPT_FA, help="First text message to send before the image")
    parser.add_argument("--logs-dir", default=os.path.join("logs", "scenario7_runs"), help="Directory to write run logs")
    parser.add_argument("--run-id", default=None, help="Optional run id used in output filename")
    args = parser.parse_args()

    items = load_items(args.file)
    if not isinstance(items, list) or len(items) == 0:
        print("No items found in scenarios file.")
        return 1

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.logs_dir, exist_ok=True)
    run_log_path = os.path.join(args.logs_dir, f"run_{run_id}.json")

    sample = items if args.limit in (0, None) else items[: max(0, min(args.limit, len(items)))]

    print(f"Base URL: {args.base_url}")
    print(f"Run ID: {run_id}")
    print(f"Run log: {run_log_path}")

    results: List[Dict[str, Any]] = []
    for i, it in enumerate(sample, 1):
        payload = build_payload(it, args.prompt)
        print(f"\n[{i}] Chat ID: {payload['chat_id']}")
        entry: Dict[str, Any] = {
            "index": i,
            "chat_id": payload.get("chat_id"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request": {
                "messages_count": len(payload.get("messages", [])),
                "text_prompt": (payload.get("messages", [{}])[0].get("content", "") if payload.get("messages") else ""),
                "image_length": len((payload.get("messages", [{}, {}])[-1].get("content", "")) if payload.get("messages") else ""),
                "image_prefix": ((payload.get("messages", [{}, {}])[-1].get("content", "")) if payload.get("messages") else "")[:40],
            },
        }
        try:
            resp = post_chat(args.base_url, payload, timeout=args.timeout)
            print(json.dumps(resp, ensure_ascii=False, indent=2))
            entry["response"] = resp
        except Exception as e:
            err = {"status": "error", "error": str(e)}
            print(json.dumps(err, ensure_ascii=False))
            entry["error"] = err
        results.append(entry)

    summary = {
        "run_id": run_id,
        "base_url": args.base_url,
        "file": args.file,
        "limit": args.limit,
        "timeout": args.timeout,
        "prompt": args.prompt,
        "count": len(results),
        "results": results,
    }
    with open(run_log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved run log: {run_log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

