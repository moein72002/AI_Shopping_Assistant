import os
import sys
import json
import uuid
import argparse
from typing import List, Dict, Any

import requests


def post_chat(base_url: str, chat_id: str, messages: List[Dict[str, Any]], timeout: int = 30) -> Dict[str, Any]:
    url = base_url.rstrip('/') + "/chat"
    payload = {"chat_id": chat_id, "messages": messages}
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    return {"status": resp.status_code, "data": data}


def run_ping(base_url: str) -> None:
    chat_id = f"cli-{uuid.uuid4().hex[:8]}"
    messages = [{"type": "text", "content": "ping"}]
    result = post_chat(base_url, chat_id, messages)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def interactive(base_url: str, chat_id: str | None) -> None:
    if not chat_id:
        chat_id = f"cli-{uuid.uuid4().hex[:8]}"
    print(f"Base URL: {base_url}")
    print(f"Chat ID: {chat_id}")
    print("Type your message and press Enter. Empty line to exit.")

    history: List[Dict[str, Any]] = []
    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break
        if not user_text:
            break

        history.append({"type": "text", "content": user_text})
        result = post_chat(base_url, chat_id, history)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # If the assistant returned final keys, you may choose to end the session
        data = result.get("data", {})
        if data.get("base_random_keys") or data.get("member_random_keys"):
            print("Session ended by assistant with product keys.")
            break


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Play with the /chat endpoint interactively or via quick checks")
    # parser.add_argument("--url", default=os.environ.get("CHAT_BASE_URL", "https://moein-ai-shopping-assistant.darkube.app"), help="Base URL of the API (default: %(default)s)")
    parser.add_argument("--url", default=os.environ.get("CHAT_BASE_URL", "http://localhost:8080"), help="Base URL of the API (default: %(default)s)")
    parser.add_argument("--chat-id", default="play_chat", help="Optional fixed chat_id to reuse across turns")
    parser.add_argument("--ping", action="store_true", help="Send a single 'ping' request and print the result")
    parser.add_argument("--message", default=None, help="Send a single message and print the result, then exit.")
    args = parser.parse_args(argv)

    if args.ping:
        run_ping(args.url)
        return 0

    if args.message:
        chat_id = args.chat_id or f"cli-{uuid.uuid4().hex[:8]}"
        messages = [{"type": "text", "content": args.message}]
        result = post_chat(args.url, chat_id, messages)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    interactive(args.url, args.chat_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


