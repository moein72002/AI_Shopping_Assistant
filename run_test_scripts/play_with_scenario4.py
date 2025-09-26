import argparse
import json
import sys
import uuid
from typing import Any, Dict, List, Optional

import requests


def post_chat(url: str, chat_id: str, text: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "messages": [
            {
                "type": "text",
                "content": text,
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def pretty_response(resp: Dict[str, Any]) -> str:
    msg = resp.get("message")
    brk = resp.get("base_random_keys")
    mrk = resp.get("member_random_keys")
    parts: List[str] = []
    if msg:
        parts.append(f"assistant(message): {msg}")
    if brk:
        parts.append(f"assistant(base_random_keys): {brk}")
    if mrk:
        parts.append(f"assistant(member_random_keys): {mrk}")
    return "\n".join(parts) if parts else json.dumps(resp, ensure_ascii=False)


def run_flow(url: str, initial_text: str, chat_id: Optional[str] = None) -> None:
    cid = chat_id or f"scenario4-{uuid.uuid4()}"
    print(f"Using chat_id: {cid}")

    # Send the initial request
    resp = post_chat(url, cid, initial_text)
    print(pretty_response(resp))

    # Continue until we receive member_random_keys or user exits
    step = 1
    while True:
        mrk = resp.get("member_random_keys") or None
        if mrk:
            print("\nScenario 4 finalized. member_random_keys:", mrk)
            break

        msg = resp.get("message")
        if not msg:
            print("No clarifying question received. Exiting.")
            break

        try:
            user_ans = input("\nYour answer (or 'exit'): ").strip()
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        if not user_ans or user_ans.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        # Send next turn with the same chat_id
        resp = post_chat(url, cid, user_ans)
        print()
        print(pretty_response(resp))
        step += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Play interactively through Scenario 4 flow against /chat endpoint.")
    parser.add_argument("--url", default="http://localhost:8080/chat", help="Full /chat endpoint URL")
    parser.add_argument("--chat-id", default=None, help="Custom chat_id to reuse across runs")
    parser.add_argument("--text", default=None, help="Initial request text. If omitted, will prompt.")
    args = parser.parse_args()

    if args.text is None:
        try:
            initial_text = input("Initial request (e.g., 'به یک بخاری برقی خونگی نیاز دارم'): ").strip()
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(1)
    else:
        initial_text = args.text.strip()

    if not initial_text:
        print("No initial text provided.")
        sys.exit(1)

    try:
        run_flow(args.url, initial_text, chat_id=args.chat_id)
    except requests.HTTPError as e:
        print("HTTP error:", e)
        try:
            print("Response:", e.response.text)
        except Exception:
            pass
        sys.exit(2)
    except requests.RequestException as e:
        print("Request error:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()


