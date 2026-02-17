import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
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


def setup_logger(log_path: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("run_scenarios_sample")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def post_chat(base_url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat"
    started_at = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    content_type = r.headers.get("content-type", "")
    return {
        "status": r.status_code,
        "elapsed_ms": elapsed_ms,
        "request_id": r.headers.get("x-request-id"),
        "content_type": content_type,
        "data": r.json() if content_type.startswith("application/json") else r.text,
    }


def main() -> int:
    cpu_count = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description="Run 5 random /chat requests from the scenarios file")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"), help="Service base URL (default: http://localhost:8000)")
    # parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://moein-ai-shopping-assistant.darkube.app"), help="Service base URL (default: http://localhost:8000)")
    parser.add_argument("--file", default=os.path.join("server_tests", "scenario1.json"), help="Path to scenarios JSON file")
    parser.add_argument("--limit", type=int, default=20, help="Max number of samples to run (default: 20)")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30)")
    parser.add_argument("--workers", type=int, default=max(1, min(cpu_count, 4)), help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--logs-dir", default=os.path.join("logs", "scenario_runs"), help="Directory to write scenario run logs/results")
    parser.add_argument("--run-id", default=None, help="Optional run identifier used in output filenames")
    parser.add_argument("--verbose", action="store_true", help="Print debug-level logs to console")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.logs_dir, exist_ok=True)
    run_log_path = os.path.join(args.logs_dir, f"run_{run_id}.log")
    run_results_path = os.path.join(args.logs_dir, f"run_{run_id}.json")
    logger = setup_logger(run_log_path, verbose=args.verbose)

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
    print(f"Run ID: {run_id}")
    print(f"Detailed log: {run_log_path}")
    logger.info(
        "Run started | base_url=%s file=%s samples=%s workers=%s timeout=%s",
        args.base_url,
        args.file,
        len(tasks),
        args.workers,
        args.timeout,
    )

    def _worker(idx: int, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Dict[str, Any] | None, Dict[str, Any] | None]:
        started_at = time.perf_counter()
        logger.info(
            "[%s] Request start | chat_id=%s last_msg=%s",
            idx,
            payload.get("chat_id"),
            (payload.get("messages") or [{}])[-1].get("content", "")[:200],
        )
        logger.debug("[%s] Full payload: %s", idx, json.dumps(payload, ensure_ascii=False))
        try:
            resp = post_chat(args.base_url, payload, timeout=args.timeout)
            logger.info(
                "[%s] Request done | status=%s elapsed_ms=%s request_id=%s",
                idx,
                resp.get("status"),
                resp.get("elapsed_ms"),
                resp.get("request_id"),
            )
            logger.debug("[%s] Response body: %s", idx, json.dumps(resp, ensure_ascii=False))
            return idx, payload, resp, None
        except Exception as e:
            err = {
                "type": type(e).__name__,
                "error": str(e),
                "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                "traceback": traceback.format_exc(),
            }
            logger.error("[%s] Request failed | %s: %s", idx, err["type"], err["error"])
            logger.debug("[%s] Error traceback: %s", idx, err["traceback"])
            return idx, payload, None, err

    results: List[Tuple[int, Dict[str, Any], Dict[str, Any] | None, Dict[str, Any] | None]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        future_map = {ex.submit(_worker, idx, payload): (idx, payload) for idx, payload in tasks}
        for fut in as_completed(future_map):
            idx, payload = future_map[fut]
            try:
                res = fut.result()
            except Exception as e:
                err = {
                    "type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                logger.error("[%s] Future failed | %s: %s", idx, err["type"], err["error"])
                logger.debug("[%s] Future traceback: %s", idx, err["traceback"])
                res = (idx, payload, None, err)
            results.append(res)

    # Print results in original input order
    results.sort(key=lambda t: t[0])
    success = 0
    failed = 0
    serialized_results: List[Dict[str, Any]] = []
    for idx, payload, resp, err in results:
        last = (payload.get("messages") or [{}])[-1].get("content", "")
        print(f"\n[{idx}] Chat ID: {payload['chat_id']}")
        print(f"> {str(last)[:200]}")
        if err is not None:
            failed += 1
            print(json.dumps({"status": "error", **err}, ensure_ascii=False))
            serialized_results.append(
                {
                    "index": idx,
                    "chat_id": payload.get("chat_id"),
                    "payload": payload,
                    "error": err,
                }
            )
        else:
            success += 1
            print(json.dumps(resp or {}, ensure_ascii=False, indent=2))
            serialized_results.append(
                {
                    "index": idx,
                    "chat_id": payload.get("chat_id"),
                    "payload": payload,
                    "response": resp,
                }
            )

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "file": args.file,
        "samples": len(tasks),
        "workers": args.workers,
        "timeout": args.timeout,
        "success": success,
        "failed": failed,
        "results": serialized_results,
    }
    with open(run_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "Run finished | success=%s failed=%s results_file=%s",
        success,
        failed,
        run_results_path,
    )
    print(f"\nSaved run results: {run_results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
