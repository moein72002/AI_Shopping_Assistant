import os
import sys
import json
import re
import sqlite3
import random
import requests
import pytest
from xprocess import ProcessStarter
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8091"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def server(xprocess):
    class Starter(ProcessStarter):
        pattern = "Application startup complete"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        # Speed up tests by limiting BM25 docs indexed lazily
        env.setdefault("BM25_LIMIT_DOCS", "10000")
        # Avoid hitting LLM in tests; rely on heuristic fallback in bm25_llm_tool
        env.setdefault("DISABLE_LLM_FOR_TESTS", "1")
        # Force bypass router LLM to directly use bm25_llm_search for stable tests
        env.setdefault("DISABLE_ROUTER_LLM", "1")
        args = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8091",
        ]

    xprocess.ensure("server_bm25_llm", Starter)
    yield
    xprocess.getinfo("server_bm25_llm").terminate()


requires_openai = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("TOROB_PROXY_URL")),
    reason="OpenAI proxy config not set",
)


def _parse_expected_bases(response_field: str):
    # Expect formats like: "base=['chjknu', 'ppndkv']" or "base=['amrdye']"
    if not isinstance(response_field, str):
        return []
    m = re.search(r"base=\[(.*?)\]", response_field)
    if not m:
        return []
    inner = m.group(1)
    keys = re.findall(r"'([a-z]{6})'", inner)
    return keys


@requires_openai
def test_bm25_llm_on_scenarios_subset(server):
    # Ensure DB exists with expected table
    conn = sqlite3.connect(os.path.join(PROJECT_ROOT, "torob.db"))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='base_products'")
    assert cur.fetchone() is not None, "torob.db missing base_products table"
    conn.close()

    # Load scenarios file
    scenarios_path = os.path.join(PROJECT_ROOT, "server_tests", "scenarios_1_to_5_requests.json")
    with open(scenarios_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Restrict to requests with explicit expected base(s) recorded
    candidates = []
    for it in items:
        q = it.get("Last Query") or (it.get("messages", [{}])[-1].get("content"))
        resp = it.get("Response", "")
        if not isinstance(q, str):
            continue
        if "base=[" in str(resp):
            candidates.append((q, resp))
    assert candidates, "No product-finding candidates (with base=[...]) found in scenarios file"

    # Sample 5
    random.seed(0)
    sample = candidates[:5] if len(candidates) >= 5 else candidates

    for idx, (query, resp_hint) in enumerate(sample, 1):
        r = requests.post(
            f"{BASE_URL}/chat",
            json={"chat_id": f"bm25-llm-{idx}", "messages": [{"type": "text", "content": query}]},
            timeout=120,
        )
        assert r.status_code == 200
        data = r.json()
        # We expect the router to choose a product-finding tool and produce base keys
        bases = data.get("base_random_keys")
        assert bases is not None and len(bases) > 0, f"No base_random_keys for query: {query[:120]}"

        # Assert at least one overlap with expected bases from the recorded response
        expected = _parse_expected_bases(resp_hint)
        overlap = set(expected) & set(bases)
        assert overlap, f"No overlap with expected bases. expected={expected} got={bases} for query: {query[:120]}"
