### Torob AI Shopping Assistant

An end-to-end FastAPI service for the Torob hackathon. It implements a tool-based agent that understands Persian user requests, routes them to specialized tools (BM25 product search, SQL/database queries, image search with CLIP+FAISS, comparison and conversational tools), and returns structured results via a single /chat endpoint.

---

### Key Features
- **Monolithic FastAPI app** with a single `/chat` endpoint and JSON I/O.
- **Scenario 0 health checks**: ping, echo base/member keys.
- **Scenario 1 (Direct Product Lookup)**: BM25-based candidate retrieval (+ optional LLM disambiguation) to return `base_random_keys`.
- **Scenario 2 (Product Feature Q&A)**: Database Q&A via `tools/answer_question_about_a_product_tool.py` and helpers in `tools/database_tool.py`.
- **Scenario 3 (Seller/Price Q&A)**: Seller-focused answers (e.g., min price) via `tools/answer_question_about_a_product_sellers_tool.py` and SQL helpers.
- **Scenario 4 (Guided Discovery)**: Multi-turn Persian clarifying questions using `tools/ask_clarifying_questions_tool.py` with a lightweight in-memory session manager and final `member_random_keys` selection via `tools/member_picker_tool.py`.
- **Scenario 5 (Product Comparison)**: Extract two products, compare, and return the better base with reason via `tools/compare_two_products_tool.py`.
- **Scenarios 6 & 7 (Vision)**: CLIP+FAISS image search (`tools/image_search.py`) to resolve the most similar base product from a base64 image, then:
  - Scenario 7: return the matched `base_random_keys`.
  - Scenario 6: generate a short Persian noun describing the main object using the product name and a small LLM call.
- **BM25 lexical search** over `base_products` names (`tools/bm25_tool.py`).
- **Startup self-heal**: optional Kaggle download for `torob.db` and product image index; rebuild DB from `db_data/*.parquet` if needed.
- **Request logging** to `requests.log` and a lightweight admin HTML page at `/admin` for recent activity.

Note: A separate explicit translation layer is not required at this stage because product names and prompts are already Persian-aware. You can add a translation layer later if desired.

---

### Architecture Overview
- `main.py`: FastAPI app, `/chat` endpoint, routing and orchestration, sliding-window chat memory, scenario flows, and image handling.
- `tools/`:
  - `bm25_tool.py`: BM25 product-name retrieval.
  - `database_tool.py`: Helpers for feature extraction, schema, and minimal Text-to-SQL fallback.
  - `product_name_extractor_tool.py`, `product_id_lookup_tool.py`: Product parsing and resolution.
  - `answer_question_about_a_product_tool.py`, `answer_question_about_a_product_sellers_tool.py`: Scenario 2/3 logic.
  - `ask_clarifying_questions_tool.py`, `shop_preference_extractor_tool.py`, `member_picker_tool.py`: Scenario 4 multi-turn discovery and seller selection.
  - `comparison_extractor_tool.py`, `compare_two_products_tool.py`: Scenario 5 product extraction and comparison.
  - `image_search.py`: CLIP-based image embeddings + FAISS search over product images (Scenarios 6/7).
- `utils/utils.py`: Dataset path helpers, chat logging helpers, and product name lookup.
- `download_data_scripts/`: Kaggle download helpers for `torob.db` and image index files (`products.index`, `image_paths.json`).
- `tests/`: Pytest suite with scenario 0 and example checks.
- `start.sh`: Production entrypoint (gunicorn + uvicorn worker) and Streamlit utilities.

Data paths:
- SQLite DB path resolved to `/datasets/torob.db` (or `DATASETS_DIR/torob.db`).
- Image search assets at `/datasets/products.index` and `/datasets/image_paths.json`.

---

### API
POST `/chat`

Request
```json
{
  "chat_id": "string",
  "messages": [
    { "type": "text" | "image", "content": "string" }
  ]
}
```

Response
```json
{
  "message": "string | null",
  "base_random_keys": ["string"] | null,
  "member_random_keys": ["string"] | null
}
```

Important
- When `base_random_keys` or `member_random_keys` is returned, the judge concludes the session. Use them only for final answers.
- Images must be base64 strings prefixed like `data:image/jpeg;base64,<data>`.

Scenario examples (abbrev.)
- Scenario 0 ping request: returns `{ "message": "pong" }`.
- Scenario 1: precise base lookup → returns exactly one `base_random_keys`.
- Scenario 3: price questions → returns a numeric `message` (parsable int/float).
- Scenario 4: up to 5 clarification turns, then returns exactly one `member_random_keys`.

---

### Setup
Prerequisites
- Python 3.11+
- (Optional) Docker for containerized runs

Install
```bash
uv pip sync requirements.txt
```

Environment
Create a `.env` with at least:
```
OPENAI_API_KEY="sk-..."               # required for LLM calls via Torob proxy
TOROB_PROXY_URL="https://<proxy>/v1"   # provided by organizers

# Optional for dataset auto-download on startup
KAGGLE_USERNAME="..."
KAGGLE_KEY="..."

# Optional tuning
# BM25_LIMIT=100000
# DATASETS_DIR=/datasets
# FORCE_KAGGLE_DOWNLOAD=1
# LLM_ROUTER_MODEL=gpt-4o-mini
# LLM_CLARIFY_MODEL=gpt-4o-mini
# DISABLE_ROUTER_LLM=1             # bypass router and return BM25 results directly
# DISABLE_LLM_FOR_TESTS=1          # deterministic clarifying questions
```

Data
- Default DB path: `/datasets/torob.db`. On startup, the app will:
  1) Try Kaggle download if credentials are set; else
  2) Verify/repair DB; else
  3) Rebuild from `db_data/*.parquet` via `scripts/load_data.py`.
- For image search, provide `/datasets/products.index` and `/datasets/image_paths.json` (Kaggle script can fetch them).

---

### Run (Development)
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
API: `http://localhost:8080`

Admin logs page: `GET /admin` (simple HTML over the same port). Requests are also logged to `requests.log`.

---

### Run (Docker)
Build
```bash
docker build -t torob-assistant .
```

Run
```bash
docker run --rm -p 8080:8080 --env-file .env torob-assistant
```

The container starts:
- API (gunicorn/uvicorn) on port 8080
- Streamlit UIs on 8501/8502 bound to 127.0.0.1 inside the container (not exposed by default)

---

### Testing
Pytest (spins up a local server on :8090 for tests that need it):
```bash
pytest -q
```

Sample scenario runner (CLI):
```bash
python run_test_scripts/run_scenarios_sample.py --base-url http://localhost:8080 --file server_tests/scenario1.json --limit 10
```

Troubleshooting
- 500 errors during scenario 2/3 may indicate missing OpenAI proxy or exhausted credit. Ensure `OPENAI_API_KEY` and `TOROB_PROXY_URL` are set.
- If `/datasets/torob.db` fails integrity, the app attempts repair/rebuild at startup.
- Set `DISABLE_ROUTER_LLM=1` to bypass LLM routing and quickly test BM25 search paths.

---

### Deployment
- Use the provided `Dockerfile` and run on your preferred platform (e.g., HamRavesh/Darkube, Cloud Run, etc.).
- Expose the API publicly and submit the base URL to the judge. Keep the `/chat` response under 30 seconds; the app enforces a 30s timeout for `/chat` requests.
- Consider enabling basic auth at the platform level to restrict public traffic if needed.

Cost controls
- Router and classification models default to small variants (e.g., `gpt-4o-mini`). Keep usage within ~$0.10 for local testing.
- Use the BM25 bypass (`DISABLE_ROUTER_LLM=1`) when debugging retrieval logic.

---

### Project Layout (selected)
- `main.py`: API and orchestration, scenario flows, image handling.
- `tools/`: retrieval, database, comparison, conversation, and image tools.
- `utils/utils.py`: shared helpers.
- `download_data_scripts/`: Kaggle fetchers.
- `tests/`: pytest suite (scenario 0, examples for 1–3).
- `start.sh`: production entrypoint.
- `Dockerfile`: container build for API + UIs.

---

### Acknowledgements
- Torob Turbo organizers for the dataset and proxy setup.
- Open-source libraries: FastAPI, Uvicorn/Gunicorn, bm25s, FAISS, Transformers/CLIP, Streamlit, PyTest.


