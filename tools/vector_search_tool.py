import json
import os
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_index.faiss")
NAMES_INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_index_names.faiss")
DB_PATH = os.path.join(PROJECT_ROOT, "torob.db")

# The global mapping file is now gone, so these paths point to the specific mappings
FULL_MAPPING_PATH = os.path.join(PROJECT_ROOT, "vector_index.faiss.json")
NAMES_MAPPING_PATH = os.path.join(PROJECT_ROOT, "vector_index_names.faiss.json")


def get_embedding(text, model="text-embedding-3-small"):
   from openai import OpenAI
   client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("TOROB_PROXY_URL"),
   )
   text = text.replace("\n", " ")
   return client.embeddings.create(input=[text], model=model).data[0].embedding


def _fallback_text_search(query: str, k: int) -> list[str]:
    tokens = [t for t in query.lower().split() if len(t) > 2]
    if not tokens:
        tokens = [query.lower()]
    like = " AND ".join(["(LOWER(persian_name) LIKE ? OR LOWER(english_name) LIKE ? OR LOWER(extra_features) LIKE ?)"] * len(tokens))
    params = []
    for t in tokens:
        pat = f"%{t}%"
        params.extend([pat, pat, pat])
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            f"SELECT random_key FROM base_products WHERE {like} LIMIT ?",
            (*params, k),
        )
        rows = cur.fetchall()
        if not rows:
            try:
                cur.execute("SELECT random_key FROM base_products LIMIT ?", (k,))
                rows = cur.fetchall()
            except Exception:
                rows = []
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        # Fallback to the fallback: read from the full mapping file if it exists
        if os.path.exists(FULL_MAPPING_PATH):
            try:
                with open(FULL_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    keys = json.load(f)
                return keys[:k]
            except Exception:
                return []
        return []


def _direct_keyword_search(query: str, k: int) -> list[str] | None:
    """Attempts a direct, case-insensitive keyword search first for precision."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # This is a simple but effective heuristic for specific product names
        cur.execute(
            "SELECT random_key FROM base_products WHERE LOWER(persian_name) = ? OR LOWER(english_name) = ? LIMIT ?",
            (query.lower(), query.lower(), k),
        )
        rows = cur.fetchall()
        conn.close()
        if rows:
            return [r[0] for r in rows]
        return None
    except Exception:
        return None


def _faiss_search(index_path: str, query_vector, top_k: int):
    import faiss
    index = faiss.read_index(index_path)
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]


def vector_search(query: str, k: int = 5):
    """
    Dual-retrieval with dedicated mappings and reciprocal rank fusion.
    """
    try:
        # Step 1: Attempt a direct keyword search for precision
        direct_match = _direct_keyword_search(query, k)
        if direct_match:
            return direct_match

        # Step 2: If no direct match, proceed with semantic search
        import numpy as np

        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')

        candidates: dict[str, float] = {}

        # Full-text index search
        if os.path.exists(INDEX_PATH) and os.path.exists(FULL_MAPPING_PATH):
            try:
                with open(FULL_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    full_keys = json.load(f)
                
                _, idxs = _faiss_search(INDEX_PATH, query_vector, max(k, 20))
                for rank, i in enumerate(idxs):
                    if 0 <= i < len(full_keys):
                        rk = full_keys[i]
                        candidates[rk] = candidates.get(rk, 0.0) + 1.0 / (rank + 60) # RRF with k=60
            except Exception as e:
                print(f"Error searching full index: {e}")

        # Names-only index search
        if os.path.exists(NAMES_INDEX_PATH) and os.path.exists(NAMES_MAPPING_PATH):
            try:
                with open(NAMES_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    names_keys = json.load(f)

                _, idxs_n = _faiss_search(NAMES_INDEX_PATH, query_vector, max(k, 20))
                for rank, i in enumerate(idxs_n):
                    if 0 <= i < len(names_keys):
                        rk = names_keys[i]
                        candidates[rk] = candidates.get(rk, 0.0) + 1.0 / (rank + 60) # RRF with k=60
            except Exception as e:
                print(f"Error searching names index: {e}")

        if not candidates:
            return _fallback_text_search(query, k)

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [rk for rk, _ in ranked[:k]]
    except Exception as e:
        print(f"Error in vector_search: {e}")
        return _fallback_text_search(query, k)


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    # Example usage:
    # Make sure OPENAI_API_KEY and TOROB_PROXY_URL are in your .env file
    print("Searching for 'a fancy red mug'...")
    print(vector_search("a fancy red mug", k=5))
    print("\nSearching for 'لپ تاپ گیمینگ ایسوس'...")
    print(vector_search("لپ تاپ گیمینگ ایسوس", k=5))
