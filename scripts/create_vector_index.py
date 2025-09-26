import argparse
import json
import math
import os
import sqlite3
import concurrent.futures
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# --- Constants ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join("/datasets/", "torob.db")
PROGRESS_PATH = os.path.join(PROJECT_ROOT, "index_build_progress.json")
STATUS_PATH = os.path.join(PROJECT_ROOT, "index_build_status.json")
GLOBAL_KEYS_PATH = os.path.join(PROJECT_ROOT, "base_random_keys.json")
DISABLE_TQDM = os.getenv("DISABLE_TQDM", "false").lower() in {"1", "true", "yes"}

# --- Helper Functions ---

def _price_per_m_token(model: str) -> float:
    if "text-embedding-3-small" in model: return 0.02
    if "text-embedding-3-large" in model: return 0.13
    return 0.02

def _log_jsonl(path: str, data: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _write_status(**fields):
    try:
        # Merge with existing status to preserve fields like started_at, pid
        existing = {}
        if os.path.exists(STATUS_PATH):
            try:
                with open(STATUS_PATH, "r", encoding="utf-8") as rf:
                    existing = json.load(rf) or {}
            except Exception:
                existing = {}
        existing.update(fields)
        existing.setdefault("last_update", _now_iso())
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception:
        # Best-effort status update; never crash the build
        pass

def get_progress():
    if not os.path.exists(PROGRESS_PATH): return {}
    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {}

def save_progress(progress_data: dict):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2)

def get_openai_embeddings(texts, client, model, batch_size, workers, phase, dimensions=1536, start_batch_idx=0, progress_cb=None, total_cost_so_far=0.0):
    price_m = _price_per_m_token(model)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    n_batches = len(batches)
    final_embeddings = [None] * n_batches
    total_prompt_tokens = 0
    chunk_running_cost = 0.0

    def _process_batch(batch_with_idx):
        idx, batch = batch_with_idx
        response = client.embeddings.create(input=batch, model=model, dimensions=dimensions)
        embeddings = [item.embedding for item in response.data]
        prompt_tokens = getattr(response.usage, "prompt_tokens", sum(len(t) // 4 for t in batch))
        batch_cost = (prompt_tokens / 1_000_000.0) * price_m
        return idx, embeddings, prompt_tokens, batch_cost

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_process_batch, enumerate(batches))
        
        pbar = None
        if not DISABLE_TQDM:
            pbar = tqdm(iterator, total=n_batches, desc=f"Embedding ({model})")
            iterator = pbar

        for i, (original_idx, batch_embeddings, prompt_tokens, batch_cost) in enumerate(iterator):
            final_embeddings[original_idx] = batch_embeddings
            total_prompt_tokens += prompt_tokens
            chunk_running_cost += batch_cost
            
            if pbar:
                pbar.set_postfix_str(f"batch_cost=${batch_cost:.6f}, total_cost=${total_cost_so_far + chunk_running_cost:.4f}")

            if progress_cb is not None:
                try:
                    progress_cb({
                        "event": "embedding_batch_progress",
                        "phase": phase,
                        "embedding_batch_index": start_batch_idx + i + 1,
                        "embedding_batches_total": n_batches,
                        "embedding_running_cost_usd": round(total_cost_so_far + chunk_running_cost, 8),
                    })
                except Exception:
                    pass
    
    return [item for sublist in final_embeddings if sublist for item in sublist], chunk_running_cost

# --- Main Logic ---

def create_vector_index(build_names_only: bool = False):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("TOROB_PROXY_URL"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "512"))
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))
    workers = int(os.getenv("EMBEDDING_WORKERS", "4"))
    db_chunk_size = int(os.getenv("DB_CHUNK_SIZE", "50000"))

    conn = sqlite3.connect(DB_PATH)
    total_rows = pd.read_sql_query("SELECT COUNT(*) FROM base_products", conn).iloc[0, 0]
    if not os.path.exists(GLOBAL_KEYS_PATH):
        print("Creating global base_random_keys.json mapping file...")
        all_keys = pd.read_sql_query("SELECT random_key FROM base_products", conn)['random_key'].tolist()
        with open(GLOBAL_KEYS_PATH, 'w', encoding='utf-8') as f: json.dump(all_keys, f)
    conn.close()

    phases = ["full", "names"] if build_names_only else ["full"]
    progress = get_progress()
    _write_status(status="running", phase=None, total_rows=total_rows, rows_scanned=0, percent_rows_scanned=0.0,
                  vectors_built=0, chunk_index=0, chunks_total=math.ceil(total_rows / db_chunk_size),
                  embedding_model=embedding_model, embedding_dimensions=embedding_dimensions, batch_size=batch_size, workers=workers)

    for phase in phases:
        is_names = phase == "names"
        final_index_path = os.path.join(PROJECT_ROOT, f"vector_index{'_names' if is_names else ''}.faiss")
        final_mapping_path = f"{final_index_path}.json"
        part_emb_path = f"{final_index_path}.part.npy"
        part_map_path = f"{final_index_path}.part.json"

        if os.path.exists(final_index_path):
            print(f"Index for phase '{phase}' already exists. Skipping.")
            _write_status(status="running", phase=phase, phase_skipped=True, total_rows=total_rows,
                          rows_scanned=total_rows, percent_rows_scanned=1.0, vectors_built=None,
                          chunk_index=math.ceil(total_rows / db_chunk_size),
                          chunks_total=math.ceil(total_rows / db_chunk_size))
            continue

        print(f"--- Starting phase: {phase} ---")
        phase_total_cost = progress.get(f"{phase}_total_cost", 0.0)
        offset = progress.get(f"{phase}_offset", 0)
        phase_processed_vectors = 0
        if os.path.exists(part_map_path):
            try:
                with open(part_map_path, 'r', encoding='utf-8') as f:
                    existing_map = json.load(f)
                phase_processed_vectors = len(existing_map)
            except Exception:
                phase_processed_vectors = 0

        _write_status(status="running", phase=phase, total_rows=total_rows, rows_scanned=offset,
                      percent_rows_scanned=round(min(offset / max(1, total_rows), 1.0), 6),
                      vectors_built=phase_processed_vectors,
                      chunk_index=(offset // db_chunk_size) + 1 if offset < total_rows else math.ceil(total_rows / db_chunk_size),
                      chunks_total=math.ceil(total_rows / db_chunk_size))

        if offset > 0:
            print(f"Resuming from row {offset}")

        while offset < total_rows:
            chunk_index = (offset // db_chunk_size) + 1
            chunks_total = math.ceil(total_rows / db_chunk_size)
            percent_rows = round(min(offset / max(1, total_rows), 1.0) * 100, 4)
            print(f"Processing chunk {chunk_index}/{chunks_total} | rows {offset}/{total_rows} ({percent_rows}%) | Phase cost: ${phase_total_cost:.4f}")
            conn = sqlite3.connect(DB_PATH)
            cols = "persian_name, english_name" if is_names else "persian_name, english_name, extra_features"
            df = pd.read_sql_query(f"SELECT {cols} FROM base_products LIMIT {db_chunk_size} OFFSET {offset}", conn)
            conn.close()

            if df.empty: break

            texts = (df['persian_name'].fillna('') + ' ' + df['english_name'].fillna('') + (' ' + df['extra_features'].fillna('') if not is_names else '')).tolist()
            
            original_indices = [offset + i for i in range(len(texts))]
            texts_with_indices = [(idx, text) for idx, text in zip(original_indices, texts) if text.strip()]

            if texts_with_indices:
                valid_indices, valid_texts = zip(*texts_with_indices)
                
                def _progress_cb(extra_fields: dict):
                    _write_status(status="running", phase=phase, total_rows=total_rows,
                                  rows_scanned=offset, percent_rows_scanned=round(min(offset / max(1, total_rows), 1.0), 6),
                                  vectors_built=phase_processed_vectors,
                                  chunk_index=chunk_index, chunks_total=chunks_total, **extra_fields)

                new_embeddings, chunk_cost = get_openai_embeddings(
                    list(valid_texts), client, embedding_model, batch_size, workers, phase,
                    dimensions=embedding_dimensions, start_batch_idx=0, progress_cb=_progress_cb,
                    total_cost_so_far=phase_total_cost
                )
                phase_total_cost += chunk_cost
                
                if new_embeddings:
                    new_embeddings_np = np.array(new_embeddings, dtype='float32')
                    if os.path.exists(part_emb_path):
                        existing_embeddings = np.load(part_emb_path)
                        all_embeddings = np.vstack([existing_embeddings, new_embeddings_np])
                    else:
                        all_embeddings = new_embeddings_np
                    np.save(part_emb_path, all_embeddings, allow_pickle=False)

                    if os.path.exists(part_map_path):
                        with open(part_map_path, 'r', encoding='utf-8') as f:
                            existing_map = json.load(f)
                        existing_map.extend(valid_indices)
                        with open(part_map_path, 'w', encoding='utf-8') as f:
                            json.dump(existing_map, f)
                    else:
                        with open(part_map_path, 'w', encoding='utf-8') as f:
                            json.dump(list(valid_indices), f)

                phase_processed_vectors += len(valid_indices)

            offset += len(df)
            progress[f"{phase}_offset"] = offset
            progress[f"{phase}_total_cost"] = phase_total_cost
            save_progress(progress)
            _write_status(status="running", phase=phase, total_rows=total_rows, rows_scanned=offset,
                          percent_rows_scanned=round(min(offset / max(1, total_rows), 1.0), 6),
                          vectors_built=phase_processed_vectors,
                          chunk_index=chunk_index, chunks_total=chunks_total)

        print(f"Finished processing all chunks for phase '{phase}'. Finalizing index...")
        _write_status(status="finalizing", phase=phase, total_rows=total_rows, rows_scanned=total_rows,
                      percent_rows_scanned=1.0, vectors_built=phase_processed_vectors)
        
        if not os.path.exists(part_emb_path):
            print("No embeddings were generated. Skipping final index creation.")
            _write_status(status="skipped_no_embeddings", phase=phase)
            continue

        final_embeddings_np = np.load(part_emb_path)
        
        with open(part_map_path, 'r', encoding='utf-8') as f:
            final_indices = json.load(f)
        
        # De-duplication logic for robustness against resume-crashes
        if len(final_indices) > len(set(final_indices)):
            print("Duplicates found in index map, de-duplicating...")
            unique_indices_map = {original_row_idx: i for i, original_row_idx in enumerate(final_indices)}
            unique_embedding_indices = list(unique_indices_map.values())
            
            if max(unique_embedding_indices) < final_embeddings_np.shape[0]:
                final_embeddings_np = final_embeddings_np[unique_embedding_indices]
                final_indices = list(unique_indices_map.keys())
                print(f"De-duplicated down to {len(final_indices)} vectors.")
            else:
                print(f"Warning: Mismatch between embedding count ({final_embeddings_np.shape[0]}) and de-duplication map. Skipping de-duplication.")

        if final_embeddings_np.dtype == 'object':
            print("Warning: old object-based embedding file found. Stacking to float32 array.")
            final_embeddings_np = np.vstack(final_embeddings_np).astype('float32')

        dimension = final_embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(final_embeddings_np)
        faiss.write_index(index, final_index_path)

        with open(part_map_path, 'r', encoding='utf-8') as f:
            final_indices = json.load(f)
        
        with open(GLOBAL_KEYS_PATH, 'r', encoding='utf-8') as f:
            all_keys = json.load(f)

        final_keys = [all_keys[i] for i in final_indices]
        with open(final_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(final_keys, f)

        print(f"Phase '{phase}' complete. Index with {len(final_keys)} vectors at: {final_index_path}")
        os.remove(part_emb_path)
        os.remove(part_map_path)
        progress[f"{phase}_completed"] = True
        save_progress(progress)
        _write_status(status="phase_completed", phase=phase, vectors_built=len(final_keys))

    if all(progress.get(f"{p}_completed", False) for p in phases):
        print("All phases completed. Cleaning up progress file.")
        os.remove(PROGRESS_PATH)
        _write_status(status="completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create FAISS vector indexes.")
    parser.add_argument("--names-only", action="store_true", help="Build the names-only index as well.")
    args = parser.parse_args()
    create_vector_index(build_names_only=args.names_only)
