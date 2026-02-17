import os
import glob
import json
import time
import traceback
import threading
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import base64
import io
from utils.utils import _append_chat_log

# --- Configuration ---
MODEL_ID = "openai/clip-vit-large-patch14"
# Updated directory paths to include all 7 parts
IMAGE_DIRS = [f"/kaggle/input/images-part{i}/images_part{i}" for i in range(1, 8)]
TEST_IMAGE_DIR = "/kaggle/input/test-images/test_images"
# Adjust batch size based on your GPU's VRAM.
BATCH_SIZE = 64

# --- Setup device (use GPU if available) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

_SEARCH_RESOURCES_LOCK = threading.Lock()
_SEARCH_RESOURCES = {
    "model": None,
    "processor": None,
    "image_embeddings": None,
    "index_ntotal": None,
    "index_dim": None,
    "image_paths_map": None,
    "faiss_index_path": None,
    "image_path_map_path": None,
}


def _log_image_step(chat_id: str, stage: str, payload: dict | None = None):
    data = {"stage": stage}
    if payload:
        data.update(payload)
    _append_chat_log(chat_id, data)


def _error_payload(exc: Exception) -> dict:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _extract_embedding_tensor(embedding_output):
    """Normalize CLIP embedding output across transformers versions."""
    if isinstance(embedding_output, torch.Tensor):
        return embedding_output
    if hasattr(embedding_output, "image_embeds") and embedding_output.image_embeds is not None:
        return embedding_output.image_embeds
    if hasattr(embedding_output, "pooler_output") and embedding_output.pooler_output is not None:
        return embedding_output.pooler_output
    raise TypeError(
        f"Unsupported embedding output type: {type(embedding_output).__name__}"
    )


def _normalize_l2_inplace(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize vectors in-place along the last axis (safe for zero vectors)."""
    if vectors.ndim == 1:
        norm = float(np.linalg.norm(vectors))
        if norm > eps:
            vectors /= norm
        return vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    vectors /= norms
    return vectors


def _cache_is_ready(faiss_index_path: str, image_path_map_path: str) -> bool:
    return (
        _SEARCH_RESOURCES.get("model") is not None
        and _SEARCH_RESOURCES.get("processor") is not None
        and _SEARCH_RESOURCES.get("image_embeddings") is not None
        and _SEARCH_RESOURCES.get("index_ntotal") is not None
        and _SEARCH_RESOURCES.get("index_dim") is not None
        and _SEARCH_RESOURCES.get("image_paths_map") is not None
        and _SEARCH_RESOURCES.get("faiss_index_path") == faiss_index_path
        and _SEARCH_RESOURCES.get("image_path_map_path") == image_path_map_path
    )


def preload_image_search_resources(chat_id: str = "startup") -> dict:
    """Load model/index/map once per process and reuse for future requests."""
    faiss_index_path = _faiss_index_path()
    image_path_map_path = _image_path_map_path()
    index_exists = os.path.exists(faiss_index_path)
    map_exists = os.path.exists(image_path_map_path)

    if not index_exists or not map_exists:
        payload = {
            "status": "error",
            "reason": "missing_index_or_map",
            "index_exists": index_exists,
            "map_exists": map_exists,
            "faiss_index_path": faiss_index_path,
            "image_path_map_path": image_path_map_path,
        }
        _log_image_step(chat_id, "image_search_preload_error", payload)
        return payload

    with _SEARCH_RESOURCES_LOCK:
        if _cache_is_ready(faiss_index_path, image_path_map_path):
            image_paths_map = _SEARCH_RESOURCES["image_paths_map"]
            payload = {
                "status": "cache_hit",
                "index_ntotal": int(_SEARCH_RESOURCES["index_ntotal"]),
                "index_dim": int(_SEARCH_RESOURCES["index_dim"]),
                "map_count": len(image_paths_map) if isinstance(image_paths_map, list) else None,
            }
            _log_image_step(chat_id, "image_search_preload_cache_hit", payload)
            return payload

        try:
            t0 = time.perf_counter()
            _log_image_step(chat_id, "image_search_preload_load_start", {"device": "cpu"})
            model, processor = load_model("cpu")
            model_ms = int((time.perf_counter() - t0) * 1000)

            t1 = time.perf_counter()
            index = faiss.read_index(faiss_index_path)
            index_ms = int((time.perf_counter() - t1) * 1000)
            index_ntotal = int(index.ntotal)
            index_dim = int(index.d)

            t2 = time.perf_counter()
            with open(image_path_map_path, "r") as f:
                image_paths_map = json.load(f)
            map_ms = int((time.perf_counter() - t2) * 1000)

            t3 = time.perf_counter()
            image_embeddings = np.empty((index_ntotal, index_dim), dtype="float32")
            index.reconstruct_n(0, index_ntotal, image_embeddings)
            _normalize_l2_inplace(image_embeddings)
            embeddings_ms = int((time.perf_counter() - t3) * 1000)

            if not isinstance(image_paths_map, list):
                raise TypeError(f"image path map must be a list, got {type(image_paths_map).__name__}")

            _SEARCH_RESOURCES.update(
                {
                    "model": model,
                    "processor": processor,
                    "image_embeddings": image_embeddings,
                    "index_ntotal": index_ntotal,
                    "index_dim": index_dim,
                    "image_paths_map": image_paths_map,
                    "faiss_index_path": faiss_index_path,
                    "image_path_map_path": image_path_map_path,
                }
            )

            payload = {
                "status": "loaded",
                "model_duration_ms": model_ms,
                "index_duration_ms": index_ms,
                "map_duration_ms": map_ms,
                "embeddings_duration_ms": embeddings_ms,
                "index_ntotal": index_ntotal,
                "index_dim": index_dim,
                "map_count": len(image_paths_map),
            }
            _log_image_step(chat_id, "image_search_preload_load_done", payload)
            return payload
        except Exception as e:
            _SEARCH_RESOURCES.update(
                {
                    "model": None,
                    "processor": None,
                    "image_embeddings": None,
                    "index_ntotal": None,
                    "index_dim": None,
                    "image_paths_map": None,
                    "faiss_index_path": None,
                    "image_path_map_path": None,
                }
            )
            payload = {"status": "error", **_error_payload(e)}
            _log_image_step(chat_id, "image_search_preload_load_error", payload)
            return payload


def _datasets_dir() -> str:
    """Resolve datasets directory at runtime to stay consistent with app startup."""
    env_dir = os.environ.get("DATASETS_DIR")
    if env_dir:
        try:
            os.makedirs(env_dir, exist_ok=True)
        except Exception:
            pass
        return env_dir

    default_dir = "/datasets"
    if os.path.isdir(default_dir):
        return default_dir

    # Fallback to project-local datasets directory for local runs.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_dir = os.path.join(project_root, "datasets")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def _faiss_index_path() -> str:
    return os.path.join(_datasets_dir(), "products.index")


def _image_path_map_path() -> str:
    # The map stores full image paths, not just IDs
    return os.path.join(_datasets_dir(), "image_paths.json")


def load_model(device):
    """Loads the CLIP model and processor onto a specified device."""
    use_fast = os.environ.get("CLIP_USE_FAST", "").strip().lower() in {"1", "true", "yes", "on"}
    print(f"Loading CLIP model onto {device} (use_fast_processor={use_fast})...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    # Default to the slow processor to avoid OpenMP/runtime conflicts seen on some macOS setups.
    processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=use_fast)
    print("Model loaded successfully.")
    return model, processor

def create_index():
    """Creates a FAISS index for all images from all specified directories."""
    faiss_index_path = _faiss_index_path()
    image_path_map_path = _image_path_map_path()

    if os.path.exists(faiss_index_path):
        print("FAISS index already exists. Skipping creation.")
        return

    # For index creation, use the globally defined best available device (GPU or CPU)
    model, processor = load_model(DEVICE)
    
    # Collect image paths from all directories
    image_paths = []
    print("Gathering image paths from all directories...")
    for directory in IMAGE_DIRS:
        image_paths.extend(glob.glob(os.path.join(directory, "*.jpg")))
    image_paths = sorted(image_paths)
    
    # --- FIX: Explicitly ignore the known corrupted files ---
    print(f"Found {len(image_paths)} total image paths.")
    corrupted_filenames = {"qltwil", "blnepk", "ebthjt", "iystky", "ytqvdq"}
    
    original_count = len(image_paths)
    image_paths = [
        p for p in image_paths 
        if os.path.splitext(os.path.basename(p))[0] not in corrupted_filenames
    ]
    print(f"Ignoring {original_count - len(image_paths)} known corrupted files.")
    # --- END FIX ---
    
    if not image_paths:
        print(f"Error: No valid .jpg images found after filtering.")
        return

    all_embeddings = []
    # This list will store the full path for each corresponding embedding
    all_image_paths_for_map = []

    print(f"Generating embeddings for {len(image_paths)} images in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Processing Batches"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        
        image_batch = []
        valid_batch_paths = [] 
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                image_batch.append(img)
                valid_batch_paths.append(path)
            except Exception as e:
                # This safety net catches any other unforeseen corrupted images
                print(f"\nWarning: Skipping corrupted or unreadable image '{path}': {e}")
        
        if not image_batch:
            continue

        with torch.no_grad():
            inputs = processor(images=image_batch, return_tensors="pt", padding=True).to(DEVICE)
            batch_embeddings = model.get_image_features(**inputs)

        all_embeddings.extend(batch_embeddings.cpu().numpy())
        all_image_paths_for_map.extend(valid_batch_paths)

    if not all_embeddings:
        print("Could not generate any embeddings. Exiting.")
        return

    embeddings_np = np.array(all_embeddings).astype("float32")
    _normalize_l2_inplace(embeddings_np)
    
    embedding_dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_np)
    
    print(f"\nIndex created with {index.ntotal} vectors.")
    
    faiss.write_index(index, faiss_index_path)
    with open(image_path_map_path, 'w') as f:
        json.dump(all_image_paths_for_map, f)
        
    print(f"Index saved to {faiss_index_path}")
    print(f"Image path map saved to {image_path_map_path}")


def search_with_images(k=5):
    """Searches and plots the most similar images for each image in the test folder."""
    import matplotlib.pyplot as plt

    faiss_index_path = _faiss_index_path()
    image_path_map_path = _image_path_map_path()

    if not os.path.exists(faiss_index_path):
        print("FAISS index not found. Please run create_index() first.")
        return

    print("\n--- Starting Image Search on CPU ---")
    SEARCH_DEVICE = "cpu"
    model, processor = load_model(SEARCH_DEVICE)
    
    index = faiss.read_index(faiss_index_path)
    image_embeddings = np.empty((int(index.ntotal), int(index.d)), dtype="float32")
    index.reconstruct_n(0, int(index.ntotal), image_embeddings)
    _normalize_l2_inplace(image_embeddings)
    with open(image_path_map_path, 'r') as f:
        image_paths_map = json.load(f)
        
    test_image_paths = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpeg"))
    
    if not test_image_paths:
        print(f"Error: No .jpeg images found in the '{TEST_IMAGE_DIR}' directory.")
        return

    for test_path in test_image_paths:
        start_time = time.time()
        
        print(f"\nSearching for similar products to: {os.path.basename(test_path)}")
        
        try:
            query_image = Image.open(test_path).convert("RGB")
        except Exception as e:
            print(f"Could not open test image {test_path}: {e}")
            continue
            
        with torch.no_grad():
            inputs = processor(images=query_image, return_tensors="pt").to(SEARCH_DEVICE)
            raw_embedding = model.get_image_features(**inputs)
            query_embedding = _extract_embedding_tensor(raw_embedding)
        
        query_embedding_np = query_embedding.cpu().numpy().astype("float32")
        _normalize_l2_inplace(query_embedding_np)

        if image_embeddings.shape[0] == 0:
            print("Error: Loaded index has no vectors.")
            continue

        scores = np.matmul(image_embeddings, query_embedding_np[0])
        top_k = min(max(1, int(k)), scores.shape[0])
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        # --- Plotting Results ---
        fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 5))
        
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        for i in range(top_k):
            result_index = int(top_indices[i])
            result_image_path = image_paths_map[result_index]
            similarity = float(scores[result_index])
            product_id = os.path.splitext(os.path.basename(result_image_path))[0]

            try:
                result_image = Image.open(result_image_path)
                axes[i+1].imshow(result_image)
                axes[i+1].set_title(f"Result {i+1}\nID: {product_id}\nSim: {similarity:.2f}")
            except FileNotFoundError:
                axes[i+1].set_title(f"Result {i+1}\nID: {product_id}\n(Image not found)")

            axes[i+1].axis('off')
        
        plt.tight_layout()
        # plt.show()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for this search: {elapsed_time:.2f} seconds")

# --- NEW METHOD ---
def find_most_similar_product(chat_id, base64_image_string):
    """
    Finds the most similar product ID for a given base64 encoded image.

    Note: For production use, it's more efficient to load the model, 
    processor, index, and map only once outside this function.

    Args:
        base64_image_string (str): An image string in "data:image/jpeg;base64,<data>" format.

    Returns:
        str: The product ID of the most similar image, or an error message.
    """

    started_at = time.perf_counter()
    raw_input = base64_image_string or ""
    faiss_index_path = _faiss_index_path()
    image_path_map_path = _image_path_map_path()
    datasets_dir = _datasets_dir()

    _log_image_step(
        chat_id,
        "image_search_start",
        {
            "datasets_dir": datasets_dir,
            "faiss_index_path": faiss_index_path,
            "image_path_map_path": image_path_map_path,
            "input_length": len(raw_input),
            "input_prefix": raw_input[:40],
            "has_data_uri_prefix": raw_input.startswith("data:image"),
            "has_base64_comma": "," in raw_input,
        },
    )

    index_exists = os.path.exists(faiss_index_path)
    map_exists = os.path.exists(image_path_map_path)
    _log_image_step(
        chat_id,
        "image_search_file_check",
        {
            "index_exists": index_exists,
            "map_exists": map_exists,
            "index_size_bytes": os.path.getsize(faiss_index_path) if index_exists else None,
            "map_size_bytes": os.path.getsize(image_path_map_path) if map_exists else None,
        },
    )

    if not index_exists or not map_exists:
        _log_image_step(
            chat_id,
            "image_search_error",
            {
                "reason": "missing_index_or_map",
                "index_exists": index_exists,
                "map_exists": map_exists,
            },
        )
        return "Error: FAISS index or image map not found. Please run create_index() first."

    preload_status = preload_image_search_resources(chat_id)
    _log_image_step(chat_id, "image_search_resources_status", preload_status)
    if preload_status.get("status") == "error":
        return f"Error loading models or index files: {preload_status.get('error') or preload_status.get('reason')}"

    with _SEARCH_RESOURCES_LOCK:
        model = _SEARCH_RESOURCES.get("model")
        processor = _SEARCH_RESOURCES.get("processor")
        image_embeddings = _SEARCH_RESOURCES.get("image_embeddings")
        index_dim = _SEARCH_RESOURCES.get("index_dim")
        image_paths_map = _SEARCH_RESOURCES.get("image_paths_map")

    if (
        model is None
        or processor is None
        or image_embeddings is None
        or index_dim is None
        or image_paths_map is None
    ):
        _log_image_step(chat_id, "image_search_error", {"reason": "search_resources_not_ready"})
        return "Error loading models or index files: search resources were not ready."

    # 3. Decode the base64 input string to get the image.
    try:
        if "," in raw_input:
            data_uri_header, encoded_data = raw_input.split(",", 1)
        else:
            data_uri_header = ""
            encoded_data = raw_input

        _log_image_step(
            chat_id,
            "image_search_decode_start",
            {
                "data_uri_header": data_uri_header[:80],
                "encoded_length": len(encoded_data),
            },
        )
        image_bytes = base64.b64decode(encoded_data)
        query_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        _log_image_step(
            chat_id,
            "image_search_decode_done",
            {
                "decoded_bytes": len(image_bytes),
                "image_size": list(query_image.size),
                "image_mode": query_image.mode,
            },
        )
    except (ValueError, base64.binascii.Error, IOError) as e:
        _log_image_step(chat_id, "image_search_decode_error", _error_payload(e))
        return f"Error: Invalid base64 string or corrupted image data. Details: {e}"
    except Exception as e:
        _log_image_step(chat_id, "image_search_decode_unexpected_error", _error_payload(e))
        return f"Error: Unexpected image decode failure. Details: {e}"

    # 4. Generate the embedding for the query image.
    try:
        t3 = time.perf_counter()
        with torch.no_grad():
            SEARCH_DEVICE = "cpu"
            inputs = processor(images=query_image, return_tensors="pt").to(SEARCH_DEVICE)
            raw_embedding = model.get_image_features(**inputs)
            query_embedding = _extract_embedding_tensor(raw_embedding)
        _log_image_step(
            chat_id,
            "image_search_embedding_done",
            {
                "duration_ms": int((time.perf_counter() - t3) * 1000),
                "embedding_shape": list(query_embedding.shape),
                "embedding_dtype": str(query_embedding.dtype),
            },
        )

        # 5. Prepare the embedding and search the cached normalized embeddings.
        query_embedding_np = query_embedding.cpu().numpy().astype("float32")
        _normalize_l2_inplace(query_embedding_np)
        embedding_dim = int(query_embedding_np.shape[1]) if query_embedding_np.ndim == 2 else None
        index_dim = int(index_dim)
        if embedding_dim != index_dim:
            _log_image_step(
                chat_id,
                "image_search_dimension_mismatch",
                {
                    "embedding_dim": embedding_dim,
                    "index_dim": index_dim,
                },
            )
            return (
                "Error: Embedding dimension mismatch. "
                f"embedding_dim={embedding_dim}, index_dim={index_dim}"
            )

        if not isinstance(image_embeddings, np.ndarray) or image_embeddings.ndim != 2:
            _log_image_step(
                chat_id,
                "image_search_error",
                {"reason": "invalid_cached_embeddings", "type": str(type(image_embeddings))},
            )
            return "Error: Cached image embeddings were invalid."

        if image_embeddings.shape[1] != index_dim:
            _log_image_step(
                chat_id,
                "image_search_error",
                {
                    "reason": "cached_embeddings_dim_mismatch",
                    "embeddings_dim": int(image_embeddings.shape[1]),
                    "index_dim": index_dim,
                },
            )
            return "Error: Cached embeddings dimension mismatch."

        if image_embeddings.shape[0] == 0:
            _log_image_step(chat_id, "image_search_error", {"reason": "empty_embeddings"})
            return "Error: Search index has no vectors."

        t4 = time.perf_counter()
        similarity_scores = np.matmul(image_embeddings, query_embedding_np[0])
        result_index = int(np.argmax(similarity_scores))
        top_distance = float(similarity_scores[result_index])
        _log_image_step(
            chat_id,
            "image_search_similarity_done",
            {
                "duration_ms": int((time.perf_counter() - t4) * 1000),
                "result_index": result_index,
                "top_distance": top_distance,
            },
        )
    except Exception as e:
        _log_image_step(chat_id, "image_search_embedding_or_search_error", _error_payload(e))
        return f"Error: Embedding/search failed. Details: {e}"

    # 6. Retrieve the corresponding product ID and return it.
    try:
        if not isinstance(image_paths_map, list) or result_index < 0 or result_index >= len(image_paths_map):
            _log_image_step(
                chat_id,
                "image_search_error",
                {
                    "reason": "result_index_out_of_range",
                    "result_index": result_index,
                    "map_count": len(image_paths_map) if isinstance(image_paths_map, list) else None,
                },
            )
            return "Error: Result index was out of range for image map."

        result_image_path = image_paths_map[result_index]
        product_id = os.path.splitext(os.path.basename(result_image_path))[0]
        _log_image_step(
            chat_id,
            "image_search_success",
            {
                "result_index": result_index,
                "result_image_path": result_image_path,
                "result_image_exists": os.path.exists(result_image_path),
                "top_distance": top_distance,
                "product_id": product_id,
                "total_duration_ms": int((time.perf_counter() - started_at) * 1000),
            },
        )
        return product_id
    except Exception as e:
        _log_image_step(chat_id, "image_search_result_parse_error", _error_payload(e))
        return f"Error: Failed to parse search results. Details: {e}"


def _load_warmup_image_sample() -> str | None:
    """Load one sample image from server_tests/scenario7.json for startup warm-up."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scenario_path = os.path.join(project_root, "server_tests", "scenario7.json")
    if not os.path.exists(scenario_path):
        return None

    try:
        with open(scenario_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return None

    if not isinstance(items, list):
        return None

    for item in items:
        if not isinstance(item, dict):
            continue
        sample = item.get("Last Query") or item.get("last_query")
        if isinstance(sample, str) and sample:
            return sample

    return None


def warm_up_image_search():
    sample = _load_warmup_image_sample()
    if not sample:
        return "warm_up_skipped:no_scenario7_sample"

    return find_most_similar_product("test_chat", sample)

#     create_index()
#     search_with_images()
#
#     # --- Example Usage for the new function ---
#     # This is a placeholder for a real base64 string.
#     # In a real application, you would get this from a web request or file.
