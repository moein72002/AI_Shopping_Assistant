import os
import glob
import json
import time
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import base64
import io
# from utils.utils import _append_chat_log

# --- Configuration ---
MODEL_ID = "openai/clip-vit-large-patch14"
# Updated directory paths to include all 7 parts
IMAGE_DIRS = [f"/kaggle/input/images-part{i}/images_part{i}" for i in range(1, 8)]
TEST_IMAGE_DIR = "/kaggle/input/test-images/test_images"
FAISS_INDEX_PATH = "/datasets/products.index"
# The map now stores full image paths, not just IDs
IMAGE_PATH_MAP_PATH = "/datasets/image_paths.json"
# Adjust batch size based on your GPU's VRAM.
BATCH_SIZE = 64

# --- Setup device (use GPU if available) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def load_model(device):
    """Loads the CLIP model and processor onto a specified device."""
    print(f"Loading CLIP model onto {device}...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print("Model loaded successfully.")
    return model, processor

def create_index():
    """Creates a FAISS index for all images from all specified directories."""
    if os.path.exists(FAISS_INDEX_PATH):
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

    embeddings_np = np.array(all_embeddings).astype('float32')
    faiss.normalize_L2(embeddings_np)
    
    embedding_dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_np)
    
    print(f"\nIndex created with {index.ntotal} vectors.")
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(IMAGE_PATH_MAP_PATH, 'w') as f:
        json.dump(all_image_paths_for_map, f)
        
    print(f"Index saved to {FAISS_INDEX_PATH}")
    print(f"Image path map saved to {IMAGE_PATH_MAP_PATH}")


def search_with_images(k=5):
    """Searches and plots the most similar images for each image in the test folder."""
    if not os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index not found. Please run create_index() first.")
        return

    print("\n--- Starting Image Search on CPU ---")
    SEARCH_DEVICE = "cpu"
    model, processor = load_model(SEARCH_DEVICE)
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(IMAGE_PATH_MAP_PATH, 'r') as f:
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
            query_embedding = model.get_image_features(**inputs)
        
        query_embedding_np = query_embedding.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_embedding_np)
        
        distances, indices = index.search(query_embedding_np, k)
        
        # --- Plotting Results ---
        fig, axes = plt.subplots(1, k + 1, figsize=(20, 5))
        
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        for i in range(k):
            result_index = indices[0][i]
            result_image_path = image_paths_map[result_index]
            similarity = distances[0][i]
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

    # _append_chat_log(request.chat_id, {"stage": "image_search_start"})
    # 1. Check if the necessary index and map files exist.
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(IMAGE_PATH_MAP_PATH):
        return "Error: FAISS index or image map not found. Please run create_index() first."

    # 2. Load the search tools. Using CPU is often sufficient and safer for single inference.
    try:
        SEARCH_DEVICE = "cpu"
        model, processor = load_model(SEARCH_DEVICE)
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(IMAGE_PATH_MAP_PATH, 'r') as f:
            image_paths_map = json.load(f)
    except Exception as e:
        return f"Error loading models or index files: {e}"

    # 3. Decode the base64 input string to get the image.
    try:
        # Split the header ("data:image/jpeg;base64,") from the actual base64 data.
        _, encoded_data = base64_image_string.split(",", 1)
        image_bytes = base64.b64decode(encoded_data)
        # Create an in-memory binary stream from the bytes and open it as an image.
        query_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (ValueError, base64.binascii.Error, IOError) as e:
        return f"Error: Invalid base64 string or corrupted image data. Details: {e}"

    # 4. Generate the embedding for the query image.
    with torch.no_grad():
        inputs = processor(images=query_image, return_tensors="pt").to(SEARCH_DEVICE)
        query_embedding = model.get_image_features(**inputs)

    # 5. Prepare the embedding and search the FAISS index.
    query_embedding_np = query_embedding.cpu().numpy().astype('float32')
    faiss.normalize_L2(query_embedding_np)
    
    # Search for the single most similar vector (k=1).
    distances, indices = index.search(query_embedding_np, k=1)

    # 6. Retrieve the corresponding product ID and return it.
    if indices.size == 0:
        return "Error: Search failed to return any results."

    # Get the index of the top match.
    result_index = indices[0][0]
    # Look up the full image path using the index.
    result_image_path = image_paths_map[result_index]
    # Extract the filename without the extension to get the product ID.
    product_id = os.path.splitext(os.path.basename(result_image_path))[0]

    return product_id


def warm_up_image_search():

    return find_most_similar_product("test_chat", "data:image/jpeg;base64,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")
#     create_index()
#     search_with_images()
#
#     # --- Example Usage for the new function ---
#     # This is a placeholder for a real base64 string.
#     # In a real application, you would get this from a web request or file.
