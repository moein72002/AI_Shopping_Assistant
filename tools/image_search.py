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

# --- Configuration ---
MODEL_ID = "openai/clip-vit-large-patch14"
# Updated directory paths to include all 7 parts
IMAGE_DIRS = [f"/kaggle/input/images-part{i}/images_part{i}" for i in range(1, 8)]
TEST_IMAGE_DIR = "/kaggle/input/test-images/test_images"
# On runtime we expect Kaggle to have placed it in project root as 'product.index'
FAISS_INDEX_PATH = "product.index"
# The map now stores full image paths, not just IDs
IMAGE_PATH_MAP_PATH = "image_paths.json"
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


# if __name__ == "__main__":
#     create_index()
#     search_with_images()