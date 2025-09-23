# download_data_from_kaggle.py

import os
from dotenv import load_dotenv

# --- CRITICAL STEP ---
# Load environment variables from the .env file FIRST.
load_dotenv()

# Now, import Kaggle. It will find the environment variables.
import kaggle

# --------------------

print("Authenticating with Kaggle...")
# The KaggleApi class will now automatically use the loaded environment variables.
# An explicit authenticate() call is good practice to confirm it works.
api = kaggle.KaggleApi()
api.authenticate()
print("Authentication successful.")

# --- Configuration ---
dataset_slug = 'moeinmadadi/ai-shopping-assistant-db'
download_path = './shopping_dataset'

# Create the target directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# --- Download and Unzip ---
print(f"Downloading dataset '{dataset_slug}' to '{download_path}'...")
api.dataset_download_files(
    dataset=dataset_slug,
    path=download_path,
    unzip=True
)

print(f"Dataset downloaded and unzipped successfully in '{download_path}'.")