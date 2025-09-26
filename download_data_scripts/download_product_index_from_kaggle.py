import os
from dotenv import load_dotenv

# Attempt to load Kaggle credentials from Docker secret first, then local .env
secret_path = "/run/secrets/kaggle_env"
if os.path.exists(secret_path):
    print("Docker secret file found. Loading credentials...")
    load_dotenv(dotenv_path=secret_path)
else:
    print("Loading credentials from local .env file...")
    load_dotenv()

kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

if not kaggle_username or not kaggle_key:
    raise ValueError(
        "Kaggle credentials not found in environment! Ensure your .env or Docker secret is configured."
    )

import kaggle

print("Authenticating with Kaggle...")
api = kaggle.KaggleApi()
api.authenticate()
print("Authentication successful.")

# --- Configuration ---
dataset_slug = 'moeinmadadi/product-index'
download_path = './shopping_dataset'
os.makedirs(download_path, exist_ok=True)

print(f"Downloading dataset '{dataset_slug}' to '{download_path}'...")
api.dataset_download_files(
    dataset=dataset_slug,
    path=download_path,
    unzip=True,
)

print(f"Dataset downloaded and unzipped successfully in '{download_path}'.")


