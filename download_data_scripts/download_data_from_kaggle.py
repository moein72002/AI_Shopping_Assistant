import os
from dotenv import load_dotenv

# --- MODIFIED SECTION ---
# This makes the script work both locally and inside Docker with secrets.
secret_path = "/run/secrets/kaggle_env"
if os.path.exists(secret_path):
    # Running inside Docker with a build secret
    print("Docker secret file found. Loading credentials...")
    load_dotenv(dotenv_path=secret_path)
else:
    # Running locally, load from a local .env file
    print("Loading credentials from local .env file...")
    load_dotenv()

# NEW: Explicitly check if the environment variables are now set
# kaggle_username = os.getenv('KAGGLE_USERNAME')
# kaggle_key = os.getenv('KAGGLE_KEY')
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

print(f"TOROB_PROXY_URL: {os.getenv('TOROB_PROXY_URL')}")

if not kaggle_username or not kaggle_key:
    raise ValueError("Kaggle credentials not found in environment! "
                     "Ensure your .env file is correctly formatted and mounted as a secret during the build.")

print("Kaggle credentials loaded successfully.")

# Now, import Kaggle. It will find the environment variables.
import kaggle
# --- END MODIFIED SECTION ---

print("Authenticating with Kaggle...")
api = kaggle.KaggleApi()
api.authenticate()
print("Authentication successful.")

# --- Configuration ---
dataset_slug = 'moeinmadadi/ai-shopping-assistant-db'
if os.environ.get('DATASETS_DIR'):
    download_path = os.environ.get('DATASETS_DIR')
else:
    # Default to project's db_data when running locally (avoids read-only /datasets/ on macOS)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    download_path = os.path.join(project_root, 'db_data')
os.makedirs(download_path, exist_ok=True)

# --- Download and Unzip ---
print(f"Downloading dataset '{dataset_slug}' to '{download_path}'...")
try:
    api.dataset_download_files(
        dataset=dataset_slug,
        path=download_path,
        unzip=True
    )
except Exception as e:
    err_msg = str(e)
    if "403" in err_msg or "Forbidden" in err_msg or "denied" in err_msg.lower():
        print("\nKaggle returned 403 (Permission denied). Common fixes:")
        print("  1. Open the dataset in your browser and accept the rules (if any):")
        print(f"     https://www.kaggle.com/datasets/{dataset_slug}")
        print("  2. In Kaggle: Account → API → ensure your token has not expired and has dataset access.")
        print("  3. If the dataset is private, ensure your Kaggle account has been granted access.")
    if hasattr(e, "body") and e.body:
        print(f"  API response: {e.body}")
    raise
print(f"Dataset downloaded and unzipped successfully in '{download_path}'.")