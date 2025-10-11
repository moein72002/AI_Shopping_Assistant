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
download_path = os.environ.get('DATASETS_DIR') or '/datasets/'
os.makedirs(download_path, exist_ok=True)

# --- Download and Unzip ---
print(f"Downloading dataset '{dataset_slug}' to '{download_path}'...")
api.dataset_download_files(
    dataset=dataset_slug,
    path=download_path,
    unzip=True
)

print(f"Dataset downloaded and unzipped successfully in '{download_path}'.")