import os
import json
import shutil
from dotenv import load_dotenv

# --- Configuration ---
# You can change these values to customize your dataset.
DATASET_TITLE = "My Log File Dataset From Server three"
DATASET_SLUG = "my-log-file-dataset-from-server-three" # Must be unique on Kaggle, uses hyphens
FOLDER_NAME = "temp_dataset_folder"
FILE_NAME = "requests.log" # The name of the file you want to upload
IS_PUBLIC = True # Set to False for a private dataset

def create_dataset():
    """
    Handles the creation of a Kaggle dataset by preparing files,
    metadata, and calling the Kaggle API.
    """
    # 1. Load environment variables from .env file
    load_dotenv()

    # 2. Check for Kaggle credentials in environment variables
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
        print("ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set in your .env file.")
        print("Please refer to the README.md for instructions.")
        return

    #
    # Import kaggle here, AFTER load_dotenv() has run.
    # This ensures KAGGLE_USERNAME and KAGGLE_KEY are set before kaggle tries to authenticate.
    import kaggle
    #

    # 3. Create a directory to house the dataset files
    if os.path.exists(FOLDER_NAME):
        shutil.rmtree(FOLDER_NAME)
    os.makedirs(FOLDER_NAME)
    print(f"Created temporary directory: ./{FOLDER_NAME}/")

    try:
        #
        # --- MODIFIED SECTION ---
        # 4. Copy your local log file to the temporary directory
        source_file_path = FILE_NAME # Assumes the file is in the same directory as the script

        if not os.path.exists(source_file_path):
            print(f"ERROR: Source file '{source_file_path}' not found in the current directory.")
            shutil.rmtree(FOLDER_NAME) # Clean up the created folder
            return

        destination_file_path = os.path.join(FOLDER_NAME, FILE_NAME)
        shutil.copy(source_file_path, destination_file_path)
        print(f"Copied '{source_file_path}' to temporary folder for upload.")
        # --- END MODIFIED SECTION ---
        #

        # 5. Create the dataset-metadata.json file (This is still required)
        metadata = {
            "title": DATASET_TITLE,
            "id": f"{kaggle_username}/{DATASET_SLUG}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        metadata_file_path = os.path.join(FOLDER_NAME, 'dataset-metadata.json')
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Created metadata file: {metadata_file_path}")

        # 6. Authenticate and create the dataset
        print("\nAuthenticating with Kaggle API...")
        kaggle.api.authenticate()

        print("Uploading dataset to Kaggle...")
        kaggle.api.dataset_create_new(
            folder=FOLDER_NAME,
            public=IS_PUBLIC
        )

        dataset_url = f"https://www.kaggle.com/datasets/{kaggle_username}/{DATASET_SLUG}"
        print("\n--- SUCCESS! ---")
        print(f"Dataset created successfully.")
        print(f"You can view it here: {dataset_url}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your credentials and dataset slug uniqueness.")

    finally:
        # 7. Clean up the created directory and its contents
        if os.path.exists(FOLDER_NAME):
            shutil.rmtree(FOLDER_NAME)
            print(f"\nCleaned up temporary directory: ./{FOLDER_NAME}/")

if __name__ == "__main__":
    create_dataset()