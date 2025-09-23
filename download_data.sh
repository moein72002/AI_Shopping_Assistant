#!/bin/sh
# Exit immediately if any command fails
set -e

# The secret file is mounted by Docker at this path
SECRET_FILE="/run/secrets/kaggle_env"

echo "Sourcing Kaggle credentials..."
# Source the .env file to load KAGGLE_USERNAME and KAGGLE_KEY
. "$SECRET_FILE"
# Export the variables to make them available to subprocesses (like python)
export KAGGLE_USERNAME KAGGLE_KEY

# A quick check to ensure the variables were loaded correctly
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "ERROR: KAGGLE_USERNAME or KAGGLE_KEY not found in the secret file." >&2
    exit 1
fi

echo "Downloading dataset from Kaggle..."
python download_data_from_kaggle.py

echo "Moving dataset file..."
mv ./shopping_dataset/torob.db ./torob.db

echo "Dataset downloaded successfully! ✨"