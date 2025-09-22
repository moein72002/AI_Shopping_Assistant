#!/bin/bash

# This script builds the Docker image for the AI Shopping Assistant.
# It loads necessary secrets from the .env file and passes them as build arguments.

# Exit immediately if a command exits with a non-zero status.
set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
else
  echo "Error: .env file not found."
  exit 1
fi

# Check if the required variables are set
if [ -z "$OPENAI_API_KEY" ] || [ -z "$TOROB_PROXY_URL" ]; then
  echo "Error: OPENAI_API_KEY and TOROB_PROXY_URL must be set in the .env file."
  exit 1
fi

echo "Starting Docker build..."

# Build the Docker image
docker build \
  --build-arg OPENAI_API_KEY="$OPENAI_API_KEY" \
  --build-arg TOROB_PROXY_URL="$TOROB_PROXY_URL" \
  -t shopping-assistant .

echo "Docker build completed successfully."
echo "You can now run the container with: docker run -p 8080:80 shopping-assistant"
