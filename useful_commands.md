docker run --rm -p 8080:8080 \
  --env-file .env \
  -e DATASETS_DIR=/datasets \
  -e KAGGLE_USERNAME= \
  -e KAGGLE_KEY= \
  -v "$PWD/datasets:/datasets" \
  shopping-assistant
