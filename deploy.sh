#!/bin/bash

set -e  # Exit immediately if a command fails
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_TAG="chip-assistant-backup-$TIMESTAMP"

echo "🛠️  Pulling latest code..."
git pull origin main

echo "📦  Backing up current container..."
docker tag chip-assistant $BACKUP_TAG || true

echo "🧼  Stopping old container..."
docker stop chip-assistant || true
docker rm chip-assistant || true

echo "🐳  Building new image..."
docker build -t chip-assistant .

echo "🚀  Starting container..."
docker run -d \
  --name chip-assistant \
  -p 8501:8501 \
  --env-file .env.prod \
  chip-assistant

sleep 10

echo "✅ Verifying health..."
curl -fs http://localhost:8501 || {
  echo "❌ Deployment failed. Rolling back..."
  docker stop chip-assistant
  docker rm chip-assistant
  docker run -d \
    --name chip-assistant \
    -p 8501:8501 \
    --env-file .env.prod \
    $BACKUP_TAG
  echo "✅ Rolled back to previous version."
  exit 1
}

echo "🎉 Deployment successful!"
