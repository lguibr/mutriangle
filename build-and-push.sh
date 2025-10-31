#!/bin/bash
# Build and push MuTriangle Docker images to DockerHub
# Usage: ./build-and-push.sh [--no-cache]

set -e

DOCKER_USERNAME="lguibr"
IMAGE_NAME="mutriangle"
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

echo "========================================"
echo "MuTriangle Docker Build & Push Script"
echo "========================================"
echo "Version: ${VERSION}"
echo "DockerHub: ${DOCKER_USERNAME}/${IMAGE_NAME}"
echo ""

# Parse arguments
NO_CACHE_FLAG=""
if [[ "$1" == "--no-cache" ]]; then
    NO_CACHE_FLAG="--no-cache"
    echo "Building with --no-cache flag"
fi

# Check if logged into DockerHub
echo "Checking DockerHub authentication..."
if ! docker info | grep -q "Username: ${DOCKER_USERNAME}"; then
    echo "Not logged into DockerHub. Running 'docker login'..."
    docker login
fi

echo ""
echo "========================================"
echo "Step 1: Building GPU Image (linux/amd64)"
echo "========================================"
docker build ${NO_CACHE_FLAG} \
    --platform linux/amd64 \
    -f Dockerfile \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:gpu \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-gpu \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:latest \
    .

echo ""
echo "========================================"
echo "Step 2: Building CPU Image (linux/amd64)"
echo "========================================"
docker build ${NO_CACHE_FLAG} \
    --platform linux/amd64 \
    -f Dockerfile.cpu \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:cpu \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-cpu \
    .

echo ""
echo "========================================"
echo "Step 3: Pushing Images to DockerHub"
echo "========================================"

echo "Pushing: ${DOCKER_USERNAME}/${IMAGE_NAME}:gpu"
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:gpu

echo "Pushing: ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-gpu"
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-gpu

echo "Pushing: ${DOCKER_USERNAME}/${IMAGE_NAME}:cpu"
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:cpu

echo "Pushing: ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-cpu"
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-cpu

echo "Pushing: ${DOCKER_USERNAME}/${IMAGE_NAME}:latest (GPU)"
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo ""
echo "========================================"
echo "✓ Build & Push Complete!"
echo "========================================"
echo ""
echo "Available images on DockerHub:"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest (GPU)"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:gpu"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-gpu"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:cpu"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-cpu"
echo ""
echo "Pull commands:"
echo "  docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:gpu"
echo "  docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:cpu"
echo ""

