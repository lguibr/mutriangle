#!/bin/bash
# Helper script to run MuTriangle training with resource limits

set -e

# Default configuration
IMAGE_NAME="mutriangle:latest"
DATA_DIR="$(pwd)/.mutriangle_data"
CONFIG_NAME="${1:-simple}"
SEED="${2:-42}"

# Resource limits (adjust based on your system)
MEMORY_LIMIT="16g"
CPU_LIMIT="8"

# Detect GPU availability
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    GPU_STATUS="NVIDIA GPU detected"
else
    GPU_STATUS="CPU-only mode (no NVIDIA GPU)"
fi

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MuTriangle Docker Runner ===${NC}"
echo -e "${YELLOW}Configuration:${NC} $CONFIG_NAME"
echo -e "${YELLOW}Seed:${NC} $SEED"
echo -e "${YELLOW}Memory Limit:${NC} $MEMORY_LIMIT"
echo -e "${YELLOW}CPU Limit:${NC} $CPU_LIMIT"
echo -e "${YELLOW}GPU:${NC} $GPU_STATUS"
echo ""

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo -e "${RED}Error: Image $IMAGE_NAME not found${NC}"
    echo -e "${YELLOW}Run: docker-compose build${NC}"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Run training with resource limits
echo -e "${GREEN}Starting training...${NC}"
echo ""

docker run --rm -it \
    $GPU_FLAG \
    --memory="$MEMORY_LIMIT" \
    --cpus="$CPU_LIMIT" \
    --shm-size=16g \
    -v "$DATA_DIR:/app/.mutriangle_data" \
    -p 6006:6006 \
    -p 5000:5000 \
    -p 8265:8265 \
    -e NUMBA_DISABLE_JIT=1 \
    -e RAY_DISABLE_DOCKER_CPU_WARNING=1 \
    "$IMAGE_NAME" \
    train "$CONFIG_NAME" --seed "$SEED"

echo ""
echo -e "${GREEN}=== Training Complete ===${NC}"
echo -e "${YELLOW}Data saved to:${NC} $DATA_DIR"
echo -e "${YELLOW}TensorBoard:${NC} http://localhost:6006"
echo -e "${YELLOW}MLflow:${NC} http://localhost:5000"

