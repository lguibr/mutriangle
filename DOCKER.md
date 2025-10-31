# 🐳 MuTriangle Docker Guide

Run MuTriangle training in an isolated container with GPU support and resource controls.

## Quick Start (DockerHub)

**Pull pre-built images from DockerHub:**

```bash
# GPU version (recommended for NVIDIA GPUs)
docker pull lguibr/mutriangle:gpu

# CPU version (for Mac M1/M2/M3/M4 or systems without GPU)
docker pull lguibr/mutriangle:cpu

# Latest version (always points to GPU)
docker pull lguibr/mutriangle:latest
```

**Run immediately with GPU:**

```bash
# Using docker-compose (recommended)
docker-compose -f docker-compose.gpu.yml up

# Or run directly
docker run --rm -it --gpus all \
  -v "$(pwd)/.mutriangle_data:/app/.mutriangle_data" \
  -p 6006:6006 -p 5000:5000 -p 8265:8265 \
  lguibr/mutriangle:gpu train large --seed 42
```

**See [DOCKER_GPU_DEPLOY.md](DOCKER_GPU_DEPLOY.md) for optimized RTX 5090 / high-end GPU deployment.**

## Prerequisites

### 1. Docker Installation

Install Docker Engine:
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# macOS
brew install --cask docker

# Or download from https://www.docker.com/products/docker-desktop
```

### 2. NVIDIA Container Toolkit (GPU Support)

Required for GPU-accelerated training:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Option 1: Pull from DockerHub (Fastest)

```bash
# GPU version
docker pull lguibr/mutriangle:gpu

# CPU version
docker pull lguibr/mutriangle:cpu
```

### Option 2: Build Locally

**For systems with NVIDIA GPU:**
```bash
# Navigate to mutriangle directory
cd mutriangle

# Build GPU version
docker build -f Dockerfile -t lguibr/mutriangle:gpu .

# Or using docker-compose
docker-compose -f docker-compose.gpu.yml build
```

**For Mac (M1/M2/M3/M4) or CPU-only systems:**
```bash
# Navigate to mutriangle directory
cd mutriangle

# Build CPU-only version
docker build -f Dockerfile.cpu -t lguibr/mutriangle:cpu .

# Or use CPU-only compose file
docker-compose -f docker-compose.cpu.yml build
```

Build takes ~10-15 minutes (compiles C++ dependencies).

### Run Training

**Option 1: Using docker-compose (Recommended)**
```bash
# High-end GPU systems (RTX 5090, A100, etc.) - UNLIMITED RESOURCES
docker-compose -f docker-compose.gpu.yml up

# GPU systems with resource limits
docker-compose run mutriangle train simple --seed 42

# CPU-only (Mac M1/M2/M3/M4)
docker-compose -f docker-compose.cpu.yml run mutriangle train simple --seed 42

# Run in detached mode
docker-compose -f docker-compose.gpu.yml up -d

# View logs
docker-compose -f docker-compose.gpu.yml logs -f mutriangle
```

**Option 2: Using helper script (auto-detects GPU)**
```bash
# From mutriangle directory
cd mutriangle

# Automatically uses GPU if available, otherwise CPU
./docker-run.sh simple 42

# Run with large config
./docker-run.sh large 123
```

**Option 3: Manual docker run**
```bash
# From mutriangle directory - GPU with DockerHub image
docker run --rm -it \
  --gpus all \
  --shm-size=32g \
  -v "$(pwd)/.mutriangle_data:/app/.mutriangle_data" \
  -p 6006:6006 -p 5000:5000 -p 8265:8265 \
  lguibr/mutriangle:gpu train large --seed 42

# With resource limits
docker run --rm -it \
  --gpus all \
  --memory="16g" \
  --cpus="8" \
  --shm-size=2g \
  -v "$(pwd)/.mutriangle_data:/app/.mutriangle_data" \
  -p 6006:6006 -p 5000:5000 -p 8265:8265 \
  lguibr/mutriangle:gpu train simple --seed 42
```

## Resource Management

### CPU Limits
```bash
# Limit to 4 CPU cores
docker run --cpus="4" ...

# Limit to 50% of available CPUs
docker run --cpus="0.5" ...
```

### Memory Limits
```bash
# Limit to 8GB RAM
docker run --memory="8g" ...

# With swap limit
docker run --memory="8g" --memory-swap="10g" ...
```

### GPU Control
```bash
# Use all GPUs
docker run --gpus all ...

# Use specific GPU
docker run --gpus device=0 ...

# Use 2 GPUs
docker run --gpus 2 ...
```

### Combined Example
```bash
docker run --rm -it \
  --gpus device=0 \
  --memory="16g" \
  --cpus="8" \
  --shm-size=2g \
  -v "$(pwd)/.mutriangle_data:/app/.mutriangle_data" \
  -p 6006:6006 -p 5000:5000 \
  mutriangle:latest train medium --seed 42
```

## Configuration Management

### List Available Configurations
```bash
docker-compose run mutriangle list
```

### Create New Configuration
```bash
docker-compose run mutriangle create
```

### Show Configuration Details
```bash
docker-compose run mutriangle show simple
```

### Set Default Configuration
```bash
docker-compose run mutriangle default large
```

## Monitoring

### TensorBoard
```bash
# Access in browser
open http://localhost:6006

# Or run dedicated TensorBoard service
docker-compose --profile monitoring up tensorboard
```

### MLflow
```bash
# Access in browser
open http://localhost:5000

# Or run dedicated MLflow service
docker-compose --profile monitoring up mlflow
```

### Ray Dashboard
```bash
# Access in browser
open http://localhost:8265
```

### Container Logs
```bash
# Follow logs in real-time
docker-compose logs -f mutriangle

# View last 100 lines
docker logs --tail 100 mutriangle-training
```

## Data Persistence

All training data is stored in `.mutriangle_data/` which is mounted as a volume:

```
.mutriangle_data/
└── MuTriangle/
    ├── configs/          # Configuration files
    ├── mlruns/           # MLflow tracking data
    └── runs/             # Training runs
        └── <run_name>/
            ├── checkpoints/
            ├── buffers/
            ├── logs/
            ├── tensorboard/
            └── profile_data/
```

**Data persists between container restarts.**

## Advanced Usage

### Interactive Shell
```bash
# Enter container shell
docker-compose run mutriangle bash

# Inside container
mutriangle list
mutriangle train toy --seed 42
exit
```

### Custom Command
```bash
# Run any mutriangle CLI command
docker-compose run mutriangle stats --host 0.0.0.0 --port 6006
```

### Resume Training
```bash
# Data is persisted in .mutriangle_data/
# Simply rerun with same config and run name
docker-compose run mutriangle train simple --run-name my_run
```

### Profiling
```bash
docker-compose run mutriangle train simple --profile

# Analyze profiles (from host)
python scripts/analyze_profiles.py .mutriangle_data/MuTriangle/runs/*/profile_data/*.prof
```

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA drivers
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check nvidia-container-toolkit
sudo systemctl status docker
```

### Out of Memory
```bash
# Increase memory limit
docker run --memory="32g" --shm-size=4g ...

# Or reduce batch size in config
docker-compose run mutriangle edit simple
# Set BATCH_SIZE to 128 or lower
```

### Port Already in Use
```bash
# Use different ports
docker run -p 6007:6006 -p 5001:5000 ...

# Or stop conflicting services
sudo lsof -i :6006
```

### Container Exits Immediately
```bash
# Check logs
docker-compose logs mutriangle

# Run with interactive shell
docker-compose run mutriangle bash
```

### Build Failures
```bash
# Clean rebuild
docker-compose build --no-cache

# Check disk space
df -h

# Check Docker daemon
sudo systemctl status docker
```

## Performance Tips

1. **Use SSD for data volume**: Mount `.mutriangle_data/` on SSD
2. **Increase shared memory**: Set `--shm-size=4g` or higher for Ray
3. **Pin CPU cores**: Use `--cpuset-cpus="0-7"` for dedicated cores
4. **Monitor resources**: Use `docker stats` to track usage
5. **Batch size tuning**: Start small, increase if GPU memory allows

## Clean Up

```bash
# Stop all containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove image
docker rmi mutriangle:latest

# Clean Docker cache
docker system prune -a
```

## Environment Variables

Key environment variables:

```bash
# Disable Numba JIT (default: 1)
NUMBA_DISABLE_JIT=1

# Select specific GPU
CUDA_VISIBLE_DEVICES=0

# Ray memory limits
RAY_memory=16000000000  # 16GB
```

Set in `docker-compose.yml` or via `-e` flag:
```bash
docker run -e CUDA_VISIBLE_DEVICES=1 ...
```

## Production Deployment

### High-End GPU Deployment (RTX 5090, A100)

See **[DOCKER_GPU_DEPLOY.md](DOCKER_GPU_DEPLOY.md)** for complete RTX 5090 deployment guide.

```bash
# Pull latest GPU image
docker pull lguibr/mutriangle:gpu

# Run with GPU compose (unlimited resources)
docker-compose -f docker-compose.gpu.yml up -d

# Monitor progress
docker-compose -f docker-compose.gpu.yml logs -f
```

### Long-Running Training
```bash
# GPU systems - already has restart policy
docker-compose -f docker-compose.gpu.yml up -d

# Monitor progress
docker-compose -f docker-compose.gpu.yml logs -f
```

### Automated Backups
```bash
# Backup training data
tar -czf mutriangle-backup-$(date +%Y%m%d).tar.gz .mutriangle_data/

# Restore
tar -xzf mutriangle-backup-20250101.tar.gz
```

### Multi-GPU Training
```bash
# Use all GPUs
docker run --gpus all ...

# Ray will automatically distribute workers across GPUs
```

## Building and Pushing to DockerHub (Maintainers)

```bash
# Build and push both GPU and CPU images
./build-and-push.sh

# Build with --no-cache flag
./build-and-push.sh --no-cache
```

This creates and pushes:
- `lguibr/mutriangle:gpu` - GPU version
- `lguibr/mutriangle:cpu` - CPU version  
- `lguibr/mutriangle:latest` - Points to GPU
- `lguibr/mutriangle:X.Y.Z-gpu` - Versioned GPU
- `lguibr/mutriangle:X.Y.Z-cpu` - Versioned CPU

## References

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose GPU Support](https://docs.docker.com/compose/gpu-support/)
- [DockerHub Repository](https://hub.docker.com/r/lguibr/mutriangle)
- [RTX 5090 GPU Deployment Guide](DOCKER_GPU_DEPLOY.md)
- [MuTriangle Main README](README.md)

---

**Ready to containerize your training? Pull from DockerHub: `docker pull lguibr/mutriangle:gpu` 🚀**

