# 🚀 MuTriangle GPU Deployment Guide

Complete guide for deploying MuTriangle on high-end GPU machines (RTX 5090, A100, H100, etc.)

## TL;DR - Deploy in 3 Commands

```bash
# 1. Pull GPU image from DockerHub
docker pull lguibr/mutriangle:gpu

# 2. Navigate to project directory (or create empty dir with docker-compose.gpu.yml)
cd mutriangle

# 3. Start training with unlimited resources
docker-compose -f docker-compose.gpu.yml up -d
```

Access monitoring:
- TensorBoard: http://localhost:6006
- MLflow: http://localhost:5000
- Ray Dashboard: http://localhost:8265

---

## Prerequisites

### 1. Docker + NVIDIA Container Toolkit

```bash
# Ubuntu/Debian - Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Verify GPU Access

```bash
# Test NVIDIA drivers
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Deployment Options

### Option A: From DockerHub (Recommended)

**Fastest deployment - no build required**

```bash
# Pull latest GPU image
docker pull lguibr/mutriangle:gpu

# Create data directory
mkdir -p .mutriangle_data

# Download docker-compose.gpu.yml
curl -O https://raw.githubusercontent.com/lguibr/mutriangle/main/docker-compose.gpu.yml

# Start training
docker-compose -f docker-compose.gpu.yml up -d
```

### Option B: Build Locally

```bash
# Clone repository
git clone https://github.com/lguibr/mutriangle.git
cd mutriangle

# Build GPU image (takes 10-15 minutes)
docker build -f Dockerfile -t lguibr/mutriangle:gpu .

# Or use docker-compose
docker-compose -f docker-compose.gpu.yml build
```

---

## RTX 5090 Optimized Configuration

The `docker-compose.gpu.yml` is optimized for maximum performance:

### Resource Allocation
- **Memory**: Unlimited (uses all available host RAM)
- **CPUs**: Unlimited (uses all available cores)
- **GPUs**: All available GPUs (`--gpus all`)
- **Shared Memory**: 32GB (for Ray distributed processing)

### CUDA Optimizations
```yaml
environment:
  # Use all GPUs
  - CUDA_VISIBLE_DEVICES=all
  
  # Disable blocking for async kernel launches
  - CUDA_LAUNCH_BLOCKING=0
  
  # Enable TF32 for Ampere+ GPUs (RTX 30XX/40XX/50XX)
  - TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
  
  # cuDNN optimizations
  - TORCH_CUDNN_V8_API_ENABLED=1
  
  # Lazy module loading for faster startup
  - CUDA_MODULE_LOADING=LAZY
  
  # CPU threading for data loading
  - OMP_NUM_THREADS=16
  - MKL_NUM_THREADS=16
```

### System Optimizations
```yaml
# Use host IPC namespace for better Ray performance
ipc: host

# Increased ulimits for Ray workers
ulimits:
  nofile: 65536
  memlock: -1
  stack: 67108864
```

---

## Training Configurations

### Large-Scale Training (Recommended for RTX 5090)

```bash
# Using docker-compose (detached mode)
docker-compose -f docker-compose.gpu.yml up -d

# Using docker run directly
docker run -d --gpus all --shm-size=32g \
  --ipc=host --restart=unless-stopped \
  -v "$(pwd)/.mutriangle_data:/app/.mutriangle_data" \
  -p 6006:6006 -p 5000:5000 -p 8265:8265 \
  lguibr/mutriangle:gpu train large --seed 42 --run-name rtx5090_run
```

### Custom Configuration

```bash
# Interactive mode - choose config
docker-compose -f docker-compose.gpu.yml run mutriangle train

# Specific config with custom parameters
docker-compose -f docker-compose.gpu.yml run mutriangle \
  train large --seed 123 --run-name production_v1
```

### Multiple Runs (Hyperparameter Search)

```bash
# Run multiple seeds in parallel containers
for seed in 42 123 456; do
  docker run -d --name mutriangle-seed-${seed} \
    --gpus all --shm-size=32g --ipc=host \
    -v "$(pwd)/.mutriangle_data:/app/.mutriangle_data" \
    lguibr/mutriangle:gpu train large --seed ${seed} --run-name seed_${seed}
done
```

---

## Monitoring & Management

### Access Monitoring Services

```bash
# TensorBoard - training metrics and graphs
open http://localhost:6006

# MLflow - experiment tracking
open http://localhost:5000

# Ray Dashboard - distributed worker status
open http://localhost:8265
```

### View Logs

```bash
# Follow logs in real-time
docker-compose -f docker-compose.gpu.yml logs -f

# Last 100 lines
docker logs --tail 100 mutriangle-gpu

# Specific worker logs (inside container)
docker exec mutriangle-gpu tail -f /app/.mutriangle_data/MuTriangle/runs/*/logs/worker_*.log
```

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Inside container
docker exec mutriangle-gpu nvidia-smi
```

### Monitor Container Resources

```bash
# Real-time stats
docker stats mutriangle-gpu

# Detailed inspection
docker inspect mutriangle-gpu
```

---

## Performance Tuning

### Maximize GPU Utilization

1. **Increase batch size** in configuration:
```bash
docker-compose -f docker-compose.gpu.yml run mutriangle edit large
# Set BATCH_SIZE=512 or 1024 (depends on GPU memory)
```

2. **Increase MCTS simulations**:
```bash
# Edit config: max_simulations=128 or 256
# More simulations = better policy, slower self-play
```

3. **Increase self-play workers**:
```bash
# Edit config: NUM_SELF_PLAY_WORKERS=-1 (auto-detect all cores)
# Or set to specific count: NUM_SELF_PLAY_WORKERS=32
```

### Optimize for RTX 5090 (32GB VRAM)

Recommended large configuration:
- `BATCH_SIZE`: 1024
- `BUFFER_CAPACITY`: 1000000
- `NUM_SELF_PLAY_WORKERS`: -1 (auto)
- `max_simulations`: 256
- `UNROLL_STEPS`: 10
- `HIDDEN_STATE_DIM`: 256
- `NUM_RESIDUAL_BLOCKS`: 20

### Memory Optimization

```bash
# If running out of GPU memory, reduce batch size
docker-compose -f docker-compose.gpu.yml run mutriangle edit large
# Set BATCH_SIZE=256

# Monitor memory usage
watch -n 1 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
```

---

## Data Persistence

All training data persists in `.mutriangle_data/`:

```
.mutriangle_data/
└── MuTriangle/
    ├── configs/          # Configuration files
    ├── mlruns/           # MLflow experiment tracking
    └── runs/             # Training runs
        └── <run_name>/
            ├── checkpoints/      # Model weights (*.pt)
            ├── buffers/          # Replay buffer (*.pkl)
            ├── logs/             # Text logs
            ├── tensorboard/      # TensorBoard events
            └── profile_data/     # Performance profiles
```

### Backup Training Data

```bash
# Create timestamped backup
tar -czf mutriangle-backup-$(date +%Y%m%d-%H%M%S).tar.gz .mutriangle_data/

# Exclude large replay buffers
tar -czf mutriangle-backup-no-buffers.tar.gz \
  --exclude='.mutriangle_data/*/runs/*/buffers' \
  .mutriangle_data/
```

### Resume Training

```bash
# Data automatically persists between container restarts
docker-compose -f docker-compose.gpu.yml up -d

# Or explicitly with same run name
docker-compose -f docker-compose.gpu.yml run mutriangle \
  train large --run-name my_continued_run
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Verify nvidia-smi works
nvidia-smi

# Check NVIDIA Container Toolkit
sudo systemctl status docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
```

### Out of GPU Memory

```bash
# Reduce batch size
docker-compose -f docker-compose.gpu.yml run mutriangle edit large
# Set BATCH_SIZE=128 or 256

# Or use gradient accumulation (future feature)
```

### Container Exits Immediately

```bash
# Check logs for errors
docker-compose -f docker-compose.gpu.yml logs

# Run interactively for debugging
docker-compose -f docker-compose.gpu.yml run mutriangle bash
mutriangle list
mutriangle train large --seed 42
```

### Port Already in Use

```bash
# Find process using port
sudo lsof -i :6006

# Kill process or change ports in docker-compose.gpu.yml
ports:
  - "6007:6006"  # Map to different host port
```

### Slow Training

1. Check GPU utilization: `nvidia-smi` (should be >80%)
2. Check CPU usage: `docker stats` (should use most cores)
3. Increase batch size if GPU memory allows
4. Verify data is on fast SSD (not network mount)
5. Check Ray Dashboard for worker bottlenecks

---

## Production Best Practices

### 1. Use Detached Mode with Restart Policy

```bash
# Already configured in docker-compose.gpu.yml
docker-compose -f docker-compose.gpu.yml up -d
```

### 2. Set Up Log Rotation

```bash
# Configure Docker daemon log rotation
sudo nano /etc/docker/daemon.json
```

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  }
}
```

```bash
sudo systemctl restart docker
```

### 3. Automated Backups

```bash
# Add to crontab (daily backup at 2 AM)
crontab -e
```

```cron
0 2 * * * cd /path/to/mutriangle && tar -czf backups/mutriangle-$(date +\%Y\%m\%d).tar.gz .mutriangle_data/
```

### 4. Monitoring Alerts

Set up alerts for:
- Container crash: `docker-compose -f docker-compose.gpu.yml ps`
- GPU temperature: `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader`
- Disk space: `df -h`

---

## Multi-GPU Training

```bash
# Use all GPUs (default)
docker-compose -f docker-compose.gpu.yml up -d

# Verify all GPUs are visible
docker exec mutriangle-gpu nvidia-smi

# Ray automatically distributes workers across GPUs
# Check Ray Dashboard at http://localhost:8265
```

### GPU Selection

```bash
# Use specific GPUs (modify docker-compose.gpu.yml)
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use only GPU 0 and 1

# Or in docker run
docker run --gpus '"device=0,1"' ...
```

---

## Cost Optimization (Cloud GPUs)

### Spot Instances / Preemptible VMs

```bash
# Ensure auto-restart on VM preemption
restart: unless-stopped  # Already in docker-compose.gpu.yml

# Save checkpoints frequently (edit config)
# CHECKPOINT_INTERVAL: 1000  # Save every 1000 steps
```

### Suspend/Resume Training

```bash
# Stop container (data persists)
docker-compose -f docker-compose.gpu.yml stop

# Resume later
docker-compose -f docker-compose.gpu.yml start
```

### Optimize Training Duration

```bash
# Set max training steps
docker-compose -f docker-compose.gpu.yml run mutriangle edit large
# MAX_TRAINING_STEPS: 500000  # Adjust based on budget
```

---

## Example: Full RTX 5090 Deployment

```bash
# 1. Rent GPU machine (e.g., Lambda Labs, Vast.ai, RunPod)
#    - RTX 5090 (32GB VRAM)
#    - 32+ CPU cores
#    - 128+ GB RAM
#    - 500+ GB SSD

# 2. SSH into machine
ssh user@gpu-machine-ip

# 3. Install Docker + NVIDIA toolkit (if not installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
# ... (install nvidia-container-toolkit as shown above)

# 4. Pull MuTriangle
docker pull lguibr/mutriangle:gpu

# 5. Create project directory
mkdir mutriangle-training && cd mutriangle-training
mkdir .mutriangle_data

# 6. Download docker-compose.gpu.yml
curl -O https://raw.githubusercontent.com/lguibr/mutriangle/main/docker-compose.gpu.yml

# 7. Start training
docker-compose -f docker-compose.gpu.yml up -d

# 8. Monitor via SSH tunnel (from local machine)
ssh -L 6006:localhost:6006 -L 5000:localhost:5000 user@gpu-machine-ip
# Open http://localhost:6006 in browser

# 9. Check progress
docker-compose -f docker-compose.gpu.yml logs -f
```

---

## Quick Reference Commands

```bash
# Pull latest image
docker pull lguibr/mutriangle:gpu

# Start training (detached)
docker-compose -f docker-compose.gpu.yml up -d

# Stop training
docker-compose -f docker-compose.gpu.yml stop

# View logs
docker-compose -f docker-compose.gpu.yml logs -f

# Interactive shell
docker-compose -f docker-compose.gpu.yml run mutriangle bash

# List configurations
docker-compose -f docker-compose.gpu.yml run mutriangle list

# Edit configuration
docker-compose -f docker-compose.gpu.yml run mutriangle edit large

# Check GPU usage
docker exec mutriangle-gpu nvidia-smi

# Container stats
docker stats mutriangle-gpu

# Backup data
tar -czf backup.tar.gz .mutriangle_data/

# Clean up
docker-compose -f docker-compose.gpu.yml down
docker system prune -a
```

---

## Support & Resources

- **Main README**: [README.md](README.md)
- **Docker Guide**: [DOCKER.md](DOCKER.md)
- **DockerHub**: https://hub.docker.com/r/lguibr/mutriangle
- **GitHub**: https://github.com/lguibr/mutriangle
- **Issues**: https://github.com/lguibr/mutriangle/issues

---

**Happy training on your RTX 5090! 🚀**

