# Multi-stage Dockerfile for MuTriangle ML Training
# GPU-enabled with CUDA 11.8 support

# ============================================================================
# Stage 1: Builder - Compile C++ dependencies
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    cmake \
    g++ \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support and compatible NumPy
RUN pip install "numpy<2.0" torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Create workspace
WORKDIR /build

# Clone and install trianglengin (C++ game engine)
RUN git clone https://github.com/lguibr/trianglengin.git && \
    cd trianglengin && \
    pip install -v .

# Clone and install mutrimcts (C++ MCTS library, depends on trianglengin)
RUN git clone https://github.com/lguibr/mutrimcts.git && \
    cd mutrimcts && \
    pip install -v .

# Clone and install trieye
RUN git clone https://github.com/lguibr/trieye.git && \
    cd trieye && \
    pip install -v .

# ============================================================================
# Stage 2: Runtime - Lean production image
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NUMBA_DISABLE_JIT=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Set working directory
WORKDIR /app

# Copy only mutriangle package (dependencies already built from git in builder)
COPY . /app/

# Install mutriangle package
RUN pip install -e .

# Create data directory
RUN mkdir -p /app/.mutriangle_data

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set entrypoint to custom script that starts monitoring + training
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["train", "simple", "--seed", "42"]

