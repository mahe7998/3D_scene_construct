# Reinforcement Learning Docker Image
# PyTorch + Stable-Baselines3 with CUDA support

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install RL dependencies
RUN pip install --no-cache-dir \
    stable-baselines3[extra]==2.2.1 \
    gymnasium==0.29.1 \
    shimmy[gym-v26]==1.3.0

# Install monitoring and logging
RUN pip install --no-cache-dir \
    tensorboard==2.15.1 \
    wandb==0.16.2 \
    mlflow==2.9.2

# Install additional RL libraries
COPY requirements/rl.txt /tmp/rl.txt
RUN pip install --no-cache-dir -r /tmp/rl.txt

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create data directories
RUN mkdir -p /data/assets/raw \
             /data/assets/rendered \
             /data/scenes \
             /data/database \
             /data/models \
             /data/logs \
             /data/checkpoints

# Set default command
CMD ["python3", "-m", "src.rl.trainer"]
