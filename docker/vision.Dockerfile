# Vision LLM Docker Image
# PyTorch with CUDA for GPU-accelerated inference

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
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install transformers and related packages
RUN pip install --no-cache-dir \
    transformers==4.37.0 \
    accelerate==0.26.0 \
    bitsandbytes==0.42.0 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3

# Install Vision-specific dependencies
COPY requirements/vision.txt /tmp/vision.txt
RUN pip install --no-cache-dir -r /tmp/vision.txt

# Install LLaVA (if using)
RUN pip install --no-cache-dir git+https://github.com/haotian-liu/LLaVA.git

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

# Expose port for API server
EXPOSE 8000

# Set default command
CMD ["python3", "-m", "src.vision.server"]
