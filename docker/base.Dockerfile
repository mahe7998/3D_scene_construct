# Base Docker Image for 3D Scene Construction System
# Ubuntu 22.04 with Python 3.10 and common dependencies

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python base dependencies
COPY requirements/base.txt /tmp/base.txt
RUN pip install --no-cache-dir -r /tmp/base.txt

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
CMD ["/bin/bash"]
