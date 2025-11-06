# Blender Docker Image for 3D Rendering
# Based on NVIDIA CUDA for GPU acceleration

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Blender version
ENV BLENDER_VERSION=4.0
ENV BLENDER_VERSION_FULL=4.0.2
ENV BLENDER_URL=https://mirrors.ocf.berkeley.edu/blender/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    wget \
    xz-utils \
    libx11-6 \
    libxi6 \
    libxxf86vm1 \
    libxfixes3 \
    libxrender1 \
    libgl1 \
    libglu1-mesa \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install Blender
RUN wget -O blender.tar.xz ${BLENDER_URL} && \
    tar -xf blender.tar.xz && \
    mv blender-${BLENDER_VERSION_FULL}-linux-x64 /opt/blender && \
    rm blender.tar.xz && \
    ln -s /opt/blender/blender /usr/local/bin/blender

# Install Python dependencies for Blender
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install additional Python packages
COPY requirements/rendering.txt /tmp/rendering.txt
RUN pip install --no-cache-dir -r /tmp/rendering.txt

# Install Blender Python packages
RUN /opt/blender/${BLENDER_VERSION}/python/bin/python3.10 -m ensurepip && \
    /opt/blender/${BLENDER_VERSION}/python/bin/python3.10 -m pip install --upgrade pip

COPY requirements/blender.txt /tmp/blender.txt
RUN /opt/blender/${BLENDER_VERSION}/python/bin/python3.10 -m pip install --no-cache-dir -r /tmp/blender.txt || true

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV BLENDER_USER_SCRIPTS=/app/src/rendering/blender_scripts
ENV BLENDER_USER_CONFIG=/tmp/blender_config

# Create data directories
RUN mkdir -p /data/assets/raw \
             /data/assets/rendered \
             /data/scenes \
             /data/database \
             /data/models \
             /data/logs \
             /data/checkpoints

# Verify Blender installation
RUN blender --version && \
    blender --background --python-expr "import bpy; print('Blender Python API OK')"

# Set default command
CMD ["python3", "-m", "src.rendering.renderer"]
