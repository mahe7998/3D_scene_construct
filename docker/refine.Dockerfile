# Render-and-compare refinement image: PyTorch + CUDA + nvdiffrast.
# Differentiable rasterizer for the pose-refinement loop. Uses nvdiffrast's CUDA
# rasterizer (no OpenGL/EGL needed in headless containers).
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip git \
    libgl1-mesa-glx libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Torch with CUDA 12.1 (matches the vision image's known-good versions).
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Mesh loading + differentiable rasterizer + IO.
RUN pip install --no-cache-dir \
    ninja \
    trimesh==4.0.10 pygltflib==1.16.1 scipy==1.11.4 networkx \
    numpy==1.24.3 imageio pillow

# Target the RTX 4090 (sm_89). MUST be set BEFORE the nvdiffrast build: there is
# no GPU during `docker build`, so torch reads this env var to pick the CUDA arch
# when compiling the _nvdiffrast_c extension (else _get_cuda_arch_flags crashes
# with "IndexError: list index out of range").
ENV TORCH_CUDA_ARCH_LIST="8.9"

# nvdiffrast is NOT on PyPI. Clone it and install with --no-build-isolation so it
# compiles its CUDA extension against the torch installed above. With setuptools>=64
# (above) this installs proper "nvdiffrast" metadata, so importlib.metadata.version
# works. Do NOT put the source dir on PYTHONPATH - the metadata-less source copy
# would shadow the installed package and re-break the version lookup.
# setuptools must be in a specific window for this torch+nvdiffrast combo:
#  - >=64 so it honors nvdiffrast's PEP 621 [project] name/version (older versions
#    ignore it and install a nameless "UNKNOWN-0.0.0" with no importable metadata);
#  - <70 so it still ships the bundled distutils that torch 2.1.2's
#    cpp_extension imports (setuptools 74+ drops it, breaking the build with
#    nvdiffrast's "Cannot compile CUDA extension" message).
# 69.5.1 sits in that window. Done in this layer to keep the torch layer cached.
RUN pip install --no-cache-dir "setuptools==69.5.1" wheel && \
    git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast && \
    pip install --no-cache-dir --no-build-isolation /opt/nvdiffrast

ENV PYTHONPATH=/app

CMD ["python3", "-c", "import torch, nvdiffrast.torch as dr; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"]
