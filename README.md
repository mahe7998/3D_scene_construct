# 3D Scene Construction & Recognition System

A comprehensive Deep Reinforcement Learning system for 3D scene generation and recognition using hardware-accelerated rendering and Vision LLMs.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with 24GB VRAM (PC) or M1/M2/M3 Mac
- 500GB+ free disk space
- NVIDIA Container Toolkit (for Linux/Windows with NVIDIA GPU)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 3D_scene_construct
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Build Docker containers:
```bash
docker-compose build
```

4. Download initial assets:
```bash
docker-compose run asset_downloader
```

5. Test rendering:
```bash
docker-compose run renderer --test
```

## Project Structure

```
3D_scene_construct/
├── src/                          # Source code
│   ├── asset_management/         # Asset download and management
│   ├── rendering/                # 3D rendering engine
│   ├── vision/                   # Vision LLM integration
│   ├── scene/                    # Scene generation
│   ├── rl/                       # RL training loop
│   └── utils/                    # Shared utilities
├── docker/                       # Docker configurations
├── config/                       # Configuration files
├── tests/                        # Unit and integration tests
├── data/                         # Data storage (Docker volume)
├── notebooks/                    # Jupyter notebooks for analysis
├── scripts/                      # Utility scripts
└── docker-compose.yml           # Docker orchestration
```

## Usage

### Phase 1: Asset Pipeline

Download and render 1000+ 3D assets:

```bash
# Download assets
docker-compose run asset_downloader --count 1000

# Render all assets
docker-compose run renderer --mode assets --resolution 512
```

### Phase 2: Vision LLM Annotation

Annotate rendered assets with Vision LLM:

```bash
docker-compose run annotator --batch-size 8
```

### Phase 3: Scene Generation

Generate training scenes with increasing complexity:

```bash
docker-compose run scene_generator --complexity 1 --count 1000
```

### Phase 4: RL Training

Train the RL agent for scene reconstruction:

```bash
docker-compose run rl_trainer --algorithm ppo --timesteps 1000000
```

### Monitor Training

View training progress:

```bash
docker-compose up monitor
# Open http://localhost:6006 for Tensorboard
```

## Configuration

Edit `config/config.yaml` or set environment variables in `.env`:

### Rendering Settings
- `RENDER_RESOLUTION`: Image size (default: 512)
- `RENDER_SAMPLES`: Ray tracing samples (default: 128)
- `RENDER_ENGINE`: CYCLES or EEVEE
- `RENDER_DEVICE`: GPU or CPU

### Vision LLM Settings
- `VISION_MODEL`: Model name (default: llava-1.6-vicuna-7b)
- `VISION_BATCH_SIZE`: Batch size for inference

### RL Settings
- `RL_ALGORITHM`: PPO, A2C, SAC, etc.
- `RL_LEARNING_RATE`: Learning rate
- `RL_TOTAL_TIMESTEPS`: Total training steps

## Development

### Running Tests

```bash
# Unit tests
docker-compose run tests pytest tests/unit

# Integration tests
docker-compose run tests pytest tests/integration

# Full test suite
docker-compose run tests pytest
```

### Code Quality

```bash
# Format code
docker-compose run tests black src/

# Lint
docker-compose run tests flake8 src/

# Type checking
docker-compose run tests mypy src/
```

## Architecture

See [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md) for detailed architecture documentation.

### Key Components

1. **Asset Management**: Download and organize 3D models from Objaverse/ShapeNet
2. **Rendering Engine**: Blender-based hardware-accelerated ray tracing
3. **Vision LLM**: LLaVA for scene understanding and object recognition
4. **Scene Generator**: Procedural scene composition with progressive complexity
5. **RL Agent**: PPO-based agent for scene reconstruction

## Hardware Requirements

### Minimum
- 16GB RAM
- 100GB disk space
- GPU with 8GB VRAM

### Recommended
- 32GB+ RAM
- 500GB+ SSD storage
- NVIDIA RTX GPU with 24GB VRAM or Apple M1/M2/M3

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# For Mac M1/M2/M3, ensure Metal is available in Docker
docker run --rm -it ubuntu:22.04 ls /dev/dri
```

### Blender Rendering Issues
```bash
# Test Blender in container
docker-compose run renderer blender --version
docker-compose run renderer blender --background --python-expr "import bpy; print(bpy.context.preferences.addons['cycles'].preferences.devices)"
```

### Out of Memory
- Reduce `RENDER_RESOLUTION`
- Reduce `VISION_BATCH_SIZE`
- Reduce `RL_N_ENVS`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this project in your research, please cite:

```bibtex
@software{3d_scene_construct,
  title={3D Scene Construction and Recognition with Deep RL},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/3D_scene_construct}
}
```

## Acknowledgments

- Objaverse for 3D assets
- Blender Foundation for rendering engine
- LLaVA team for Vision LLM
- Stable-Baselines3 for RL algorithms
