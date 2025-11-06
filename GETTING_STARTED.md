# Getting Started with 3D Scene Construction System

This guide will help you set up and run the 3D scene construction and recognition system.

## Prerequisites

Before you begin, ensure you have:

1. **Docker & Docker Compose** installed
   ```bash
   # Check installation
   docker --version
   docker-compose --version
   ```

2. **GPU Support** (for optimal performance)
   - **NVIDIA GPU (PC/Linux)**: Install NVIDIA Container Toolkit
     ```bash
     # Ubuntu/Debian
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
     curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo systemctl restart docker
     ```

   - **Apple M1/M2/M3 (Mac)**: Docker Desktop with Metal support (built-in)

3. **Disk Space**: At least 500GB free (for assets, renders, and models)

4. **Memory**: 16GB+ RAM recommended (32GB+ for training)

## Quick Start

### Step 1: Clone and Configure

```bash
# Clone the repository
git clone <your-repo-url>
cd 3D_scene_construct

# Create environment configuration
cp .env.example .env

# Edit .env to customize settings (optional)
nano .env
```

### Step 2: Build Docker Containers

This will take 15-30 minutes depending on your internet connection:

```bash
# Build all containers
docker-compose build

# Or build specific containers
docker-compose build base
docker-compose build blender
docker-compose build vision
docker-compose build rl
```

### Step 3: Test the Setup

```bash
# Test Blender renderer
docker-compose run renderer --test

# This should create /data/test_render.jpg
# Verify GPU support
docker-compose run blender blender --background --python-expr \
  "import bpy; print(bpy.context.preferences.addons['cycles'].preferences.compute_device_type)"
```

## Phase-by-Phase Setup

### Phase 1: Asset Pipeline

#### Download 3D Assets

Start with a small batch for testing:

```bash
# Download 10 assets for testing
docker-compose run asset_downloader --count 10

# Once confirmed working, download full set
docker-compose run asset_downloader --count 1000
```

**Expected output:**
- Assets downloaded to `./data/assets/raw/`
- Database created at `./data/database/assets.db`
- Assets organized by category

**Verification:**
```bash
# Check downloaded assets
ls -lh ./data/assets/raw/

# Check database
docker-compose run tests python -c "
from src.utils.database import Database
db = Database()
stats = len(db.get_all_objects())
print(f'Total objects in database: {stats}')
"
```

#### Render Assets

Render individual asset views:

```bash
# Test render single object
docker-compose run renderer --mode assets --object-id <object-id>

# Render all downloaded assets
docker-compose run renderer --mode assets
```

**Expected output:**
- Rendered images in `./data/assets/rendered/<category>/<object_id>/`
- 12 views per object (configurable)
- Resolution: 512x512 (configurable via `.env`)

**Monitor progress:**
```bash
# Watch render directory grow
watch -n 5 "du -sh ./data/assets/rendered/"

# Count rendered images
find ./data/assets/rendered -name "*.jpg" | wc -l
```

### Phase 2: Vision LLM Annotation

#### Start Vision Model Server

```bash
# Start vision model server (downloads model on first run)
docker-compose up -d vision_model

# Check logs
docker-compose logs -f vision_model

# Test API
curl http://localhost:8000/health
```

#### Annotate Rendered Assets

```bash
# Annotate all rendered images
docker-compose run annotator

# Or annotate specific objects
docker-compose run annotator --object-ids obj1 obj2 obj3
```

**Expected output:**
- Descriptions added to database
- Progress bar showing annotation status

**Verification:**
```bash
# Check annotations
docker-compose run tests python -c "
from src.utils.database import Database
db = Database()
renders = db.conn.execute('SELECT COUNT(*) FROM renders').fetchone()[0]
annotations = db.conn.execute('SELECT COUNT(*) FROM annotations').fetchone()[0]
print(f'Renders: {renders}, Annotations: {annotations}')
"
```

### Phase 3: Scene Generation

Generate training scenes with increasing complexity:

```bash
# Generate simple scenes (complexity level 1)
docker-compose run scene_generator --complexity 1 --count 100

# Generate more complex scenes
docker-compose run scene_generator --complexity 2 --count 100
docker-compose run scene_generator --complexity 3 --count 100
```

**Expected output:**
- Scene configurations in database
- Rendered scene images in `./data/scenes/`

### Phase 4: RL Training

#### Start Training

```bash
# Start RL training
docker-compose run rl_trainer --algorithm PPO --timesteps 1000000

# Monitor training with Tensorboard
docker-compose up -d monitor
# Open http://localhost:6006 in browser
```

#### Evaluate Model

```bash
# Evaluate trained model
docker-compose run rl_trainer --eval --model-path /data/checkpoints/PPO_final.zip
```

## Configuration

### Key Environment Variables

Edit `.env` to customize:

```bash
# Rendering
RENDER_RESOLUTION=512        # Image size (512, 1024, 2048)
RENDER_SAMPLES=128          # Ray tracing quality (higher = slower)
RENDER_DEVICE=GPU           # GPU or CPU

# Assets
ASSET_COUNT=1000            # Number of assets to download

# Vision LLM
VISION_MODEL=llava-1.6-vicuna-7b
VISION_BATCH_SIZE=8

# RL Training
RL_ALGORITHM=PPO
RL_LEARNING_RATE=0.0003
RL_TOTAL_TIMESTEPS=1000000
RL_N_ENVS=4                 # Parallel environments

# Hardware
CUDA_VISIBLE_DEVICES=0      # GPU device ID
```

### Advanced Configuration

Edit `config/config.yaml` for detailed settings:
- Camera angles and distances
- Lighting presets
- Scene complexity parameters
- Reward function weights
- Network architecture

## Monitoring and Debugging

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f renderer
docker-compose logs -f vision_model
docker-compose logs -f rl_trainer
```

### Check Resource Usage

```bash
# GPU usage
nvidia-smi

# Docker container stats
docker stats

# Disk usage
du -sh ./data/*
```

### Run Tests

```bash
# Run all tests
docker-compose run tests

# Run specific test file
docker-compose run tests pytest tests/test_database.py -v

# Run with coverage
docker-compose run tests pytest --cov=src tests/
```

## Common Issues and Solutions

### Issue: Out of Memory

**Solution:**
```bash
# Reduce batch sizes in .env
VISION_BATCH_SIZE=4
RL_N_ENVS=2

# Reduce render resolution
RENDER_RESOLUTION=256
```

### Issue: Slow Rendering

**Solution:**
```bash
# Check GPU is being used
docker-compose run renderer blender --version

# Reduce ray tracing samples
RENDER_SAMPLES=64

# Or use CPU rendering
RENDER_DEVICE=CPU
```

### Issue: Asset Download Fails

**Solution:**
```bash
# Check internet connection
# Try downloading in smaller batches
docker-compose run asset_downloader --count 100

# Check logs for specific errors
docker-compose logs asset_downloader
```

### Issue: Vision Model Not Loading

**Solution:**
```bash
# Ensure HuggingFace cache directory has space
df -h

# Download model manually
docker-compose run vision_model python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('llava-hf/llava-1.6-vicuna-7b-hf', cache_dir='/data/models')
"
```

## Next Steps

Once you have the basic pipeline running:

1. **Experiment with complexity**
   - Gradually increase scene complexity
   - Monitor RL agent performance

2. **Fine-tune models**
   - Adjust reward function weights
   - Try different RL algorithms (A2C, SAC)

3. **Scale up**
   - Download more assets
   - Increase resolution
   - Add more camera angles

4. **Customize**
   - Add new object categories
   - Create custom lighting presets
   - Implement custom reward functions

## Support

For issues and questions:
- Check [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md) for architecture details
- Review [README.md](README.md) for API documentation
- Open an issue on GitHub

## Development Workflow

### Making Changes

```bash
# Source code is mounted as volume - changes reflect immediately
# No need to rebuild for Python code changes

# Edit Python files
nano src/rendering/renderer.py

# Test changes
docker-compose run renderer --test

# Rebuild only if dependencies change
docker-compose build blender
```

### Adding New Features

1. Edit source files in `src/`
2. Update tests in `tests/`
3. Run tests: `docker-compose run tests pytest`
4. Update documentation

### Debugging

```bash
# Interactive Python shell in container
docker-compose run base python

# Interactive bash session
docker-compose run --entrypoint bash base

# Run Blender interactively (with GUI, on local machine)
blender --python src/rendering/renderer.py
```

## Performance Tips

1. **Use SSD** for data directory (faster I/O)
2. **Enable GPU** for rendering and training
3. **Increase workers** for parallel processing
4. **Use batch processing** where possible
5. **Monitor resource usage** and adjust accordingly

## Backup and Recovery

```bash
# Backup database
cp ./data/database/assets.db ./backups/assets_$(date +%Y%m%d).db

# Backup checkpoints
cp -r ./data/checkpoints ./backups/checkpoints_$(date +%Y%m%d)

# Export configuration
cp .env ./backups/.env_$(date +%Y%m%d)
cp config/config.yaml ./backups/config_$(date +%Y%m%d).yaml
```

## Congratulations!

You now have a working 3D scene construction and recognition system. Start with small experiments and gradually increase complexity as you become familiar with the system.

Happy training!
