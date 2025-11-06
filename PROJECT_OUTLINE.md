# 3D Scene Construction & Recognition with Deep RL

## Project Overview

A comprehensive system that uses Deep Reinforcement Learning, hardware-accelerated 3D rendering (Vulkan/Ray Tracing), and trainable Vision LLMs to:
1. Generate 3D scenes with various objects
2. Identify and categorize objects within scenes
3. Progressively improve scene reconstruction capabilities through RL

## Architecture Components

### 1. Technology Stack

#### Deep Reinforcement Learning
- **Library**: Stable-Baselines3 (SB3)
- **Rationale**: Most mature Python RL library, excellent documentation, supports custom environments
- **Alternative**: Ray RLlib (for distributed training if needed later)

#### 3D Rendering Engine
- **Primary**: Blender Python API (bpy) with Cycles renderer
- **Backend**: Vulkan via Blender's Cycles (OptiX for NVIDIA, Metal for M1)
- **Rationale**:
  - Full Python API
  - Headless rendering support
  - Hardware-accelerated ray tracing
  - Handles all 3D formats
  - Cross-platform (M1 Mac + NVIDIA GPU)

#### Vision LLM
- **Primary**: LLaVA-1.6 or LLaVA-NeXT
- **Alternative**: CLIP + GPT-4V API, Qwen-VL, or InternVL
- **Fine-tuning**: LoRA/QLoRA for efficient training

#### 3D Asset Management
- **Libraries**: trimesh, pygltflib, bpy
- **Database**: SQLite for metadata, JSON for hierarchical categories
- **Storage**: Docker volumes for persistent storage

### 2. Hardware Requirements

#### Development Platforms
- **Mac**: M1/M2/M3 with Metal acceleration
- **PC**: NVIDIA RTX GPU (24GB VRAM) with CUDA + OptiX
- **Docker**: NVIDIA Container Toolkit for GPU passthrough

### 3. Asset Sources (~1000 objects)

1. **Objaverse** (Primary)
   - 800K+ 3D objects with CC licenses
   - API available via objaverse-xl
   - Diverse categories

2. **ShapeNet**
   - Academic dataset
   - Well-categorized

3. **Free3D / CGTrader**
   - Free models with proper licenses
   - High quality meshes

## Phase Implementation Plan

### Phase 1: Asset Pipeline
**Goal**: Download, render, and catalog 1000+ 3D assets

#### Components:
1. **Asset Downloader** (`src/asset_downloader.py`)
   - Download from Objaverse/ShapeNet
   - Organize by category in volume
   - Store metadata (URL, format, license)

2. **Asset Renderer** (`src/renderer.py`)
   - Blender headless rendering
   - Multiple views per asset (6-12 angles)
   - Configurable resolution (default: 512x512)
   - Proper scaling/framing (entire object visible)
   - Various lighting conditions

3. **Output Structure**:
   ```
   /data/
     assets/
       raw/
         <category>/
           <object_id>/
             model.obj
             textures/
             metadata.json
       rendered/
         <category>/
           <object_id>/
             view_001.jpg
             view_002.jpg
             ...
             render_metadata.json
     database/
       assets.db (SQLite)
       categories.json
   ```

4. **Metadata Schema**:
   ```json
   {
     "object_id": "uuid",
     "name": "object_name",
     "category": "primary_category",
     "subcategories": ["tag1", "tag2"],
     "source_url": "download_url",
     "file_path": "relative/path/to/model",
     "bounding_box": {"min": [], "max": []},
     "rendered_views": [
       {
         "view_id": "001",
         "image_path": "path/to/image.jpg",
         "camera_params": {
           "position": [x, y, z],
           "rotation": [rx, ry, rz],
           "focal_length": 50
         },
         "lighting": "default|bright|dark|..."
       }
     ]
   }
   ```

### Phase 2: Vision LLM Integration
**Goal**: Describe and categorize each rendered asset

#### Components:
1. **Vision Model Service** (`src/vision_model.py`)
   - Load LLaVA or similar model
   - Inference API for image description
   - Category prediction
   - Attribute extraction (color, shape, orientation)

2. **Asset Annotator** (`src/annotator.py`)
   - Process all rendered images
   - Generate descriptions
   - Extract categories/attributes
   - Store in database

3. **Description Schema**:
   ```json
   {
     "object_id": "uuid",
     "view_id": "001",
     "description": "A red sports car facing left...",
     "predicted_category": "vehicle/car/sports",
     "attributes": {
       "color": "red",
       "orientation": "left-facing",
       "visible_parts": ["front", "side"],
       "confidence": 0.95
     }
   }
   ```

### Phase 3: Scene Generation & Training
**Goal**: Create complex scenes and train Vision LLM

#### Components:
1. **Scene Generator** (`src/scene_generator.py`)
   - Compose multiple objects
   - Random/controlled placement
   - Collision detection (basic)
   - Ground plane with varying complexity
   - Configurable complexity levels

2. **Scene Complexity Progression**:
   - **Level 1**: 1-3 objects, flat ground, simple lighting
   - **Level 2**: 3-5 objects, textured ground, varied lighting
   - **Level 3**: 5-10 objects, complex ground, partial occlusions
   - **Level 4**: 10+ objects, realistic environments, heavy occlusions

3. **Scene Metadata**:
   ```json
   {
     "scene_id": "uuid",
     "complexity_level": 2,
     "objects": [
       {
         "object_id": "uuid",
         "position": [x, y, z],
         "rotation": [rx, ry, rz],
         "scale": [sx, sy, sz],
         "occlusion_percentage": 0.3
       }
     ],
     "environment": {
       "ground_type": "grass|concrete|...",
       "lighting": "parameters",
       "camera": "parameters"
     },
     "rendered_image": "path/to/scene.jpg"
   }
   ```

4. **Vision LLM Training** (`src/train_vision.py`)
   - Fine-tune on generated scenes
   - Supervised learning: scene → ground truth labels
   - Incremental training as complexity increases
   - Metrics: accuracy, precision, recall per object

### Phase 4: Deep RL Loop
**Goal**: Reconstruct scenes based on Vision LLM output

#### Components:
1. **RL Environment** (`src/rl_environment.py`)
   - OpenAI Gym interface
   - State: Vision LLM scene description
   - Action: Place object with parameters
   - Reward: Scene reconstruction accuracy

2. **State Space**:
   - Vision LLM output (scene description)
   - Current scene reconstruction state
   - Available objects in database

3. **Action Space**:
   - Select object from database
   - Set position (x, y, z)
   - Set rotation (rx, ry, rz)
   - Set scale (sx, sy, sz)
   - Or "done" action

4. **Reward Function**:
   ```python
   reward = (
     w1 * object_identification_accuracy +  # % correct objects
     w2 * position_accuracy +               # Distance error
     w3 * rotation_accuracy +               # Angle error
     w4 * scale_accuracy +                  # Scale error
     w5 * scene_completeness +              # All objects found
     w6 * false_positive_penalty            # Wrong objects added
   )
   ```

5. **Training Strategy**:
   - Curriculum learning: start simple, increase complexity
   - Co-training: alternately train Vision LLM and RL agent
   - Vision LLM update every N RL episodes
   - Dynamic complexity adjustment based on performance

6. **RL Agent** (`src/rl_agent.py`)
   - Algorithm: PPO (Proximal Policy Optimization)
   - Network architecture:
     - Scene encoder: Process Vision LLM output
     - Object database encoder: Embedding of available objects
     - Policy network: Actor-Critic
   - Training loop with SB3

## Docker Architecture

### Container Strategy
Multi-stage builds with separate containers:

1. **Base Container** (`docker/base.Dockerfile`)
   - Ubuntu 22.04 or similar
   - Python 3.10+
   - CUDA toolkit (for NVIDIA)
   - Common dependencies

2. **Blender Container** (`docker/blender.Dockerfile`)
   - Blender 4.0+ with Python API
   - Headless rendering support
   - GPU drivers (CUDA/Metal)

3. **Vision LLM Container** (`docker/vision.Dockerfile`)
   - PyTorch with GPU support
   - Transformers library
   - LLaVA model weights

4. **RL Container** (`docker/rl.Dockerfile`)
   - Stable-Baselines3
   - Training infrastructure
   - Tensorboard for monitoring

5. **Services** (`docker-compose.yml`)
   ```yaml
   services:
     asset_downloader:
       # Downloads and organizes assets

     renderer:
       # Renders individual assets and scenes

     vision_model:
       # Vision LLM inference service

     annotator:
       # Annotates rendered images

     scene_generator:
       # Generates training scenes

     rl_trainer:
       # RL training loop

     db:
       # SQLite or PostgreSQL for metadata

     monitor:
       # Tensorboard + monitoring dashboard
   ```

### Volume Structure
```yaml
volumes:
  data:
    # Persistent storage for assets and renders
  models:
    # Pre-trained and fine-tuned model weights
  logs:
    # Training logs and tensorboard data
  checkpoints:
    # Model checkpoints
```

## Configuration Management

### Environment Variables
```bash
# Rendering
RENDER_RESOLUTION=512
RENDER_SAMPLES=128
RENDER_ENGINE=CYCLES
RENDER_DEVICE=GPU

# Asset Management
ASSET_COUNT=1000
ASSET_VIEWS_PER_OBJECT=12

# Vision LLM
VISION_MODEL=llava-1.6-vicuna-7b
VISION_BATCH_SIZE=8

# RL
RL_ALGORITHM=PPO
RL_LEARNING_RATE=3e-4
RL_TOTAL_TIMESTEPS=1000000

# Hardware
GPU_DEVICE=0
NUM_WORKERS=4
```

## Testing Strategy

### Unit Tests
- Asset downloader
- Renderer (single object)
- Vision model inference
- Scene generator
- RL environment step function

### Integration Tests
- Full pipeline: download → render → annotate
- Scene generation → vision inference
- RL training loop (few steps)

### System Tests
- End-to-end on small dataset (10 objects)
- Performance benchmarks
- Cross-platform testing (Mac + PC)

## Progressive Complexity Schedule

### Week 1-2: Foundation
- Set up infrastructure
- Download 100 assets
- Single object rendering
- Basic Vision LLM inference

### Week 3-4: Simple Scenes
- 2-3 objects per scene
- Flat ground
- Simple lighting
- Initial RL training

### Week 5-6: Intermediate Scenes
- 3-5 objects
- Textured ground
- Varied lighting
- Co-training Vision + RL

### Week 7-8: Complex Scenes
- 5-10 objects
- Partial occlusions
- Complex environments
- Performance optimization

### Week 9+: Advanced Scenarios
- 10+ objects
- Heavy occlusions
- Realistic environments
- Full-scale evaluation

## Success Metrics

### Phase 1 Metrics
- Asset download success rate > 95%
- Rendering success rate > 99%
- Average render time < 10s per image

### Phase 2 Metrics
- Category prediction accuracy > 85%
- Description quality (human eval) > 4/5

### Phase 3 Metrics
- Object detection F1 score > 0.80
- Attribute prediction accuracy > 75%
- Occlusion handling (IoU) > 0.60

### Phase 4 Metrics
- Scene reconstruction accuracy > 70%
- Position error < 10% of scene size
- Rotation error < 15 degrees
- RL convergence within 1M steps

## Development Priorities

1. **Immediate**: Set up Docker environment and basic infrastructure
2. **Week 1**: Phase 1 implementation (asset pipeline)
3. **Week 2**: Phase 2 implementation (Vision LLM)
4. **Week 3-4**: Phase 3 implementation (scene generation)
5. **Week 5+**: Phase 4 implementation (RL loop)

## Risk Mitigation

### Technical Risks
1. **Cross-platform GPU support**: Test early on both platforms
2. **Blender headless issues**: Have fallback to PyTorch3D
3. **Vision LLM performance**: Start with smaller model, scale up
4. **RL convergence**: Careful reward shaping, curriculum learning

### Resource Risks
1. **Storage**: Monitor disk usage, implement cleanup
2. **Compute time**: Optimize rendering, use distributed training
3. **Memory**: Batch processing, gradient checkpointing

## Next Steps

1. Initialize project structure
2. Create Docker containers
3. Implement Phase 1: Asset downloader
4. Implement Phase 1: Basic renderer
5. Test on 10 sample objects
6. Iterate based on results
