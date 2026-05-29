# CLAUDE.md - Project Context for Claude Code

## Project Overview

**Goal**: Build a self-improving 3D vision system using Deep Reinforcement Learning.

### The Core Loop

1. **RL Agent constructs a 3D scene**
   - Selects objects from Objaverse dataset (~1000 assets)
   - Places them on a flat floor using PyBullet physics (90% upright, realistic gravity/collisions)
   - Chooses positions, rotations, scales for each object
   - Scene complexity increases over time (start 1-2 objects, scale to 5-10)

2. **Render the scene with realistic lighting**
   - Blender Cycles engine (GPU-accelerated on RTX 4090)
   - Stereo cameras (6.5cm baseline = human eye spacing) for depth perception
   - Multiple viewpoints (6 angles around scene)
   - Realistic lighting: shadows, reflections, ambient occlusion

3. **Vision LLM analyzes the rendered images**
   - LLaVA-1.6 (multimodal LLM) views stereo pairs
   - Identifies: what objects are present, where they are (3D position), orientation, scale
   - Outputs structured predictions for each object

4. **Compute reward signal**
   - Compare Vision LLM predictions vs ground truth (what RL agent actually placed)
   - Reward breakdown:
     - Object ID correct: +25%
     - Position accurate (within threshold): +20%
     - Depth/distance correct: +15%
     - Rotation correct: +15%
     - Scale correct: +10%
     - Scene completeness (all objects found): +15%
     - False positives (hallucinated objects): -5% each

5. **RL agent learns and improves**
   - PPO algorithm (Stable-Baselines3) updates policy
   - Learns which scene configurations are easier/harder for vision system
   - Gradually constructs more complex, realistic scenes
   - Curriculum learning: start simple, increase difficulty as performance improves

### Why This Matters

- **Self-supervised**: No manual labeling needed - RL agent generates infinite training data
- **Realistic scenes**: Physics + lighting = scenes that look natural, not random placements
- **Stereo vision**: Depth perception like human vision, critical for 3D understanding
- **Scalable**: Can scale to millions of objects (Objaverse XL has 10M+)
- **Transfer learning**: Vision model trained on synthetic scenes can transfer to real-world robotics

**Target hardware**: Windows 11 + NVIDIA RTX 4090 (CUDA), Docker Desktop
**Branch**: `claude/3d-vision-rl-framework-011CUrMSjqP3Uwsfv1psZcbR`

## Architecture

```
src/
├── asset_management/   # Objaverse downloader → SQLite DB
├── rendering/          # Blender Cycles renderer (mono + stereo)
├── stereo/             # Stereo camera config + SGBM disparity
├── physics/            # PyBullet simulator (gravity, collisions)
├── scene/              # Physics-based scene generator
├── vision/             # LLaVA Vision LLM (server + client)
├── rl/                 # Stable-Baselines3 PPO trainer
└── utils/              # Config, database, logging
```

### Data Flow Pipeline

**Phase 1: Asset Preparation**
```
Objaverse API → Download glTF files → Store in /data/assets/raw/
                                    ↓
                              SQLite database (metadata, categories, paths)
```

**Phase 2: RL Training Loop** (happens continuously)
```
RL Agent (PPO)
    ↓ [action: select objects, positions, rotations, scales]
PyBullet Physics Sim
    ↓ [simulate gravity, collisions → final stable positions]
Blender Renderer
    ↓ [6 stereo pairs = 12 images per scene]
Stereo Processor (SGBM)
    ↓ [disparity maps → depth images]
Vision LLM (LLaVA)
    ↓ [object detection + 3D localization]
Reward Calculation
    ↓ [compare predictions vs ground truth]
RL Agent Update
    ↓ [PPO policy gradient update]
[loop repeats]
```

**Phase 3: Evaluation**
- Render test scenes (held-out objects/configurations)
- Measure vision accuracy on increasingly complex scenes
- Export trained vision model for real-world deployment

### Key Technical Components

**3D Rendering (Blender Cycles)**
- GPU ray tracing for photorealistic lighting
- Stereo cameras: 6.5cm baseline, parallel configuration (NOT toed-in)
- Textured ground plane (procedural noise) for stereo matching features
- Objects placed at z=1m (half their height), cameras tilted 20° down to see floor
- 512x512 resolution (configurable), 128 samples/pixel

**Stereo Vision (OpenCV SGBM)**
- Semi-Global Block Matching algorithm
- Input: left + right rectified images
- Output: disparity map (pixel shift between L/R), depth map (meters)
- Formula: `depth = (focal_length × baseline) / disparity`
- Critical for 3D understanding: distinguishes small-close vs large-far objects

**Physics Simulation (PyBullet)**
- Realistic gravity (9.81 m/s²)
- Collision detection between objects and ground
- 90% probability objects land upright (as specified)
- Simulates until stable (velocity < threshold or max time)
- Provides final positions/orientations to renderer

**Vision LLM (LLaVA-1.6)**
- Multimodal transformer: vision encoder + language model
- Input: stereo image pair (or single RGB + depth map)
- Prompt: "List all objects in this 3D scene with their positions (x,y,z), orientations, and scales"
- Output: Structured JSON with per-object predictions
- Runs on RTX 4090 (~14GB VRAM for 7B model)

**RL Agent (PPO via Stable-Baselines3)**
- Observation space: scene complexity budget (how many objects to place)
- Action space: 
  - Discrete: which object_id from database (|A| = num_objects)
  - Continuous: (x, y) position, rotation angle, scale factor
  - Repeat N times per episode (N = scene complexity)
- Reward: weighted sum of accuracy metrics (see "Core Loop" above)
- Policy network: MLP with shared feature extractor
- Value network: estimates expected future reward

## Current Status (2026-05-29)

### ✅ Working
- Asset download from Objaverse 1.0 (18 assets downloaded, DB persistence verified)
- Blender stereo rendering with GPU (Cycles on RTX 4090)
- Object centering/scaling (fixed - was offsetting incorrectly)
- glTF multi-object handling (joins mesh parts, removes cameras/empties)
- Default material fallback for assets missing materials
- Parallel stereo cameras (fixed from toed-in convergent)
- Textured ground plane for stereo matching features

### 🚧 In Progress: Stereo Disparity
Just added textured ground plane. **Next test**:
```powershell
git pull origin claude/3d-vision-rl-framework-011CUrMSjqP3Uwsfv1psZcbR
docker-compose run --rm renderer --mode assets --object-id fc1339e225b7408caec82681be2746c5 --stereo
docker-compose run --rm tests python3 -m src.stereo.test_disparity
```

Expected: disparity map should follow object shape, depth values ~3-5m for object.

Previous failure: matched range 0-4.56 pixels, depth 10-100m (way off). Root cause: black background had no features for SGBM.

### ⏳ Not Started
- Vision LLM (LLaVA) integration - service exists but not tested
- Physics-based multi-object scene generation
- RL training loop
- Format compatibility verification (Blender → SGBM → LLaVA)

## Critical Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | All services. Asset downloader uses `python3 -m`, renderer uses `blender --background` |
| `src/rendering/renderer.py` | Blender renderer. `render_object_stereo_views()` for batch stereo |
| `src/stereo/stereo_camera.py` | StereoCamera class. **Parallel** cameras (not toed-in) |
| `src/stereo/disparity.py` | OpenCV SGBM. `process_stereo_pair()` returns disparity+depth |
| `src/stereo/test_disparity.py` | Quick test script (run via tests container) |
| `src/utils/database.py` | SQLite: objects, renders, annotations, scenes tables |
| `config/config.yaml` | All settings (render res, baseline, etc.) |

## Known Quirks / Gotchas

1. **Blender Python ≠ system Python** - cv2 NOT available in Blender. `src/stereo/__init__.py` lazy-imports cv2-dependent modules.
2. **Asset paths**: DB stores paths relative to `/data/assets/` (NOT `/data/`). Use `self.assets_raw_dir.parent / path`.
3. **glTF import**: Often imports multiple objects (cameras, empties, mesh parts). Filter to MESH only and join before centering.
4. **Object centering**: Don't set `obj.location` and `obj.scale` separately - apply transforms directly to vertices.
5. **Stereo cameras MUST be parallel** for SGBM. Offset along camera's local right axis, both look in same forward direction (use far look_at).
6. **Empty `--remove-orphans` warning** in docker-compose is harmless.
7. **Windows Docker Desktop**: Use direct bind mounts (`./data:/data`), NOT named volumes.

## Testing Commands

```powershell
# Pull latest
git pull origin claude/3d-vision-rl-framework-011CUrMSjqP3Uwsfv1psZcbR

# Download N assets
docker-compose run --rm -e ASSET_COUNT=10 asset_downloader

# Query database
docker-compose run --rm tests python3 -c "from src.utils.database import Database; db = Database('/data/database/assets.db'); objs = db.get_all_objects(); print(f'Objects: {len(objs)}'); [print(' ', o['id'], ':', o['file_path']) for o in objs[:3]]"

# Test render (sanity check Blender)
docker-compose run --rm renderer --test

# Render single asset stereo (6 pairs = 12 images)
docker-compose run --rm renderer --mode assets --object-id <ID> --stereo

# Render all assets stereo
docker-compose run --rm renderer --mode assets --stereo

# Test disparity
docker-compose run --rm tests python3 -m src.stereo.test_disparity

# Run unit tests
docker-compose run --rm tests
```

## Git Workflow

- Branch: `claude/3d-vision-rl-framework-011CUrMSjqP3Uwsfv1psZcbR`
- Push: `git push origin claude/3d-vision-rl-framework-011CUrMSjqP3Uwsfv1psZcbR`
- If push rejected: `git pull --rebase` then push again
- Branch name MUST start with `claude/` and end with session id

## Configuration

`config/config.yaml` has all tunable params:
- `rendering.resolution: 512` (square)
- `rendering.samples: 128` (Cycles)
- `stereo.camera.baseline: 0.065` (6.5cm human IPD)
- `stereo.disparity.num_disparities: 160`
- `physics.object_placement.upright_probability: 0.9`
- `rl.learning_rate: 0.0003` (PPO)
- `rl.total_timesteps: 1000000`
- `rl.n_envs: 4` (parallel environments)

## Design Decisions & Challenges

### Why Stereo Vision?
- Monocular depth estimation requires learned priors (works on natural images, fails on novel synthetic scenes)
- Stereo provides geometric depth from first principles (triangulation)
- Human-like: our visual system uses stereo for depth perception
- Challenge: SGBM needs textured backgrounds (hence the noisy ground plane)

### Why Physics Simulation?
- Random placement looks unnatural (floating objects, intersections)
- Physics ensures plausible scenes (objects rest on ground, no overlaps)
- 90% upright: most real-world objects sit upright on flat surfaces
- Adds realism for vision transfer to real world

### Why LLaVA instead of CLIP/Object Detection?
- Need 3D localization (x,y,z), not just 2D bounding boxes
- LLaVA can parse depth maps and reason about 3D structure
- Language output is flexible (can describe occlusions, orientations)
- Extensible to more complex queries ("which object is on top?")

### Why PPO instead of DQN/A3C?
- Hybrid action space (discrete object selection + continuous placement)
- PPO handles continuous actions naturally
- Stable training (clipped objective prevents destructive updates)
- Sample efficient with parallel environments

### Expected Challenges

1. **Sparse rewards**: Only get reward at end of episode (after rendering+vision)
   - Solution: Dense shaping (reward partial progress, e.g., object placed without collision)

2. **Exploration**: Huge action space (1000 objects × continuous positions)
   - Solution: Curriculum learning (start with 10 objects), epsilon-greedy for object selection

3. **Vision errors**: LLaVA might hallucinate or miss objects
   - Solution: Multiple viewpoints (6 angles), majority voting or confidence thresholding

4. **Computational cost**: Rendering is slow (~1 sec/scene)
   - Solution: Batch rendering, async rendering pipeline, lower samples during training

5. **Distribution shift**: Agent might learn to exploit vision weaknesses
   - Solution: Regularize for scene diversity, adversarial training (vision improves with agent)

### Performance Expectations

- **Asset download**: ~1 min for 100 objects (Objaverse API)
- **Rendering**: ~1 sec per stereo pair (RTX 4090, 128 samples)
- **SGBM disparity**: ~50ms per stereo pair (CPU)
- **Vision inference**: ~500ms per image (LLaVA-1.6-7B on RTX 4090)
- **Physics simulation**: ~100ms per scene (2-5 objects)
- **RL training**: ~10k scenes to see initial learning, 100k for good policy

**Throughput**: ~0.5-1 scenes/sec training (bottleneck: rendering+vision)
**Parallelization**: Run 4-8 environments simultaneously on RTX 4090

## Next Steps (in order)

### Immediate Tasks (Current Sprint)

1. **✅ CURRENT: Verify stereo disparity works**
   - Just added textured ground plane with procedural noise
   - Run test, check `data/assets/rendered/<cat>/<id>/stereo_test/disparity.png`
   - Object should be visible/bright in disparity map
   - Depth values should be ~3-5m for object pixels
   - Ground should show depth gradient (near=bright, far=dark)

2. **Batch render all 18 assets stereo**
   - `docker-compose run --rm renderer --mode assets --stereo`
   - Verify all 216 images created (18 × 6 × 2)
   - Spot-check: objects on textured floor, proper lighting

3. **Test Vision LLM (LLaVA) annotation on single objects**
   - Start vision_model service: `docker-compose up -d vision_model`
   - Test API: send single rendered image, verify LLaVA responds
   - Prompt engineering: tune prompt for object identification task
   - Expected output format: object name/category from vision (may differ from metadata)

### Integration Phase (Week 1-2)

4. **Connect stereo → vision pipeline**
   - Create annotator that sends (RGB image + depth map) to LLaVA
   - Or: send stereo pair directly, let LLaVA learn depth perception
   - Store annotations in database (link to render_id)
   - Test: does LLaVA correctly identify objects in different orientations/angles?

5. **Implement physics-based scene generation**
   - `src/scene/physics_scene_generator.py` exists but untested
   - Test with 2 objects: place randomly, simulate physics, render result
   - Verify 90% upright probability works
   - Check collision detection (objects don't overlap)
   - Generate ground truth: actual positions after physics settle

6. **Build scene → vision → ground truth comparison**
   - Generate scene with known object IDs and positions (ground truth)
   - Render stereo pairs
   - Run vision LLM inference
   - Compare predictions to ground truth
   - Compute accuracy metrics (object ID, position error, etc.)

### RL Training Phase (Week 3-4)

7. **Implement reward function**
   - Input: (ground_truth, vision_predictions)
   - Output: scalar reward in [0, 1]
   - Weighted components:
     - Object identification: 25%
     - Position accuracy (L2 distance): 20%
     - Depth accuracy: 15%
     - Rotation accuracy (quaternion distance): 15%
     - Scale accuracy: 10%
     - Completeness (all objects found): 15%
     - False positive penalty: -5% per hallucinated object

8. **Implement RL environment (OpenAI Gym interface)**
   - State: current scene state (what's been placed)
   - Action: (object_id, x, y, rotation, scale) - hybrid discrete/continuous
   - Step function:
     1. Add object to PyBullet scene
     2. Simulate physics → get final position
     3. Check if scene complete (N objects placed)
     4. If complete: render → vision → compute reward
   - Reset: clear scene, sample new object set

9. **Train initial PPO agent**
   - Start simple: 2 objects per scene, 512x512 images
   - Train for 100k timesteps (~10k scenes)
   - Monitor metrics:
     - Average reward over time (should increase)
     - Vision accuracy on agent-generated scenes
     - Scene diversity (don't just learn one easy config)
   - Checkpoint best model

10. **Curriculum learning & scaling**
    - Increase scene complexity: 2→3→4→5 objects
    - Add rotation/scale variation
    - Download more assets (100→1000→10k)
    - Increase image resolution (512→768→1024)
    - Train vision model on agent-generated scenes (domain adaptation)

### Production & Evaluation (Week 5+)

11. **Benchmark & analysis**
    - Hold out 20% of objects for testing
    - Generate test scenes (random vs agent-generated)
    - Measure vision accuracy on both
    - Qualitative analysis: what scenes does agent learn to construct?
    - Compare to baselines (random placement, rule-based)

12. **Real-world transfer (optional)**
    - Test vision model on real photos
    - Domain adaptation techniques if needed
    - Deploy to robotics platform (if available)

## Communication Style

User prefers:
- Concise responses
- One step at a time, wait for feedback before next
- Show commands ready to paste into PowerShell
- Explain "why" when fixes are non-obvious
- Don't dump huge code blocks - use targeted edits
