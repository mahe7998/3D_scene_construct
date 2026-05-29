# CLAUDE.md - Project Context for Claude Code

## Project Overview

Deep Reinforcement Learning system that:
1. Downloads 3D assets from Objaverse
2. Renders them with stereo cameras (Blender Cycles + GPU)
3. Uses a Vision LLM (LLaVA) to identify objects
4. Trains an RL agent (PPO via Stable-Baselines3) to construct/identify scenes
5. Uses PyBullet for physics-based object placement (90% upright, on ground)

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

**Data flow**: Objaverse → `/data/assets/raw/` → Blender render → `/data/assets/rendered/` → SGBM disparity + LLaVA annotation → RL training

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

## Next Steps (in order)

1. **Verify stereo disparity works** (current task)
   - Run test, check `data/assets/rendered/<cat>/<id>/stereo_test/disparity.png`
   - Object should be visible/bright in disparity map
   - Depth values should be ~3-5m for object pixels
   - If still failing: try tuning SGBM params or use a different test object with more texture

2. **Batch render all 18 assets stereo**
   - `docker-compose run --rm renderer --mode assets --stereo`
   - Verify all 216 images created (18 × 6 × 2)

3. **Test Vision LLM (LLaVA) annotation**
   - Service: `vision_model` (docker-compose), then `annotator`
   - LLaVA-1.6 needs ~14GB VRAM on RTX 4090 (24GB - should fit)
   - Verify it produces sensible descriptions of rendered objects

4. **Physics-based scene generation**
   - `src/scene/physics_scene_generator.py` exists, untested
   - Should place 2-5 objects on ground with PyBullet physics
   - 90% upright probability

5. **RL training**
   - `src/rl/trainer.py` - PPO via Stable-Baselines3
   - Action: choose where/how to place objects
   - Reward: Vision LLM identification accuracy
   - Start with small scenes (1-2 objects, 512x512)

## Communication Style

User prefers:
- Concise responses
- One step at a time, wait for feedback before next
- Show commands ready to paste into PowerShell
- Explain "why" when fixes are non-obvious
- Don't dump huge code blocks - use targeted edits
