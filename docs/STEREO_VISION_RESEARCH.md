# Stereoscopic Vision Integration - Research & Implementation Plan

## Stereoscopic Vision Libraries for 3D Scene Analysis

### 1. **OpenCV Stereo Vision** (Recommended for Classic CV)
- **Library**: opencv-contrib-python
- **Capabilities**:
  - Stereo calibration
  - Disparity map computation (SGBM, BM algorithms)
  - Depth estimation from stereo pairs
  - 3D reconstruction
- **Format**: Left/Right image pairs (PNG/JPG)
- **Pros**: Well-established, fast, extensive documentation
- **Cons**: Traditional CV, not deep learning based

### 2. **PyTorch3D Stereo** (Recommended for Deep Learning)
- **Library**: pytorch3d
- **Capabilities**:
  - Stereo depth estimation
  - 3D-aware neural networks
  - Differentiable rendering with stereo support
  - Point cloud generation from stereo
- **Format**: Tensor pairs (left, right) with camera parameters
- **Pros**: Deep learning ready, integrates with PyTorch ecosystem
- **Cons**: Heavier weight, GPU required

### 3. **MiDaS Stereo Depth** (State-of-the-art)
- **Library**: transformers (HuggingFace)
- **Model**: DPT-Large or MiDaS v3.1
- **Capabilities**:
  - Monocular depth estimation (can be adapted for stereo)
  - Zero-shot generalization
  - Relative depth maps
- **Format**: Single or stereo images → depth maps
- **Pros**: State-of-the-art depth estimation
- **Cons**: Primarily monocular (but can fuse stereo)

### 4. **Stereo-Vision-Transformer (SVT)** (Cutting Edge)
- **Library**: Custom implementation / timm
- **Models**: 
  - RAFT-Stereo (ECCV 2022)
  - CREStereo (CVPR 2022)
  - Unimatch (CVPR 2023)
- **Capabilities**:
  - End-to-end stereo matching
  - Dense disparity estimation
  - Occlusion handling
- **Format**: Stereo pairs → disparity maps
- **Pros**: SOTA performance, handles occlusions well
- **Cons**: Requires training or pre-trained weights

### 5. **LLaVA-Stereo / Multimodal Stereo** (For This Project)
- **Approach**: Extend LLaVA to handle stereo inputs
- **Options**:
  a. **Dual-Stream**: Process L/R separately, fuse features
  b. **Disparity-Augmented**: Compute disparity, concat with RGB
  c. **3D Point Cloud**: Convert stereo → 3D points → feed to model
- **Format**: 
  - Input: Stereo image pairs (L/R) + camera parameters
  - Output: Scene description with depth/3D info
- **Implementation**: Fine-tune LLaVA with stereo training data

## Recommended Architecture for This Project

### Phase 1: Rendering (Blender)
```
Blender Scene
    ↓
Stereo Camera Setup (L/R eye positions)
    ↓
Render Two Views (512x512 each)
    ↓
Save Format:
  - scene_001_left.jpg
  - scene_001_right.jpg
  - scene_001_cameras.json (baseline, focal length, etc.)
```

### Phase 2: Stereo Processing Pipeline

**Option A: Traditional CV (Fast, Interpretable)**
```python
import cv2
# Compute disparity
stereo = cv2.StereoSGBM_create(...)
disparity = stereo.compute(img_left, img_right)

# Convert to depth
depth = (focal_length * baseline) / disparity

# Feed to Vision LLM
vision_input = {
    "rgb": img_left,
    "depth": depth,
    "stereo_pair": (img_left, img_right)
}
```

**Option B: Deep Learning (Better for RL)**
```python
import torch
from pytorch3d.structures import Pointclouds

# Compute 3D point cloud from stereo
points_3d = stereo_to_pointcloud(img_left, img_right, cameras)

# Multi-modal input to vision model
features = vision_encoder(img_left, img_right, points_3d)

# Scene understanding
scene_desc = vision_llm(features)
```

### Phase 3: RL Integration

**Updated Observation Space**:
```python
observation = {
    "stereo_left": (3, 512, 512),      # RGB left
    "stereo_right": (3, 512, 512),     # RGB right  
    "disparity": (1, 512, 512),        # Depth/disparity
    "camera_params": (6,),              # Baseline, focal, etc.
    "scene_encoding": (512,)            # Vision LLM output
}
```

**Updated Reward Function**:
```python
reward += depth_accuracy_weight * depth_reconstruction_score
reward += 3d_position_weight * 3d_position_accuracy
```

## Implementation Recommendation

### Best Approach: **Hybrid System**

1. **Rendering**: Blender stereo pairs (baseline ~6.5cm, human IPD)
2. **Stereo Processing**: RAFT-Stereo or OpenCV SGBM for disparity
3. **Vision Understanding**: 
   - LLaVA for RGB semantic understanding
   - Disparity map for 3D spatial reasoning
   - Fuse both for complete scene understanding
4. **RL Training**: Multi-modal observations (RGB + depth)

### Format Specification

**Stereo Image Format**:
```json
{
  "scene_id": "uuid",
  "stereo_pair": {
    "left": "path/to/left.jpg",
    "right": "path/to/right.jpg",
    "baseline": 0.065,  // meters (human IPD)
    "focal_length": 50,  // mm
    "resolution": [512, 512],
    "format": "JPEG"
  },
  "camera_left": {
    "position": [x, y, z],
    "rotation": [rx, ry, rz],
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  },
  "camera_right": {
    "position": [x + baseline, y, z],
    "rotation": [rx, ry, rz],
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  }
}
```

**Disparity Map Format**:
```
- Format: NumPy array or 16-bit PNG
- Shape: (H, W)
- Values: Disparity in pixels (float or uint16)
- Conversion to depth: depth = (focal_length * baseline) / disparity
```

## Libraries to Add to requirements

```txt
# Stereo vision
opencv-contrib-python==4.9.0.80
pytorch3d==0.7.5
# Optional: RAFT-Stereo
timm==0.9.12
einops==0.7.0
```

## Next Steps

1. ✅ Create stereo camera configuration
2. ✅ Update Blender renderer for stereo pairs
3. ✅ Add disparity computation module
4. ✅ Extend Vision LLM for stereo input
5. ✅ Update RL environment for multi-modal observations
6. ✅ Modify reward function for 3D accuracy
