# Physics Integration for Realistic Object Placement

## Physics Library Selection

### **Bullet Physics via PyBullet** (Recommended)

**Why PyBullet?**
- ✅ Native Python bindings
- ✅ Integrates with Blender (Blender uses Bullet internally)
- ✅ Fast, stable, well-tested
- ✅ Supports rigid body dynamics, collision detection
- ✅ GPU acceleration available
- ✅ Open source (Zlib license)
- ✅ Used in robotics/RL research (compatible with Gym)

**Alternatives Considered**:
- **Blender Bullet**: Built-in but harder to control programmatically
- **MuJoCo**: Excellent but complex licensing
- **PhysX**: Great but primarily C++, less Python support
- **ODE**: Older, less maintained

## Implementation Approach

### 1. Two-Stage Process

**Stage 1: Physics Simulation (PyBullet)**
```python
import pybullet as p

# Setup physics world
p.connect(p.DIRECT)  # No GUI
p.setGravity(0, 0, -9.81)

# Create ground plane
ground = p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, ground)

# Load 3D object, simulate drop
collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName="object.obj")
visual_shape = p.createVisualShape(p.GEOM_MESH, fileName="object.obj")

# 90% upright, 10% random orientation
if random.random() < 0.9:
    orientation = [0, 0, 0, 1]  # Upright quaternion
else:
    orientation = random_quaternion()

# Drop from above ground
initial_pos = [x, y, 2.0]  # 2m above ground
body = p.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=collision_shape,
    baseVisualShapeIndex=visual_shape,
    basePosition=initial_pos,
    baseOrientation=orientation
)

# Simulate until stable
for _ in range(240):  # ~10 seconds at 240Hz
    p.stepSimulation()

# Get final stable position/orientation
final_pos, final_orn = p.getBasePositionAndOrientation(body)
```

**Stage 2: Render in Blender**
```python
# Use final physics position/orientation from PyBullet
blender_import_object(object_path)
blender_object.location = final_pos
blender_object.rotation_quaternion = final_orn
blender_render()
```

### 2. Workflow

```
1. Generate Scene Layout (SceneGenerator)
   ↓
2. Physics Simulation (PyBullet)
   - Create flat ground plane
   - Drop objects from above
   - 90% upright initial orientation
   - Simulate until stable
   - Check collisions, prevent overlap
   ↓
3. Export Stable Positions
   - Final (x, y, z) positions
   - Final quaternion orientations
   - On-ground confirmation
   ↓
4. Render in Blender (BlenderRenderer)
   - Import objects at physics-computed positions
   - Render stereo views
   - Save RGB + depth
```

## Floor Specification

### Flat Ground Plane
```python
ground_config = {
    "type": "plane",
    "size": [10, 10],  # 10x10 meters
    "position": [0, 0, 0],  # At z=0
    "material": {
        "color": [0.8, 0.8, 0.8],
        "roughness": 0.7,
        "metallic": 0.0
    },
    "physics": {
        "friction": 0.5,
        "restitution": 0.1  # Low bounce
    }
}
```

## Object Placement Rules

### 1. Upright Orientation (90% of time)
```python
def get_initial_orientation(object_mesh, upright_probability=0.9):
    """
    Determine initial orientation for object.
    
    Args:
        object_mesh: Trimesh object
        upright_probability: Chance of starting upright (default 0.9)
    
    Returns:
        Quaternion orientation
    """
    if random.random() < upright_probability:
        # Compute "up" direction from mesh
        # Usually the +Z axis of the object's bounding box
        up_vector = compute_up_direction(object_mesh)
        orientation = align_to_world_z(up_vector)
    else:
        # Random orientation
        orientation = random_quaternion()
    
    return orientation
```

### 2. Drop Height
```python
drop_height = ground_z + object_bbox_height + 0.5  # 50cm clearance
```

### 3. Collision Detection
```python
# Before adding object, check collisions
def check_collision(new_position, existing_objects):
    for obj in existing_objects:
        if distance(new_position, obj.position) < (obj.radius + new_obj.radius):
            return True  # Collision detected
    return False

# Only add if no collision
if not check_collision(position, placed_objects):
    place_object(position)
```

### 4. Stability Check
```python
def is_stable(body_id, threshold=0.001):
    """
    Check if object has settled.
    
    Args:
        body_id: PyBullet body ID
        threshold: Velocity threshold for "stable"
    
    Returns:
        True if stable, False otherwise
    """
    linear_vel, angular_vel = p.getBaseVelocity(body_id)
    
    linear_speed = np.linalg.norm(linear_vel)
    angular_speed = np.linalg.norm(angular_vel)
    
    return (linear_speed < threshold) and (angular_speed < threshold)

# Simulation loop with early exit
max_steps = 240
for step in range(max_steps):
    p.stepSimulation()
    
    if step > 100 and is_stable(body_id):
        break  # Object has settled
```

## Configuration Updates

### config.yaml additions
```yaml
physics:
  enabled: true
  engine: "pybullet"
  simulation:
    gravity: -9.81
    time_step: 0.004  # 250Hz
    max_simulation_time: 10.0  # seconds
    stability_threshold: 0.001
  
  ground:
    type: "plane"
    size: [10, 10]
    position: [0, 0, 0]
    friction: 0.5
    restitution: 0.1
  
  object_placement:
    upright_probability: 0.9
    drop_height: 0.5  # meters above object top
    min_separation: 0.1  # meters between objects
    max_placement_attempts: 10
```

## Requirements Update

```txt
# Physics
pybullet==3.2.5
trimesh==4.0.10  # For mesh analysis
```

## Benefits of Physics Integration

1. **Realistic Placement**: Objects rest naturally on ground
2. **No Floating Objects**: Physics ensures contact with ground
3. **Natural Orientations**: Objects settle into stable poses
4. **Collision Avoidance**: Automatic separation of objects
5. **Varied Scenes**: Random stable configurations
6. **RL Challenge**: Agent must learn real-world physics constraints
7. **Occlusion**: Natural object stacking/overlapping
8. **Depth Cues**: Objects at different z-heights on uneven placement

## Integration with Stereo Vision

Physics + Stereo provides:
- **True 3D positions**: Physics gives ground truth
- **Depth verification**: Stereo depth should match physics
- **Occlusion realism**: Objects occlude based on physics layout
- **RL reward**: Compare reconstructed 3D positions with physics ground truth

## Next Implementation Steps

1. ✅ Add PyBullet to requirements
2. ✅ Create PhysicsSimulator class
3. ✅ Update SceneGenerator to use physics
4. ✅ Add ground plane to rendering
5. ✅ Update scene metadata to include physics state
6. ✅ Modify RL reward to account for 3D position accuracy
