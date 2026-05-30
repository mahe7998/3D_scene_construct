"""
Scene builder (placement only, no rendering).

Produces scene *specs* (object placements + camera) that the Blender renderer
consumes via BlenderRenderer.render_scene_stereo. Pure Python (no bpy), so it can
run in any container.

Curriculum level 1: a few UPRIGHT objects on flat ground, well separated (no full
occlusion by construction), random yaw. Deterministic given (num_objects, index).
"""

import math
import random
from typing import Dict, List, Optional

from src.utils.database import Database


def _ring_positions(n: int, radius: float, phase: float) -> List[List[float]]:
    """Evenly spaced (x, y) points on a circle - guarantees object separation."""
    if n == 1:
        return [[0.0, 0.0]]
    return [
        [radius * math.cos(phase + 2 * math.pi * i / n),
         radius * math.sin(phase + 2 * math.pi * i / n)]
        for i in range(n)
    ]


def build_demo_scene(
    db: Database,
    num_objects: int,
    index: int = 0,
    target_size: float = 1.0,
    ground_size: float = 20.0,
) -> Optional[Dict]:
    """
    Build one deterministic level-1 scene spec.

    Args:
        db: asset database
        num_objects: number of objects to place
        index: scene index (seeds RNG -> reproducible, distinct scenes)
        target_size: normalized longest dimension of each object (meters)
        ground_size: textured ground plane size (meters)

    Returns:
        Scene spec dict for BlenderRenderer.render_scene_stereo, or None if the
        asset DB is empty.
    """
    assets = db.get_all_objects()
    if not assets:
        return None

    rng = random.Random(index)  # deterministic per scene index

    chosen = rng.sample(assets, num_objects) if num_objects <= len(assets) \
        else [rng.choice(assets) for _ in range(num_objects)]

    # Separation: ring radius scales with object size and count so footprints
    # never overlap (each needs ~target_size of room).
    radius = 0.0 if num_objects == 1 else max(1.5, target_size * num_objects * 0.6)
    positions = _ring_positions(num_objects, radius, phase=rng.uniform(0, 2 * math.pi))

    objects = []
    for asset, (x, y) in zip(chosen, positions):
        objects.append({
            "object_id": asset["id"],
            "category": asset.get("category"),
            "file_path": asset["file_path"],
            "ground_xy": [round(x, 4), round(y, 4)],
            "yaw_deg": round(rng.uniform(0, 360), 2),
            "target_size": target_size,
        })

    # Camera: far enough to frame the whole ring, tilted down to see the ground.
    distance = max(4.0, radius + 3.5 * target_size)
    camera = {
        "distance": round(distance, 3),
        "azimuth": round(rng.uniform(0, 360), 2),
        "elevation": round(rng.uniform(22, 38), 2),
        "baseline": 0.065,
        "look_at": [0.0, 0.0, target_size / 2.0],
    }

    return {
        "scene_id": f"demo_{num_objects}obj_{index:04d}",
        "complexity_level": 1,
        "ground": {"size": ground_size},
        "lighting": {"preset": "default"},
        "camera": camera,
        "objects": objects,
    }
