"""
Physics-Based Scene Generator

Generates 3D scenes using physics simulation for realistic object placement.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
import trimesh

from src.utils.config import load_config
from src.utils.database import Database
from src.utils.logger import get_logger
from src.asset_management.asset_manager import AssetManager
from src.physics.simulator import PhysicsSimulator


logger = get_logger("physics_scene_generator")


class PhysicsSceneGenerator:
    """Generate 3D scenes with physics-based object placement."""

    def __init__(self, config=None, db=None, asset_manager=None, physics_sim=None):
        """
        Initialize physics scene generator.

        Args:
            config: Configuration object
            db: Database instance
            asset_manager: AssetManager instance
            physics_sim: PhysicsSimulator instance
        """
        self.config = config or load_config()
        self.db = db or Database(self.config.get("database.path", "/data/database/assets.db"))
        self.asset_manager = asset_manager or AssetManager(self.db)
        self.physics_sim = physics_sim or PhysicsSimulator(self.config)

        # Get scene settings
        self.complexity_levels = self.config.get("scene.complexity_levels", {})

        # Paths
        self.scenes_dir = Path(self.config.get("paths.scenes", "/data/scenes"))
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        # Physics settings
        self.use_physics = self.config.get("physics.enabled", True)
        self.upright_probability = self.config.get("physics.object_placement.upright_probability", 0.9)

        logger.info("Physics scene generator initialized")

    def generate_scene_with_physics(
        self,
        complexity_level: int = 1,
        scene_id: Optional[str] = None,
    ) -> Dict:
        """
        Generate a scene with physics simulation.

        Args:
            complexity_level: Complexity level (1-5)
            scene_id: Optional scene ID

        Returns:
            Scene configuration with physics-stable positions
        """
        # Get complexity settings
        complexity_config = self.complexity_levels.get(
            complexity_level,
            {"objects": [1, 3], "ground": "flat", "occlusion": 0.0}
        )

        # Determine number of objects
        min_objs, max_objs = complexity_config["objects"]
        num_objects = random.randint(min_objs, max_objs)

        # Select random objects
        assets = self.asset_manager.get_random_assets(num_objects)

        logger.info(f"Generating scene with {len(assets)} objects using physics")

        # Start physics simulation
        self.physics_sim.start()

        try:
            # Create ground plane (always at z=0)
            # Ground is implicit in PyBullet (created in simulator)

            # Load objects into physics simulation
            body_ids = []
            asset_info = []

            for i, asset in enumerate(assets):
                # Get asset file path
                file_path = Path("/data") / asset["file_path"]

                # Load mesh to get approximate radius
                try:
                    mesh = trimesh.load(str(file_path))
                    bounds = mesh.bounds
                    radius = max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]) / 2
                except:
                    radius = 0.5  # Default radius

                # Generate random 2D position (checking for collisions)
                max_attempts = 10
                for attempt in range(max_attempts):
                    x = random.uniform(-3, 3)
                    y = random.uniform(-3, 3)

                    # Check collision with existing objects
                    existing_positions = [(info["position"][0], info["position"][1], 0)
                                        for info in asset_info]

                    if not self.physics_sim.check_collision((x, y), radius, existing_positions):
                        break
                else:
                    logger.warning(f"Could not find non-colliding position for object {i}")
                    continue

                # Determine if upright
                upright = random.random() < self.upright_probability

                # Load object into physics simulation
                body_id = self.physics_sim.load_object(
                    mesh_path=str(file_path),
                    position=(x, y),
                    upright=upright,
                    mass=1.0
                )

                if body_id is not None:
                    body_ids.append(body_id)
                    asset_info.append({
                        "asset": asset,
                        "body_id": body_id,
                        "initial_position": (x, y),
                        "radius": radius
                    })

            # Simulate until stable
            logger.info("Simulating physics...")
            final_states = self.physics_sim.simulate_until_stable(body_ids)

            # Build scene configuration from physics results
            scene_config = {
                "complexity_level": complexity_level,
                "scene_id": scene_id,
                "objects": [],
                "environment": {
                    "ground_type": "flat",
                    "ground_position": [0, 0, 0],
                    "ground_size": 10.0,
                    "lighting": self._get_lighting_config(complexity_level),
                    "camera": self._get_camera_config(),
                },
                "physics_simulated": True,
            }

            # Add objects with physics-computed positions
            for info in asset_info:
                body_id = info["body_id"]
                state = final_states[body_id]

                obj_config = {
                    "object_id": info["asset"]["id"],
                    "position": state["position"],
                    "orientation": state["orientation"],  # Quaternion (x, y, z, w)
                    "scale": [1.0, 1.0, 1.0],
                    "is_stable": state["is_stable"],
                    "initial_position": info["initial_position"],
                }

                scene_config["objects"].append(obj_config)

            logger.info(f"Scene generated with {len(scene_config['objects'])} stable objects")

            return scene_config

        finally:
            # Clean up physics simulation
            self.physics_sim.stop()

    def _get_lighting_config(self, complexity_level: int) -> Dict:
        """Get lighting configuration for complexity level."""
        presets = ["default", "bright", "warm", "cool", "sunset"]
        preset = presets[min(complexity_level - 1, len(presets) - 1)]
        return {"preset": preset}

    def _get_camera_config(self) -> Dict:
        """Get camera configuration."""
        return {
            "distance": 8,
            "azimuth": random.uniform(0, 360),
            "elevation": random.uniform(20, 60),
            "baseline": 0.065,  # Stereo baseline
        }

    def generate_scenes_batch(
        self,
        count: int = 100,
        complexity_level: int = 1,
    ) -> List[Dict]:
        """
        Generate multiple scenes with physics.

        Args:
            count: Number of scenes to generate
            complexity_level: Complexity level

        Returns:
            List of scene configurations
        """
        logger.info(f"Generating {count} physics scenes at complexity {complexity_level}")

        scenes = []
        for i in range(count):
            scene_id = f"physics_scene_{complexity_level}_{i:04d}"
            scene = self.generate_scene_with_physics(
                complexity_level=complexity_level,
                scene_id=scene_id
            )
            scenes.append(scene)

            # Store in database
            self.db.add_scene(
                complexity_level=complexity_level,
                objects=scene["objects"],
                environment=scene["environment"],
                image_path="",  # Will be set after rendering
                ground_truth=scene
            )

        logger.info(f"Generated {len(scenes)} physics scenes")
        return scenes


def main():
    """Command-line interface for physics scene generator."""
    parser = argparse.ArgumentParser(description="Generate 3D scenes with physics")
    parser.add_argument(
        "--complexity",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Scene complexity level",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of scenes to generate",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render the generated scenes",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Create generator
    generator = PhysicsSceneGenerator(config=config)

    # Generate scenes
    scenes = generator.generate_scenes_batch(
        count=args.count,
        complexity_level=args.complexity,
    )

    logger.info(f"Scene generation complete: {len(scenes)} scenes")

    if args.render:
        logger.info("Rendering scenes (TODO: implement scene rendering)")
        # TODO: Integrate with BlenderRenderer to render the scenes


if __name__ == "__main__":
    main()
