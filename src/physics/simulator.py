"""
Physics Simulator

Uses PyBullet to simulate realistic object placement with gravity and collisions.
"""

import numpy as np
import pybullet as p
import pybullet_data
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("physics_simulator")


class PhysicsSimulator:
    """Simulate physics for realistic object placement."""

    def __init__(self, config=None):
        """
        Initialize physics simulator.

        Args:
            config: Configuration object
        """
        self.config = config or load_config()

        # Physics settings
        self.gravity = self.config.get("physics.simulation.gravity", -9.81)
        self.time_step = self.config.get("physics.simulation.time_step", 0.004)
        self.max_sim_time = self.config.get("physics.simulation.max_simulation_time", 10.0)
        self.stability_threshold = self.config.get("physics.simulation.stability_threshold", 0.001)

        # Ground settings
        self.ground_friction = self.config.get("physics.ground.friction", 0.5)
        self.ground_restitution = self.config.get("physics.ground.restitution", 0.1)

        # Placement settings
        self.upright_probability = self.config.get("physics.object_placement.upright_probability", 0.9)
        self.drop_height = self.config.get("physics.object_placement.drop_height", 0.5)
        self.min_separation = self.config.get("physics.object_placement.min_separation", 0.1)

        # PyBullet connection
        self.physics_client = None
        self.ground_id = None
        self.object_bodies = {}

        logger.info("Physics simulator initialized")

    def start(self):
        """Start physics simulation (headless mode)."""
        if self.physics_client is not None:
            self.stop()

        self.physics_client = p.connect(p.DIRECT)  # Headless
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.time_step)

        # Create ground plane
        self._create_ground()

        logger.info("Physics simulation started")

    def stop(self):
        """Stop physics simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            self.object_bodies = {}
            logger.info("Physics simulation stopped")

    def _create_ground(self):
        """Create ground plane."""
        ground_shape = p.createCollisionShape(p.GEOM_PLANE)
        self.ground_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=ground_shape,
            basePosition=[0, 0, 0]
        )

        # Set friction and restitution
        p.changeDynamics(
            self.ground_id,
            -1,
            lateralFriction=self.ground_friction,
            restitution=self.ground_restitution
        )

        logger.debug("Ground plane created")

    def load_object(
        self,
        mesh_path: str,
        position: Tuple[float, float, float],
        upright: bool = None,
        mass: float = 1.0,
    ) -> int:
        """
        Load object into physics simulation.

        Args:
            mesh_path: Path to 3D mesh file
            position: Initial (x, y) position (z will be computed)
            upright: Whether to start upright (None = use probability)
            mass: Object mass in kg

        Returns:
            PyBullet body ID
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            logger.error(f"Mesh file not found: {mesh_path}")
            return None

        # Determine if upright
        if upright is None:
            upright = np.random.random() < self.upright_probability

        # Load mesh with trimesh to get bounds
        try:
            mesh = trimesh.load(str(mesh_path))

            # Get mesh bounds
            bounds = mesh.bounds
            mesh_height = bounds[1][2] - bounds[0][2]  # Z extent

            # Compute initial position (above ground)
            x, y = position[0], position[1]
            z = mesh_height / 2 + self.drop_height  # Drop from above

            # Compute orientation
            if upright:
                # Align mesh up direction with world Z
                orientation = p.getQuaternionFromEuler([0, 0, 0])
            else:
                # Random orientation
                orientation = p.getQuaternionFromEuler([
                    np.random.uniform(0, 2 * np.pi),
                    np.random.uniform(0, 2 * np.pi),
                    np.random.uniform(0, 2 * np.pi)
                ])

            # Create collision and visual shapes
            # PyBullet needs OBJ format, convert if necessary
            if mesh_path.suffix.lower() != '.obj':
                # Export as temporary OBJ
                temp_obj = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
                mesh.export(temp_obj.name, file_type='obj')
                obj_path = temp_obj.name
            else:
                obj_path = str(mesh_path)

            # Create shapes
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=obj_path,
                meshScale=[1, 1, 1]
            )

            visual_shape = p.createVisualShape(
                p.GEOM_MESH,
                fileName=obj_path,
                meshScale=[1, 1, 1]
            )

            # Create body
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x, y, z],
                baseOrientation=orientation
            )

            # Set physics properties
            p.changeDynamics(
                body_id,
                -1,
                lateralFriction=0.7,
                restitution=0.1,
                linearDamping=0.1,
                angularDamping=0.1
            )

            logger.debug(f"Loaded object at ({x:.2f}, {y:.2f}, {z:.2f}), upright={upright}")

            return body_id

        except Exception as e:
            logger.error(f"Error loading mesh {mesh_path}: {e}")
            return None

    def simulate_until_stable(
        self,
        body_ids: List[int],
        max_steps: int = None
    ) -> Dict[int, Dict]:
        """
        Simulate until all objects are stable.

        Args:
            body_ids: List of PyBullet body IDs to track
            max_steps: Maximum simulation steps (None = use config)

        Returns:
            Dictionary mapping body_id to final state
        """
        if max_steps is None:
            max_steps = int(self.max_sim_time / self.time_step)

        stable_objects = set()
        consecutive_stable = {body_id: 0 for body_id in body_ids}

        for step in range(max_steps):
            p.stepSimulation()

            # Check stability
            for body_id in body_ids:
                if body_id in stable_objects:
                    continue

                if self._is_stable(body_id):
                    consecutive_stable[body_id] += 1
                    if consecutive_stable[body_id] >= 10:  # Stable for 10 steps
                        stable_objects.add(body_id)
                        logger.debug(f"Object {body_id} stabilized at step {step}")
                else:
                    consecutive_stable[body_id] = 0

            # Early exit if all stable
            if len(stable_objects) == len(body_ids):
                logger.info(f"All objects stable after {step} steps")
                break

        # Get final states
        final_states = {}
        for body_id in body_ids:
            pos, orn = p.getBasePositionAndOrientation(body_id)
            lin_vel, ang_vel = p.getBaseVelocity(body_id)

            final_states[body_id] = {
                "position": list(pos),
                "orientation": list(orn),  # Quaternion [x, y, z, w]
                "linear_velocity": list(lin_vel),
                "angular_velocity": list(ang_vel),
                "is_stable": body_id in stable_objects
            }

        return final_states

    def _is_stable(self, body_id: int) -> bool:
        """
        Check if object is stable (not moving).

        Args:
            body_id: PyBullet body ID

        Returns:
            True if stable
        """
        linear_vel, angular_vel = p.getBaseVelocity(body_id)

        linear_speed = np.linalg.norm(linear_vel)
        angular_speed = np.linalg.norm(angular_vel)

        return (linear_speed < self.stability_threshold and
                angular_speed < self.stability_threshold)

    def check_collision(
        self,
        position: Tuple[float, float],
        radius: float,
        existing_positions: List[Tuple[float, float, float]]
    ) -> bool:
        """
        Check if position would collide with existing objects.

        Args:
            position: (x, y) position to check
            radius: Object radius
            existing_positions: List of existing (x, y, z) positions

        Returns:
            True if collision detected
        """
        x, y = position

        for ex, ey, ez in existing_positions:
            distance = np.sqrt((x - ex)**2 + (y - ey)**2)
            if distance < (radius * 2 + self.min_separation):
                return True

        return False

    def get_ground_contact_points(self, body_id: int) -> List[Dict]:
        """
        Get contact points between object and ground.

        Args:
            body_id: PyBullet body ID

        Returns:
            List of contact point dictionaries
        """
        contact_points = p.getContactPoints(bodyA=body_id, bodyB=self.ground_id)

        contacts = []
        for cp in contact_points:
            contacts.append({
                "position": cp[5],  # Contact position on A
                "normal": cp[7],    # Contact normal
                "force": cp[9]      # Normal force
            })

        return contacts

    def remove_object(self, body_id: int):
        """Remove object from simulation."""
        if body_id in self.object_bodies:
            p.removeBody(body_id)
            del self.object_bodies[body_id]

    def reset(self):
        """Reset simulation (remove all objects, keep ground)."""
        for body_id in list(self.object_bodies.keys()):
            self.remove_object(body_id)

        logger.debug("Physics simulation reset")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
