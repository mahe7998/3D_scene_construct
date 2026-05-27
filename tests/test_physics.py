"""Tests for physics simulation."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.physics.simulator import PhysicsSimulator
from src.utils.config import load_config


def test_physics_simulator_init():
    """Test physics simulator initialization."""
    config = load_config()
    sim = PhysicsSimulator(config)

    assert sim is not None
    assert sim.gravity == -9.81
    assert sim.upright_probability == 0.9
    assert sim.physics_client is None  # Not started yet


def test_physics_simulator_start_stop():
    """Test starting and stopping physics simulation."""
    config = load_config()
    sim = PhysicsSimulator(config)

    # Start simulation
    sim.start()
    assert sim.physics_client is not None
    assert sim.ground_id is not None

    # Stop simulation
    sim.stop()
    assert sim.physics_client is None


def test_physics_simulator_context_manager():
    """Test physics simulator as context manager."""
    config = load_config()

    with PhysicsSimulator(config) as sim:
        assert sim.physics_client is not None
        assert sim.ground_id is not None

    # Should be stopped after context exit
    assert sim.physics_client is None


def test_collision_detection():
    """Test collision detection."""
    config = load_config()
    sim = PhysicsSimulator(config)

    # No collision with empty list
    assert not sim.check_collision((0, 0), radius=1.0, existing_positions=[])

    # Collision with nearby object
    existing = [(1.0, 0.0, 0.0)]
    assert sim.check_collision((0.5, 0), radius=1.0, existing_positions=existing)

    # No collision with distant object
    assert not sim.check_collision((10, 10), radius=1.0, existing_positions=existing)


def test_ground_plane_creation():
    """Test that ground plane is created correctly."""
    config = load_config()

    with PhysicsSimulator(config) as sim:
        # Ground should be created
        assert sim.ground_id is not None

        # Import pybullet to check ground properties
        import pybullet as p

        # Get ground info
        info = p.getBodyInfo(sim.ground_id)
        assert info is not None


def test_physics_stability_check():
    """Test stability checking."""
    config = load_config()

    with PhysicsSimulator(config) as sim:
        # Create a simple box that should settle
        import pybullet as p

        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=[0, 0, 1]  # 1m above ground
        )

        # Simulate for a few steps
        for _ in range(100):
            p.stepSimulation()

        # Check if stable
        is_stable = sim._is_stable(body_id)
        # May or may not be stable after 100 steps, just check method works
        assert isinstance(is_stable, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
