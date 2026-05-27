"""Integration tests for physics and stereo."""

import pytest
import numpy as np
from pathlib import Path

from src.utils.config import load_config
from src.physics.simulator import PhysicsSimulator
from src.stereo.stereo_camera import StereoCamera


def test_config_loads_physics_settings():
    """Test that physics settings load from config."""
    config = load_config()

    # Physics settings should be loaded
    assert config.get("physics.enabled") is not None
    assert config.get("physics.simulation.gravity") == -9.81
    assert config.get("physics.object_placement.upright_probability") == 0.9


def test_config_loads_stereo_settings():
    """Test that stereo settings load from config."""
    config = load_config()

    # Stereo settings should be loaded
    assert config.get("stereo.enabled") is not None
    assert config.get("stereo.camera.baseline") == 0.065
    assert config.get("stereo.disparity.algorithm") in ["SGBM", "BM"]


def test_physics_simulator_with_config():
    """Test physics simulator uses config correctly."""
    config = load_config()
    sim = PhysicsSimulator(config)

    # Should use config values
    assert sim.gravity == config.get("physics.simulation.gravity", -9.81)
    assert sim.upright_probability == config.get("physics.object_placement.upright_probability", 0.9)


def test_stereo_camera_baseline_from_config():
    """Test stereo camera can use config baseline."""
    config = load_config()
    baseline = config.get("stereo.camera.baseline", 0.065)

    cam = StereoCamera(baseline=baseline)
    assert cam.baseline == 0.065


def test_modules_importable():
    """Test that all new modules can be imported."""
    # Physics
    from src.physics import PhysicsSimulator
    assert PhysicsSimulator is not None

    # Stereo
    from src.stereo import StereoCamera, DisparityEstimator, StereoVisionAnalyzer
    assert StereoCamera is not None
    assert DisparityEstimator is not None
    assert StereoVisionAnalyzer is not None

    # Scene
    from src.scene.physics_scene_generator import PhysicsSceneGenerator
    assert PhysicsSceneGenerator is not None


def test_end_to_end_physics_flow():
    """Test complete physics simulation flow."""
    config = load_config()

    with PhysicsSimulator(config) as sim:
        # Simulate a simple drop
        import pybullet as p

        # Create a box
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=[0, 0, 2]  # Start 2m above ground
        )

        # Simulate until stable
        states = sim.simulate_until_stable([body_id], max_steps=500)

        # Check we got a result
        assert body_id in states
        assert "position" in states[body_id]
        assert "orientation" in states[body_id]

        # Object should have fallen and be on or near ground
        final_z = states[body_id]["position"][2]
        assert final_z < 1.0  # Should be near ground (box half-height is 0.5)


def test_stereo_camera_and_intrinsics_match():
    """Test stereo camera intrinsics match expected format."""
    cam = StereoCamera(baseline=0.065, focal_length=50.0)

    cam_dict = cam.to_dict(resolution=512)

    # Check format matches what renderer and disparity estimator expect
    assert "baseline" in cam_dict
    assert "focal_length" in cam_dict
    assert "intrinsics" in cam_dict
    assert "left_camera" in cam_dict
    assert "right_camera" in cam_dict

    # Intrinsics should be 3x3
    intrinsics = np.array(cam_dict["intrinsics"])
    assert intrinsics.shape == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
