"""Tests for stereo vision components."""

import pytest
import numpy as np
from pathlib import Path

from src.stereo.stereo_camera import StereoCamera
from src.stereo.disparity import DisparityEstimator
from src.utils.config import load_config


def test_stereo_camera_init():
    """Test stereo camera initialization."""
    cam = StereoCamera(baseline=0.065)

    assert cam.baseline == 0.065
    assert cam.focal_length == 50.0
    assert len(cam.center_position) == 3


def test_stereo_camera_left_right_config():
    """Test getting left and right camera configs."""
    cam = StereoCamera(
        baseline=0.065,
        center_position=(0, -5, 3)
    )

    left = cam.get_left_camera_config()
    right = cam.get_right_camera_config()

    # Check positions are offset by baseline
    assert left["position"][0] == -0.065 / 2
    assert right["position"][0] == 0.065 / 2

    # Y and Z should be the same
    assert left["position"][1] == right["position"][1]
    assert left["position"][2] == right["position"][2]


def test_stereo_camera_intrinsics():
    """Test camera intrinsics computation."""
    cam = StereoCamera(focal_length=50.0, sensor_width=36.0)

    intrinsics = cam.get_camera_intrinsics(resolution=512)

    # Should be 3x3 matrix
    assert intrinsics.shape == (3, 3)

    # Focal length in pixels
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    assert fx == fy  # Square pixels

    # Principal point at image center
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    assert cx == 256  # Half of 512
    assert cy == 256


def test_stereo_camera_from_spherical():
    """Test creating stereo camera from spherical coordinates."""
    cam = StereoCamera.from_spherical(
        distance=5,
        azimuth=45,
        elevation=30,
        baseline=0.065
    )

    assert cam.baseline == 0.065
    assert len(cam.center_position) == 3

    # Check distance is approximately correct
    pos = np.array(cam.center_position)
    dist = np.linalg.norm(pos)
    assert abs(dist - 5.0) < 0.1  # Within 10cm


def test_stereo_camera_to_dict():
    """Test converting stereo camera to dictionary."""
    cam = StereoCamera(baseline=0.065)

    cam_dict = cam.to_dict(resolution=512)

    assert "baseline" in cam_dict
    assert "left_camera" in cam_dict
    assert "right_camera" in cam_dict
    assert "intrinsics" in cam_dict
    assert cam_dict["baseline"] == 0.065
    assert cam_dict["resolution"] == 512


def test_disparity_estimator_init():
    """Test disparity estimator initialization."""
    config = load_config()
    estimator = DisparityEstimator(config)

    assert estimator is not None
    assert estimator.algorithm in ["SGBM", "BM"]
    assert estimator.stereo is not None


def test_disparity_compute_with_synthetic_images():
    """Test disparity computation with synthetic images."""
    config = load_config()
    estimator = DisparityEstimator(config)

    # Create simple synthetic stereo pair
    height, width = 256, 256

    # Left image: white square on black background
    left = np.zeros((height, width, 3), dtype=np.uint8)
    left[100:150, 100:150] = 255

    # Right image: same square shifted right (simulating disparity)
    right = np.zeros((height, width, 3), dtype=np.uint8)
    right[100:150, 110:160] = 255

    # Compute disparity
    disparity = estimator.compute_disparity(left, right, normalize=True)

    # Should return array of same height and width
    assert disparity.shape == (height, width)

    # Disparity should be normalized to [0, 1]
    assert disparity.min() >= 0
    assert disparity.max() <= 1


def test_disparity_to_depth():
    """Test disparity to depth conversion."""
    config = load_config()
    estimator = DisparityEstimator(config)

    # Create simple disparity map
    disparity = np.ones((256, 256)) * 10.0  # 10 pixels disparity

    focal_length = 700.0  # pixels
    baseline = 0.065  # meters

    depth = estimator.disparity_to_depth(disparity, focal_length, baseline)

    # Check shape
    assert depth.shape == disparity.shape

    # Check depth calculation: depth = (f * b) / d
    expected_depth = (focal_length * baseline) / 10.0
    assert abs(depth[0, 0] - expected_depth) < 0.01


def test_visualize_disparity():
    """Test disparity visualization."""
    config = load_config()
    estimator = DisparityEstimator(config)

    # Create simple disparity map
    disparity = np.random.rand(256, 256) * 100

    # Visualize
    vis = estimator.visualize_disparity(disparity)

    # Should be RGB image
    assert vis.shape == (256, 256, 3)
    assert vis.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
