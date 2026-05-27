"""Stereo camera configuration for rendering."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class StereoCamera:
    """Stereo camera pair configuration."""

    # Camera separation (baseline) - human average is ~6.5cm
    baseline: float = 0.065  # meters

    # Focal length
    focal_length: float = 50.0  # mm
    sensor_width: float = 36.0  # mm

    # Camera positions (meters)
    center_position: Tuple[float, float, float] = (0, -5, 3)
    look_at: Tuple[float, float, float] = (0, 0, 0)

    # Rotation (degrees)
    rotation: Tuple[float, float, float] = (60, 0, 0)

    def get_left_camera_config(self):
        """
        Get left camera configuration.

        Returns:
            Dictionary with camera parameters
        """
        # Left camera is offset by -baseline/2 in X
        offset = np.array([-self.baseline / 2, 0, 0])
        left_position = tuple(np.array(self.center_position) + offset)

        return {
            "position": left_position,
            "rotation": self.rotation,
            "focal_length": self.focal_length,
            "sensor_width": self.sensor_width,
            "look_at": self.look_at,
        }

    def get_right_camera_config(self):
        """
        Get right camera configuration.

        Returns:
            Dictionary with camera parameters
        """
        # Right camera is offset by +baseline/2 in X
        offset = np.array([self.baseline / 2, 0, 0])
        right_position = tuple(np.array(self.center_position) + offset)

        return {
            "position": right_position,
            "rotation": self.rotation,
            "focal_length": self.focal_length,
            "sensor_width": self.sensor_width,
            "look_at": self.look_at,
        }

    def get_camera_intrinsics(self, resolution: int = 512):
        """
        Compute camera intrinsic matrix.

        Args:
            resolution: Image resolution (assumes square)

        Returns:
            3x3 intrinsic matrix
        """
        # Compute focal length in pixels
        fx = fy = (self.focal_length / self.sensor_width) * resolution

        # Principal point (image center)
        cx = cy = resolution / 2

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        return intrinsics

    def to_dict(self, resolution: int = 512):
        """
        Convert to dictionary format for storage.

        Args:
            resolution: Image resolution

        Returns:
            Dictionary with all stereo camera parameters
        """
        return {
            "baseline": self.baseline,
            "focal_length": self.focal_length,
            "sensor_width": self.sensor_width,
            "center_position": list(self.center_position),
            "look_at": list(self.look_at),
            "rotation": list(self.rotation),
            "left_camera": self.get_left_camera_config(),
            "right_camera": self.get_right_camera_config(),
            "intrinsics": self.get_camera_intrinsics(resolution).tolist(),
            "resolution": resolution,
        }

    @classmethod
    def from_spherical(
        cls,
        distance: float,
        azimuth: float,
        elevation: float,
        baseline: float = 0.065,
        look_at=(0, 0, 0)
    ):
        """
        Create stereo camera from spherical coordinates.

        Args:
            distance: Distance from look_at point
            azimuth: Horizontal angle in degrees
            elevation: Vertical angle in degrees
            baseline: Stereo baseline in meters
            look_at: Point to look at

        Returns:
            StereoCamera instance
        """
        import math

        # Convert to radians
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)

        # Calculate center position
        x = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        y = -distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        z = distance * math.sin(elevation_rad)

        center_position = (
            x + look_at[0],
            y + look_at[1],
            z + look_at[2],
        )

        # Calculate rotation
        rotation = (
            90 - elevation,
            0,
            azimuth,
        )

        return cls(
            baseline=baseline,
            center_position=center_position,
            look_at=look_at,
            rotation=rotation,
        )
