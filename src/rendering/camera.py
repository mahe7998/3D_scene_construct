"""Camera configuration for rendering."""

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class CameraConfig:
    """Camera configuration for rendering."""

    position: Tuple[float, float, float] = (0, -5, 3)
    rotation: Tuple[float, float, float] = (60, 0, 0)  # degrees
    focal_length: float = 50.0  # mm
    sensor_width: float = 36.0  # mm
    look_at: Tuple[float, float, float] = (0, 0, 0)

    @classmethod
    def from_spherical(
        cls, distance: float, azimuth: float, elevation: float, look_at=(0, 0, 0)
    ):
        """
        Create camera config from spherical coordinates.

        Args:
            distance: Distance from look_at point
            azimuth: Horizontal angle in degrees (0 = front)
            elevation: Vertical angle in degrees (0 = horizontal)
            look_at: Point to look at

        Returns:
            CameraConfig instance
        """
        # Convert to radians
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)

        # Calculate position
        x = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        y = -distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        z = distance * math.sin(elevation_rad)

        position = (
            x + look_at[0],
            y + look_at[1],
            z + look_at[2],
        )

        # Calculate rotation (pointing at look_at)
        rotation = (
            90 - elevation,  # Pitch
            0,  # Roll
            azimuth,  # Yaw
        )

        return cls(
            position=position,
            rotation=rotation,
            look_at=look_at,
        )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "position": list(self.position),
            "rotation": list(self.rotation),
            "focal_length": self.focal_length,
            "sensor_width": self.sensor_width,
            "look_at": list(self.look_at),
        }
