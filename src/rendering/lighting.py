"""Lighting configuration for rendering."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LightingConfig:
    """Lighting configuration for rendering."""

    name: str = "default"
    type: str = "SUN"  # SUN, POINT, SPOT, AREA
    strength: float = 1.0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: Tuple[float, float, float] = (5, -5, 10)
    rotation: Tuple[float, float, float] = (45, 0, 45)  # degrees

    @classmethod
    def get_preset(cls, preset_name: str):
        """
        Get predefined lighting preset.

        Args:
            preset_name: Name of preset (default, bright, warm, cool, sunset)

        Returns:
            LightingConfig instance
        """
        presets = {
            "default": cls(
                name="default",
                strength=1.0,
                color=(1.0, 1.0, 1.0),
            ),
            "bright": cls(
                name="bright",
                strength=2.0,
                color=(1.0, 1.0, 1.0),
            ),
            "warm": cls(
                name="warm",
                strength=1.5,
                color=(1.0, 0.9, 0.7),
            ),
            "cool": cls(
                name="cool",
                strength=1.5,
                color=(0.8, 0.9, 1.0),
            ),
            "sunset": cls(
                name="sunset",
                strength=1.2,
                color=(1.0, 0.6, 0.3),
            ),
        }
        return presets.get(preset_name, presets["default"])

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "strength": self.strength,
            "color": list(self.color),
            "position": list(self.position),
            "rotation": list(self.rotation),
        }
