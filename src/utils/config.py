"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for the application."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax."""
        return self.get(key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()


# Global config instance
_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Config instance
    """
    global _config

    # Load environment variables
    load_dotenv()

    # Determine config path
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "/app/config/config.yaml")

    # Load YAML config
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}

    # Override with environment variables
    _apply_env_overrides(config_dict)

    _config = Config(config_dict)
    return _config


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    if _config is None:
        return load_config()
    return _config


def _apply_env_overrides(config_dict: Dict[str, Any]) -> None:
    """Apply environment variable overrides to config dictionary."""
    # Rendering
    if "RENDER_RESOLUTION" in os.environ:
        config_dict.setdefault("rendering", {})["resolution"] = int(
            os.environ["RENDER_RESOLUTION"]
        )
    if "RENDER_SAMPLES" in os.environ:
        config_dict.setdefault("rendering", {})["samples"] = int(
            os.environ["RENDER_SAMPLES"]
        )
    if "RENDER_ENGINE" in os.environ:
        config_dict.setdefault("rendering", {})["engine"] = os.environ["RENDER_ENGINE"]
    if "RENDER_DEVICE" in os.environ:
        config_dict.setdefault("rendering", {})["device"] = os.environ["RENDER_DEVICE"]

    # Assets
    if "ASSET_COUNT" in os.environ:
        config_dict.setdefault("assets", {})["count"] = int(os.environ["ASSET_COUNT"])
    if "ASSET_BATCH_SIZE" in os.environ:
        config_dict.setdefault("assets", {})["batch_size"] = int(
            os.environ["ASSET_BATCH_SIZE"]
        )

    # Vision
    if "VISION_MODEL" in os.environ:
        config_dict.setdefault("vision", {})["model"] = os.environ["VISION_MODEL"]
    if "VISION_BATCH_SIZE" in os.environ:
        config_dict.setdefault("vision", {})["batch_size"] = int(
            os.environ["VISION_BATCH_SIZE"]
        )

    # RL
    if "RL_ALGORITHM" in os.environ:
        config_dict.setdefault("rl", {})["algorithm"] = os.environ["RL_ALGORITHM"]
    if "RL_LEARNING_RATE" in os.environ:
        config_dict.setdefault("rl", {})["learning_rate"] = float(
            os.environ["RL_LEARNING_RATE"]
        )
    if "RL_TOTAL_TIMESTEPS" in os.environ:
        config_dict.setdefault("rl", {})["total_timesteps"] = int(
            os.environ["RL_TOTAL_TIMESTEPS"]
        )

    # Paths
    if "DATA_DIR" in os.environ:
        config_dict.setdefault("paths", {})["data"] = os.environ["DATA_DIR"]
