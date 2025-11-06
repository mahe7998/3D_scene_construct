"""Tests for configuration management."""

import pytest
from src.utils.config import load_config, Config


def test_load_config():
    """Test loading configuration."""
    config = load_config()
    assert config is not None
    assert isinstance(config, Config)


def test_config_get():
    """Test getting configuration values."""
    config = load_config()

    # Test nested key access
    resolution = config.get("rendering.resolution", 512)
    assert isinstance(resolution, int)

    # Test default value
    unknown = config.get("nonexistent.key", "default")
    assert unknown == "default"


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = load_config()
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
