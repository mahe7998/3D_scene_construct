"""Stereoscopic vision module."""

from .stereo_camera import StereoCamera

# Lazy imports for cv2-dependent modules (not available in Blender environment)
__all__ = ["StereoCamera"]

try:
    from .disparity import DisparityEstimator
    from .stereo_vision import StereoVisionAnalyzer
    __all__.extend(["DisparityEstimator", "StereoVisionAnalyzer"])
except ImportError:
    # cv2 not available (e.g., in Blender Python environment)
    pass
