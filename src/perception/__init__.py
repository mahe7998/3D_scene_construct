"""Perception (scene reconstruction) module: localization, detection, retrieval.

Lazy imports: localizer pulls in cv2 (via the stereo depth estimator) which is
absent from the Blender image, so expose names via __getattr__.
"""

__all__ = ["CameraGeometry", "localize_scene"]


def __getattr__(name):
    if name in ("CameraGeometry", "localize_scene"):
        from . import localizer
        return getattr(localizer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
