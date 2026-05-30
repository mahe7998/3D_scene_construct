"""Scene generation module.

Submodules are imported lazily: ``generator``/``physics_scene_generator`` pull in
heavy deps (objaverse, pybullet, trimesh) that are absent from the Blender image,
while ``scene_builder`` is pure-Python and must import there. So we avoid eager
top-level imports and expose names via __getattr__.
"""

__all__ = ["SceneGenerator", "PhysicsSceneGenerator", "build_demo_scene"]


def __getattr__(name):
    if name == "SceneGenerator":
        from .generator import SceneGenerator
        return SceneGenerator
    if name == "PhysicsSceneGenerator":
        from .physics_scene_generator import PhysicsSceneGenerator
        return PhysicsSceneGenerator
    if name == "build_demo_scene":
        from .scene_builder import build_demo_scene
        return build_demo_scene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
