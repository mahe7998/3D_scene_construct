"""
Blender-based 3D renderer

Renders 3D assets using Blender's Cycles engine with hardware acceleration.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import Blender Python API
try:
    import bpy
    import bmesh
except ImportError:
    print("Warning: Blender Python API (bpy) not available")
    print("This module must be run within Blender or with Blender Python")
    bpy = None

from src.utils.config import load_config
from src.utils.database import Database
from src.utils.logger import get_logger
from src.rendering.camera import CameraConfig
from src.rendering.lighting import LightingConfig


logger = get_logger("renderer")


class BlenderRenderer:
    """Render 3D assets using Blender."""

    def __init__(self, config=None, db=None):
        """
        Initialize Blender renderer.

        Args:
            config: Configuration object
            db: Database instance
        """
        if bpy is None:
            raise ImportError("Blender Python API not available")

        self.config = config or load_config()
        self.db = db or Database(self.config.get("database.path", "/data/database/assets.db"))

        # Get rendering settings
        self.resolution = self.config.get("rendering.resolution", 512)
        self.samples = self.config.get("rendering.samples", 128)
        self.engine = self.config.get("rendering.engine", "CYCLES")
        self.device = self.config.get("rendering.device", "GPU")
        self.format = self.config.get("rendering.format", "JPEG")
        self.quality = self.config.get("rendering.quality", 90)

        # Paths
        self.assets_raw_dir = Path(
            self.config.get("paths.assets_raw", "/data/assets/raw")
        )
        self.assets_rendered_dir = Path(
            self.config.get("paths.assets_rendered", "/data/assets/rendered")
        )
        self.assets_rendered_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Blender scene
        self._setup_blender()

        logger.info("Blender renderer initialized")
        logger.info(f"Engine: {self.engine}, Device: {self.device}")
        logger.info(f"Resolution: {self.resolution}x{self.resolution}")

    def _setup_blender(self):
        """Configure Blender scene for rendering."""
        # Clear default scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Set render engine
        bpy.context.scene.render.engine = self.engine

        # Configure Cycles
        if self.engine == 'CYCLES':
            cycles = bpy.context.scene.cycles
            cycles.samples = self.samples

            # Set device
            if self.device == 'GPU':
                cycles.device = 'GPU'
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = 'CUDA'  # or 'OPTIX' or 'METAL'

                # Enable all available GPUs
                for device in prefs.devices:
                    device.use = True
            else:
                cycles.device = 'CPU'

        # Set resolution
        bpy.context.scene.render.resolution_x = self.resolution
        bpy.context.scene.render.resolution_y = self.resolution
        bpy.context.scene.render.resolution_percentage = 100

        # Set output format
        bpy.context.scene.render.image_settings.file_format = self.format
        if self.format == 'JPEG':
            bpy.context.scene.render.image_settings.quality = self.quality

        # Enable transparent background (optional)
        bpy.context.scene.render.film_transparent = False

        # Set background color
        bg_color = self.config.get("rendering.background.color", [0.5, 0.5, 0.5])
        bpy.context.scene.world.use_nodes = True
        bg_node = bpy.context.scene.world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = (*bg_color, 1.0)

    def clear_scene(self):
        """Clear all objects from the scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def load_object(self, file_path: str) -> Optional[bpy.types.Object]:
        """
        Load a 3D object into Blender.

        Args:
            file_path: Path to 3D model file

        Returns:
            Blender object or None
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        extension = file_path.suffix.lower()

        try:
            # Select appropriate importer
            if extension in ['.obj']:
                bpy.ops.import_scene.obj(filepath=str(file_path))
            elif extension in ['.fbx']:
                bpy.ops.import_scene.fbx(filepath=str(file_path))
            elif extension in ['.gltf', '.glb']:
                bpy.ops.import_scene.gltf(filepath=str(file_path))
            elif extension in ['.stl']:
                bpy.ops.import_mesh.stl(filepath=str(file_path))
            elif extension in ['.ply']:
                bpy.ops.import_mesh.ply(filepath=str(file_path))
            else:
                logger.warning(f"Unsupported format: {extension}")
                return None

            # Get imported objects (glTF often imports multiple: meshes, empties, cameras)
            imported_objects = list(bpy.context.selected_objects)

            # Remove non-mesh objects first (cameras, lights, empties)
            mesh_objects = []
            for o in imported_objects:
                if o.type == 'MESH' and o.data is not None:
                    mesh_objects.append(o)
                else:
                    bpy.data.objects.remove(o, do_unlink=True)

            if not mesh_objects:
                logger.warning(f"No mesh objects found in {file_path}")
                return None

            # If multiple meshes, join them into one
            if len(mesh_objects) > 1:
                logger.info(f"Joining {len(mesh_objects)} mesh objects")
                bpy.context.view_layer.objects.active = mesh_objects[0]
                bpy.ops.object.select_all(action='DESELECT')
                for obj in mesh_objects:
                    obj.select_set(True)
                bpy.ops.object.join()
                obj = bpy.context.active_object
            else:
                obj = mesh_objects[0]

            logger.info(f"Loaded object: {obj.name}, type: {obj.type}, has data: {obj.data is not None}")

            # Ensure object has at least one material
            if not obj.data.materials:
                logger.info("Object has no materials, adding default material")
                mat = bpy.data.materials.new(name="DefaultMaterial")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                bsdf = nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.5
                obj.data.materials.append(mat)

            return obj

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def center_and_scale_object(self, obj: bpy.types.Object, target_size: float = 2.0):
        """
        Center and scale object to fit within target size.

        Args:
            obj: Blender object
            target_size: Target maximum dimension
        """
        # Center object at origin
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Apply all transformations
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Get bounding box in local/world coords (matrix is identity after apply)
        bbox_corners = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        min_coords = [min([corner[i] for corner in bbox_corners]) for i in range(3)]
        max_coords = [max([corner[i] for corner in bbox_corners]) for i in range(3)]

        # Calculate dimensions and center
        dimensions = [max_coords[i] - min_coords[i] for i in range(3)]
        center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]

        max_dim = max(dimensions)
        if max_dim <= 0:
            logger.warning(f"Object has zero dimensions: {dimensions}")
            return

        scale_factor = target_size / max_dim

        # Apply centering and scaling directly to vertices for predictable result
        for vertex in obj.data.vertices:
            vertex.co.x = (vertex.co.x - center[0]) * scale_factor
            vertex.co.y = (vertex.co.y - center[1]) * scale_factor
            vertex.co.z = (vertex.co.z - center[2]) * scale_factor

        # Update mesh
        obj.data.update()

        # Reset transforms
        obj.location = (0, 0, 0)
        obj.scale = (1, 1, 1)

        logger.info(f"Object dimensions: {dimensions}, max: {max_dim:.3f}, scale: {scale_factor:.3f}, center was at: {center}")

    def setup_camera(self, camera_config: CameraConfig):
        """
        Set up camera in the scene.

        Args:
            camera_config: Camera configuration
        """
        # Delete existing cameras
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Create camera
        camera_data = bpy.data.cameras.new(name='Camera')
        camera_object = bpy.data.objects.new('Camera', camera_data)
        bpy.context.collection.objects.link(camera_object)

        # Set as active camera
        bpy.context.scene.camera = camera_object

        # Set position and rotation
        camera_object.location = camera_config.position
        camera_object.rotation_euler = tuple(
            math.radians(r) for r in camera_config.rotation
        )

        # Set camera properties
        camera_data.lens = camera_config.focal_length
        camera_data.sensor_width = camera_config.sensor_width

        # Point camera at look_at point
        if camera_config.look_at:
            direction = tuple(
                camera_config.look_at[i] - camera_config.position[i] for i in range(3)
            )
            rot_quat = direction_to_rotation(direction)
            camera_object.rotation_euler = rot_quat.to_euler()

    def setup_lighting(self, lighting_config: LightingConfig):
        """
        Set up lighting in the scene.

        Args:
            lighting_config: Lighting configuration
        """
        # Delete existing lights
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Create light
        light_data = bpy.data.lights.new(name='Light', type=lighting_config.type)
        light_object = bpy.data.objects.new('Light', light_data)
        bpy.context.collection.objects.link(light_object)

        # Set position and rotation
        light_object.location = lighting_config.position
        light_object.rotation_euler = tuple(
            math.radians(r) for r in lighting_config.rotation
        )

        # Set light properties. Blender's energy units differ by light type:
        # SUN is W/m^2 (daylight ~5), POINT/SPOT/AREA are Watts (typical 100-1000).
        if lighting_config.type == 'SUN':
            light_data.energy = lighting_config.strength * 5
        else:
            light_data.energy = lighting_config.strength * 1000
        light_data.color = lighting_config.color

    def render_to_file(self, output_path: str):
        """
        Render current scene to file.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

        logger.info(f"Rendered to: {output_path}")

    def render_object_views(
        self,
        object_id: str,
        num_views: Optional[int] = None,
    ) -> List[str]:
        """
        Render multiple views of an object.

        Args:
            object_id: Object ID from database
            num_views: Number of views to render (uses config if not specified)

        Returns:
            List of rendered image paths
        """
        # Get object from database
        obj_data = self.db.get_object(object_id)
        if not obj_data:
            logger.error(f"Object {object_id} not found in database")
            return []

        # File path is stored relative to assets_raw_dir.parent (/data/assets)
        file_path = self.assets_raw_dir.parent / obj_data["file_path"]

        # Clear scene and load object
        self.clear_scene()
        obj = self.load_object(str(file_path))
        if not obj:
            return []

        # Center and scale object
        self.center_and_scale_object(obj)

        # Get camera angles from config
        camera_angles = self.config.get("rendering.camera.angles", [])
        if not camera_angles:
            # Default angles
            camera_angles = [
                [0, 0, 0], [0, 45, 0], [0, 90, 0], [0, 135, 0],
                [0, 180, 0], [0, 225, 0], [0, 270, 0], [0, 315, 0],
                [45, 45, 0], [45, 225, 0], [-30, 45, 0], [-30, 225, 0],
            ]

        num_views = num_views or self.config.get("rendering.views_per_object", 12)
        camera_angles = camera_angles[:num_views]

        # Get lighting presets
        lighting_presets = self.config.get("rendering.lighting.presets", [])
        if not lighting_presets:
            lighting_presets = [{"name": "default"}]

        # Create output directory
        output_dir = self.assets_rendered_dir / obj_data["category"] / object_id
        output_dir.mkdir(parents=True, exist_ok=True)

        rendered_paths = []

        # Render views
        for i, angle in enumerate(camera_angles):
            view_id = f"{i+1:03d}"

            # Set up camera
            distance = self.config.get("rendering.camera.distances", [4])[0]
            camera_config = CameraConfig.from_spherical(
                distance=distance,
                azimuth=angle[1],
                elevation=angle[0],
                look_at=(0, 0, 0),
            )
            self.setup_camera(camera_config)

            # Set up lighting
            lighting_preset = lighting_presets[i % len(lighting_presets)]
            lighting_config = LightingConfig.get_preset(lighting_preset.get("name", "default"))
            self.setup_lighting(lighting_config)

            # Render
            output_path = output_dir / f"view_{view_id}.jpg"
            self.render_to_file(str(output_path))
            rendered_paths.append(str(output_path))

            # Add to database
            self.db.add_render(
                object_id=object_id,
                view_id=view_id,
                image_path=str(output_path.relative_to(self.assets_rendered_dir.parent)),
                camera_params=camera_config.to_dict(),
                lighting=lighting_config.to_dict(),
            )

            logger.info(f"Rendered view {view_id} for {object_id}")

        return rendered_paths

    def render_object_stereo_views(
        self,
        object_id: str,
        num_views: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Render multiple stereo view pairs of an object.

        Args:
            object_id: Object ID from database
            num_views: Number of stereo pairs to render (uses config if not specified)

        Returns:
            List of dicts with left/right image paths and camera params
        """
        from src.stereo.stereo_camera import StereoCamera

        # Get object from database
        obj_data = self.db.get_object(object_id)
        if not obj_data:
            logger.error(f"Object {object_id} not found in database")
            return []

        # File path is stored relative to assets_raw_dir.parent (/data/assets)
        file_path = self.assets_raw_dir.parent / obj_data["file_path"]

        # Clear scene and load object
        self.clear_scene()
        obj = self.load_object(str(file_path))
        if not obj:
            return []

        # Center and scale object
        self.center_and_scale_object(obj)

        # Get camera angles from config (simplified for stereo - use 6 key angles)
        camera_angles = self.config.get("rendering.camera.angles", [])
        if not camera_angles:
            # Default 6 angles around the object
            camera_angles = [
                [0, 0, 0],      # Front
                [0, 90, 0],     # Right
                [0, 180, 0],    # Back
                [0, 270, 0],    # Left
                [30, 45, 0],    # Top-front-right
                [-30, 225, 0],  # Bottom-back-left
            ]

        num_views = num_views or self.config.get("rendering.stereo_views_per_object", 6)
        camera_angles = camera_angles[:num_views]

        # Get lighting preset
        lighting_config = LightingConfig.get_preset("default")
        self.setup_lighting(lighting_config)

        # Get stereo baseline
        baseline = self.config.get("stereo.camera.baseline", 0.065)

        # Create output directory
        output_dir = self.assets_rendered_dir / obj_data["category"] / object_id
        output_dir.mkdir(parents=True, exist_ok=True)

        rendered_pairs = []

        # Render stereo pairs from each angle
        for i, angle in enumerate(camera_angles):
            view_id = f"{i+1:03d}"

            # Set up stereo camera
            distance = self.config.get("rendering.camera.distances", [4])[0]
            center_cam = CameraConfig.from_spherical(
                distance=distance,
                azimuth=angle[1],
                elevation=angle[0],
                look_at=(0, 0, 0),
            )

            stereo_camera = StereoCamera(
                baseline=baseline,
                center_position=center_cam.position,
                look_at=(0, 0, 0),
                focal_length=center_cam.focal_length
            )

            # Output paths
            output_left = output_dir / f"view_{view_id}_L.jpg"
            output_right = output_dir / f"view_{view_id}_R.jpg"

            # Render stereo pair
            camera_params = self.render_stereo_pair(
                str(output_left),
                str(output_right),
                stereo_camera,
                baseline
            )

            # Add both renders to database
            self.db.add_render(
                object_id=object_id,
                view_id=f"{view_id}_L",
                image_path=str(output_left.relative_to(self.assets_rendered_dir.parent)),
                camera_params=camera_params['left_camera'],
                lighting=lighting_config.to_dict(),
            )

            self.db.add_render(
                object_id=object_id,
                view_id=f"{view_id}_R",
                image_path=str(output_right.relative_to(self.assets_rendered_dir.parent)),
                camera_params=camera_params['right_camera'],
                lighting=lighting_config.to_dict(),
            )

            rendered_pairs.append({
                'left': str(output_left),
                'right': str(output_right),
                'camera_params': camera_params
            })

            logger.info(f"Rendered stereo view {view_id} for {object_id}")

        return rendered_pairs

    def render_stereo_pair(
        self,
        output_left: str,
        output_right: str,
        camera_config: 'StereoCamera' = None,
        baseline: float = 0.065
    ) -> Dict:
        """
        Render stereo pair (left and right views).

        Args:
            output_left: Output path for left image
            output_right: Output path for right image
            camera_config: StereoCamera configuration (optional)
            baseline: Stereo baseline in meters (default: 0.065m = human IPD)

        Returns:
            Dictionary with camera parameters
        """
        from src.stereo.stereo_camera import StereoCamera
        import mathutils

        # Use provided camera or create default
        if camera_config is None:
            camera_config = StereoCamera(
                baseline=baseline,
                center_position=tuple(self.config.get("rendering.camera.position", (0, -5, 3))),
                focal_length=self.config.get("rendering.camera.focal_length", 50.0)
            )

        # Get left and right camera configs
        left_cam = camera_config.get_left_camera_config()
        right_cam = camera_config.get_right_camera_config()

        # Render left view
        logger.debug("Rendering left eye view...")
        cam_config = CameraConfig(
            position=left_cam["position"],
            rotation=left_cam["rotation"],
            focal_length=left_cam["focal_length"],
            sensor_width=left_cam["sensor_width"],
            look_at=left_cam["look_at"]
        )
        self.setup_camera(cam_config)
        self.render_to_file(output_left)

        # Render right view
        logger.debug("Rendering right eye view...")
        cam_config = CameraConfig(
            position=right_cam["position"],
            rotation=right_cam["rotation"],
            focal_length=right_cam["focal_length"],
            sensor_width=right_cam["sensor_width"],
            look_at=right_cam["look_at"]
        )
        self.setup_camera(cam_config)
        self.render_to_file(output_right)

        logger.info(f"Stereo pair rendered: {output_left}, {output_right}")

        # Return camera parameters for stereo processing
        return camera_config.to_dict(resolution=self.resolution)

    def create_ground_plane(
        self,
        size: float = 10.0,
        position: Tuple[float, float, float] = (0, 0, 0),
        material_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    ):
        """
        Create a flat ground plane.

        Args:
            size: Size of the ground plane (meters)
            position: Position of the ground (default at origin)
            material_color: RGB color of the ground material
        """
        # Create plane mesh
        bpy.ops.mesh.primitive_plane_add(
            size=size,
            location=position
        )

        ground = bpy.context.active_object
        ground.name = "Ground"

        # Create material
        mat = bpy.data.materials.new(name="GroundMaterial")
        mat.use_nodes = True

        # Get material nodes
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")

        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*material_color, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.7
            bsdf.inputs["Metallic"].default_value = 0.0

        # Assign material
        if ground.data.materials:
            ground.data.materials[0] = mat
        else:
            ground.data.materials.append(mat)

        logger.info(f"Ground plane created at {position} with size {size}m")

        return ground

    def place_object_at_position(
        self,
        mesh_path: str,
        position: Tuple[float, float, float],
        rotation_quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        scale: Tuple[float, float, float] = (1, 1, 1)
    ):
        """
        Load and place object at specific position with rotation (from physics).

        Args:
            mesh_path: Path to 3D mesh file
            position: (x, y, z) position
            rotation_quat: Quaternion rotation (x, y, z, w)
            scale: Scale factors (sx, sy, sz)

        Returns:
            Blender object
        """
        # Load object
        obj = self.load_object(mesh_path)
        if not obj:
            return None

        # Set position
        obj.location = position

        # Set rotation (quaternion)
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = rotation_quat

        # Set scale
        obj.scale = scale

        logger.debug(f"Placed object at {position} with rotation {rotation_quat}")

        return obj


def direction_to_rotation(direction):
    """Convert direction vector to rotation quaternion."""
    import mathutils
    # Blender cameras look down -Z, so track -Z (not +Z) to the look-at direction.
    return mathutils.Vector(direction).to_track_quat('-Z', 'Y')


def main():
    """Command-line interface for renderer."""
    parser = argparse.ArgumentParser(description="Render 3D assets")
    parser.add_argument(
        "--mode",
        type=str,
        default="assets",
        choices=["assets", "scenes", "test"],
        help="Rendering mode",
    )
    parser.add_argument(
        "--object-id",
        type=str,
        help="Specific object ID to render",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test render",
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Render stereo pairs (default: mono views)",
    )

    # When launched via `blender --background --python script.py -- <args>`,
    # sys.argv contains Blender's own arguments too. Strip everything up to
    # and including the `--` separator so argparse only sees our flags.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    if bpy is None:
        logger.error("Blender Python API not available")
        sys.exit(1)

    # Load config
    config = load_config()

    # Create renderer
    renderer = BlenderRenderer(config=config)

    if args.test:
        logger.info("Test mode - creating simple scene")
        # Create a test cube with a procedural checker material so the render
        # shows both shape (shading) and texture.
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
        cube = bpy.context.active_object

        mat = bpy.data.materials.new(name="TestCheckerMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs["Roughness"].default_value = 0.4
        coord = nodes.new("ShaderNodeTexCoord")
        checker = nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = 5.0
        checker.inputs["Color1"].default_value = (0.85, 0.15, 0.15, 1.0)
        checker.inputs["Color2"].default_value = (0.95, 0.95, 0.95, 1.0)
        links.new(coord.outputs["Generated"], checker.inputs["Vector"])
        links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
        cube.data.materials.append(mat)

        # Solid grey ground; place it at z=-1 so the cube sits on top.
        renderer.create_ground_plane(
            size=20.0,
            position=(0, 0, -1),
            material_color=(0.5, 0.5, 0.5),
        )

        camera_config = CameraConfig.from_spherical(5, 45, 30)
        renderer.setup_camera(camera_config)
        lighting_config = LightingConfig.get_preset("default")
        renderer.setup_lighting(lighting_config)
        renderer.render_to_file("/data/test_render.jpg")
        logger.info("Test render complete")

    elif args.mode == "assets":
        if args.object_id:
            # Render specific object
            if args.stereo:
                renderer.render_object_stereo_views(args.object_id)
            else:
                renderer.render_object_views(args.object_id)
        else:
            # Render all objects
            db = Database(config.get("database.path", "/data/database/assets.db"))
            objects = db.get_all_objects()

            if args.stereo:
                logger.info(f"Rendering {len(objects)} objects with stereo pairs")
                for obj in objects:
                    logger.info(f"Rendering stereo views: {obj['id']} ({obj['name']})")
                    renderer.render_object_stereo_views(obj['id'])
            else:
                logger.info(f"Rendering {len(objects)} objects")
                for obj in objects:
                    logger.info(f"Rendering object: {obj['id']} ({obj['name']})")
                    renderer.render_object_views(obj['id'])


if __name__ == "__main__":
    main()
