"""
Stereo Vision Analyzer

Integrates stereo disparity with Vision LLM for comprehensive scene understanding.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
from PIL import Image

from src.stereo.disparity import DisparityEstimator
from src.vision.vision_model import VisionModel
from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("stereo_vision_analyzer")


class StereoVisionAnalyzer:
    """Analyze scenes using stereo vision and LLM."""

    def __init__(self, config=None, vision_model=None, disparity_estimator=None):
        """
        Initialize stereo vision analyzer.

        Args:
            config: Configuration object
            vision_model: VisionModel instance
            disparity_estimator: DisparityEstimator instance
        """
        self.config = config or load_config()
        self.vision_model = vision_model or VisionModel(self.config)
        self.disparity_estimator = disparity_estimator or DisparityEstimator(self.config)

        logger.info("Stereo vision analyzer initialized")

    def analyze_stereo_scene(
        self,
        left_path: str,
        right_path: str,
        camera_params: Dict,
        prompt: Optional[str] = None
    ) -> Dict:
        """
        Analyze a stereo scene comprehensively.

        Args:
            left_path: Path to left image
            right_path: Path to right image
            camera_params: Camera parameters (focal_length, baseline, etc.)
            prompt: Optional custom prompt for Vision LLM

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing stereo scene: {left_path}, {right_path}")

        # Compute disparity and depth
        disparity_result = self.disparity_estimator.process_stereo_pair(
            left_path,
            right_path,
            camera_params
        )

        # Get semantic description from Vision LLM (using left image)
        if prompt is None:
            prompt = self._get_stereo_prompt()

        vision_result = self.vision_model.describe_image(left_path, prompt)

        # Combine results
        analysis = {
            "stereo_pair": {
                "left": left_path,
                "right": right_path,
            },
            "camera_params": camera_params,
            "disparity": {
                "min": disparity_result["disparity_min"],
                "max": disparity_result["disparity_max"],
                "map_shape": disparity_result["disparity"].shape,
            },
            "depth": {
                "min": disparity_result["depth_min"],
                "max": disparity_result["depth_max"],
                "map_shape": disparity_result["depth"].shape,
            },
            "vision_description": vision_result["description"],
            "semantic_analysis": self._extract_semantics(vision_result["description"]),
            "3d_reconstruction_ready": True,
        }

        # Store raw arrays for further processing
        analysis["_disparity_map"] = disparity_result["disparity"]
        analysis["_depth_map"] = disparity_result["depth"]

        return analysis

    def _get_stereo_prompt(self) -> str:
        """Get prompt for stereo scene analysis."""
        return """Describe this 3D scene in detail, focusing on:
        1. Each object visible in the scene
        2. The approximate position of each object (left/center/right, near/far, foreground/background)
        3. The relative distances between objects
        4. Any occlusions (which objects are in front of others)
        5. The ground/floor surface
        6. Overall spatial layout

        For each object, provide:
        - Object type/category
        - Approximate size (small/medium/large)
        - Position in 3D space (describe depth, horizontal, and vertical position)
        - Orientation (upright, tilted, sideways, etc.)
        - Visibility (fully visible, partially occluded)

        Format as a structured description."""

    def _extract_semantics(self, description: str) -> Dict:
        """
        Extract structured semantics from description.

        Args:
            description: Vision LLM description

        Returns:
            Structured semantic information
        """
        # Simple keyword extraction (can be enhanced with NLP)
        semantics = {
            "objects_mentioned": [],
            "spatial_relations": [],
            "occlusions": [],
        }

        # Basic parsing (can be improved with proper NLU)
        description_lower = description.lower()

        # Common object keywords
        object_keywords = [
            "car", "vehicle", "chair", "table", "person", "human",
            "animal", "building", "tree", "ball", "cube", "sphere"
        ]

        for keyword in object_keywords:
            if keyword in description_lower:
                semantics["objects_mentioned"].append(keyword)

        # Spatial relation keywords
        spatial_keywords = [
            "in front", "behind", "left", "right", "above", "below",
            "near", "far", "foreground", "background"
        ]

        for keyword in spatial_keywords:
            if keyword in description_lower:
                semantics["spatial_relations"].append(keyword)

        return semantics

    def compute_3d_points(
        self,
        disparity: np.ndarray,
        camera_params: Dict
    ) -> np.ndarray:
        """
        Compute 3D point cloud from disparity map.

        Args:
            disparity: Disparity map
            camera_params: Camera intrinsics and baseline

        Returns:
            Point cloud (N, 3) array [x, y, z]
        """
        h, w = disparity.shape

        # Get camera parameters
        intrinsics = np.array(camera_params.get("intrinsics"))
        baseline = camera_params.get("baseline", 0.065)

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Compute depth
        depth = (fx * baseline) / (disparity + 1e-6)

        # Compute 3D coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Reshape to point cloud
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Filter invalid points
        valid = (disparity.reshape(-1) > 0) & (depth.reshape(-1) < 100)
        points = points[valid]

        return points

    def save_point_cloud(
        self,
        points: np.ndarray,
        output_path: str,
        colors: Optional[np.ndarray] = None
    ):
        """
        Save point cloud to PLY file.

        Args:
            points: Point cloud (N, 3)
            output_path: Output file path
            colors: Optional RGB colors (N, 3) [0-255]
        """
        from plyfile import PlyData, PlyElement

        if colors is None:
            colors = np.full((len(points), 3), 128, dtype=np.uint8)

        # Create structured array
        vertex = np.array(
            [(points[i, 0], points[i, 1], points[i, 2],
              colors[i, 0], colors[i, 1], colors[i, 2])
             for i in range(len(points))],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )

        # Create PLY element
        el = PlyElement.describe(vertex, 'vertex')
        ply_data = PlyData([el])

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ply_data.write(str(output_path))

        logger.info(f"Saved point cloud with {len(points)} points to {output_path}")

    def get_multimodal_features(
        self,
        left_path: str,
        right_path: str,
        camera_params: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Get multi-modal features for RL observation.

        Args:
            left_path: Left image path
            right_path: Right image path
            camera_params: Camera parameters

        Returns:
            Dictionary of feature arrays
        """
        # Load images
        left_img = np.array(Image.open(left_path))
        right_img = np.array(Image.open(right_path))

        # Compute disparity
        disparity_result = self.disparity_estimator.process_stereo_pair(
            left_path,
            right_path,
            camera_params
        )

        # Normalize images to [0, 1]
        left_norm = left_img.astype(np.float32) / 255.0
        right_norm = right_img.astype(np.float32) / 255.0

        # Normalize disparity
        disparity = disparity_result["disparity"]
        if disparity.max() > 0:
            disparity_norm = disparity / disparity.max()
        else:
            disparity_norm = disparity

        features = {
            "left_rgb": left_norm,  # (H, W, 3)
            "right_rgb": right_norm,  # (H, W, 3)
            "disparity": disparity_norm[..., np.newaxis],  # (H, W, 1)
            "depth": disparity_result["depth"],  # (H, W)
        }

        return features
