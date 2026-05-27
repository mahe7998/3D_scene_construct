"""
Disparity Estimation

Computes disparity maps from stereo image pairs using OpenCV or deep learning.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("disparity_estimator")


class DisparityEstimator:
    """Estimate disparity and depth from stereo image pairs."""

    def __init__(self, config=None):
        """
        Initialize disparity estimator.

        Args:
            config: Configuration object
        """
        self.config = config or load_config()

        # Stereo matching algorithm
        self.algorithm = self.config.get("stereo.disparity.algorithm", "SGBM")

        # SGBM parameters
        self.min_disparity = self.config.get("stereo.disparity.min_disparity", 0)
        self.num_disparities = self.config.get("stereo.disparity.num_disparities", 16 * 10)
        self.block_size = self.config.get("stereo.disparity.block_size", 5)

        # Create stereo matcher
        self.stereo = self._create_stereo_matcher()

        logger.info(f"Disparity estimator initialized: {self.algorithm}")

    def _create_stereo_matcher(self):
        """Create OpenCV stereo matcher."""
        if self.algorithm == "SGBM":
            # Semi-Global Block Matching (better quality)
            stereo = cv2.StereoSGBM_create(
                minDisparity=self.min_disparity,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=8 * 3 * self.block_size ** 2,
                P2=32 * 3 * self.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        elif self.algorithm == "BM":
            # Block Matching (faster)
            stereo = cv2.StereoBM_create(
                numDisparities=self.num_disparities,
                blockSize=self.block_size
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return stereo

    def compute_disparity(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute disparity map from stereo pair.

        Args:
            left_image: Left image (H, W, 3) or (H, W)
            right_image: Right image (H, W, 3) or (H, W)
            normalize: Whether to normalize disparity to [0, 1]

        Returns:
            Disparity map (H, W)
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image

        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            right_gray = right_image

        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Remove invalid disparities
        disparity[disparity < 0] = 0

        # Normalize if requested
        if normalize:
            if disparity.max() > 0:
                disparity = disparity / disparity.max()

        return disparity

    def disparity_to_depth(
        self,
        disparity: np.ndarray,
        focal_length: float,
        baseline: float
    ) -> np.ndarray:
        """
        Convert disparity to depth.

        Args:
            disparity: Disparity map in pixels
            focal_length: Focal length in pixels
            baseline: Stereo baseline in meters

        Returns:
            Depth map in meters
        """
        # Avoid division by zero
        disparity = np.where(disparity > 0, disparity, 0.001)

        # Depth = (focal_length * baseline) / disparity
        depth = (focal_length * baseline) / disparity

        # Clip unrealistic depths
        depth = np.clip(depth, 0, 100)  # Max 100m

        return depth

    def compute_depth(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        focal_length: float,
        baseline: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both disparity and depth from stereo pair.

        Args:
            left_image: Left image
            right_image: Right image
            focal_length: Focal length in pixels
            baseline: Baseline in meters

        Returns:
            Tuple of (disparity, depth) maps
        """
        # Compute disparity (not normalized for depth conversion)
        disparity = self.compute_disparity(left_image, right_image, normalize=False)

        # Convert to depth
        depth = self.disparity_to_depth(disparity, focal_length, baseline)

        return disparity, depth

    def process_stereo_pair(
        self,
        left_path: str,
        right_path: str,
        camera_params: Dict
    ) -> Dict:
        """
        Process stereo image pair from files.

        Args:
            left_path: Path to left image
            right_path: Path to right image
            camera_params: Dictionary with focal_length, baseline, resolution

        Returns:
            Dictionary with disparity, depth, and metadata
        """
        # Load images
        left_image = np.array(Image.open(left_path))
        right_image = np.array(Image.open(right_path))

        # Get camera parameters
        focal_length_mm = camera_params.get("focal_length", 50)
        sensor_width_mm = camera_params.get("sensor_width", 36)
        baseline = camera_params.get("baseline", 0.065)
        resolution = camera_params.get("resolution", 512)

        # Convert focal length to pixels
        focal_length_px = (focal_length_mm / sensor_width_mm) * resolution

        # Compute disparity and depth
        disparity, depth = self.compute_depth(
            left_image,
            right_image,
            focal_length_px,
            baseline
        )

        result = {
            "disparity": disparity,
            "depth": depth,
            "disparity_min": float(disparity.min()),
            "disparity_max": float(disparity.max()),
            "depth_min": float(depth[depth > 0].min()) if (depth > 0).any() else 0,
            "depth_max": float(depth.max()),
            "camera_params": camera_params,
        }

        return result

    def save_disparity(self, disparity: np.ndarray, output_path: str):
        """
        Save disparity map as 16-bit PNG.

        Args:
            disparity: Disparity map
            output_path: Output file path
        """
        # Convert to 16-bit
        disparity_16 = (disparity * 256).astype(np.uint16)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), disparity_16)
        logger.debug(f"Saved disparity to {output_path}")

    def save_depth(self, depth: np.ndarray, output_path: str):
        """
        Save depth map as numpy array.

        Args:
            depth: Depth map in meters
            output_path: Output file path (.npy)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(output_path), depth)
        logger.debug(f"Saved depth to {output_path}")

    def visualize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Create visualization of disparity map.

        Args:
            disparity: Disparity map

        Returns:
            Color-mapped disparity image (H, W, 3)
        """
        # Normalize
        if disparity.max() > 0:
            disparity_norm = disparity / disparity.max()
        else:
            disparity_norm = disparity

        # Apply colormap
        disparity_vis = cv2.applyColorMap(
            (disparity_norm * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )

        return cv2.cvtColor(disparity_vis, cv2.COLOR_BGR2RGB)
