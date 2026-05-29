"""Test stereo disparity computation on a rendered stereo pair."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.stereo.disparity import DisparityEstimator
from src.utils.database import Database


def main():
    # Get first rendered stereo pair from database
    db = Database("/data/database/assets.db")
    objects = db.get_all_objects()

    if not objects:
        print("No objects in database")
        return

    obj = objects[0]
    print(f"Testing with object: {obj['id']} ({obj['name']})")

    # Find a stereo pair in rendered directory
    rendered_dir = Path("/data/assets/rendered") / obj["category"] / obj["id"]
    left_path = rendered_dir / "view_001_L.jpg"
    right_path = rendered_dir / "view_001_R.jpg"

    if not left_path.exists() or not right_path.exists():
        print(f"Stereo pair not found at {rendered_dir}")
        print("Run: docker-compose run --rm renderer --mode assets --stereo")
        return

    print(f"Left image:  {left_path}")
    print(f"Right image: {right_path}")

    # Initialize disparity estimator
    estimator = DisparityEstimator()

    # Use default camera params (from stereo config)
    camera_params = {
        "focal_length": 50.0,
        "sensor_width": 36.0,
        "baseline": 0.065,
        "resolution": 512,
    }

    # Compute disparity and depth
    result = estimator.process_stereo_pair(
        str(left_path), str(right_path), camera_params
    )

    print(f"\nDisparity stats:")
    print(f"  Min: {result['disparity_min']:.2f} pixels")
    print(f"  Max: {result['disparity_max']:.2f} pixels")

    print(f"\nDepth stats:")
    print(f"  Min: {result['depth_min']:.3f} meters")
    print(f"  Max: {result['depth_max']:.3f} meters")

    # Save visualizations
    output_dir = rendered_dir / "stereo_test"
    output_dir.mkdir(exist_ok=True)

    disp_vis = estimator.visualize_disparity(result["disparity"])
    Image.fromarray(disp_vis).save(output_dir / "disparity.png")

    estimator.save_depth(result["depth"], str(output_dir / "depth.png"))

    print(f"\nVisualization saved to: {output_dir}")
    print(f"  - disparity.png (colored disparity map)")
    print(f"  - depth.png (depth map)")


if __name__ == "__main__":
    main()
