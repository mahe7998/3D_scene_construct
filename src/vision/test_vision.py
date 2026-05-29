"""Smoke-test the vision model: send one rendered image through vLLM."""

import argparse
import json
from pathlib import Path

from src.vision.vision_model import VisionModel
from src.utils.database import Database


def _find_rendered_image(db: Database) -> Path | None:
    """Return the first existing rendered left-eye view for any DB object."""
    for obj in db.get_all_objects():
        rendered_dir = Path("/data/assets/rendered") / obj["category"] / obj["id"]
        for name in ("view_001_L.jpg", "view_001.jpg"):
            candidate = rendered_dir / name
            if candidate.exists():
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Vision model smoke test")
    parser.add_argument("--image", type=str, help="Path to an image (overrides DB lookup)")
    args = parser.parse_args()

    model = VisionModel()

    # Fail fast if the vLLM server is not up / model not served.
    print("Probing vLLM server...")
    model.load_model()

    if args.image:
        image_path = Path(args.image)
    else:
        db = Database("/data/database/assets.db")
        image_path = _find_rendered_image(db)
        if image_path is None:
            print("No rendered images found. Run the renderer first.")
            return

    print(f"Sending image: {image_path}")
    result = model.describe_image(str(image_path))

    print("\n=== Model output ===")
    print(result["description"])

    if result["parsed"] is not None:
        print("\n=== Parsed JSON ===")
        print(json.dumps(result["parsed"], indent=2))
    else:
        print("\n(No JSON object could be parsed from the reply.)")

    if result.get("usage"):
        print(f"\nTokens: {result['usage']}")


if __name__ == "__main__":
    main()
