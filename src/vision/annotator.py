"""
Annotator

Processes rendered images and generates annotations using Vision LLM.
"""

import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.database import Database
from src.utils.logger import get_logger
from src.vision.vision_model import VisionModel


logger = get_logger("annotator")


class Annotator:
    """Annotate rendered images using Vision LLM."""

    def __init__(self, config=None, db=None, vision_model=None):
        """
        Initialize annotator.

        Args:
            config: Configuration object
            db: Database instance
            vision_model: VisionModel instance
        """
        self.config = config or load_config()
        self.db = db or Database(self.config.get("database.path", "/data/database/assets.db"))
        self.vision_model = vision_model or VisionModel(self.config)

        self.batch_size = self.config.get("vision.batch_size", 8)

        logger.info("Annotator initialized")

    def annotate_renders(self, object_ids: List[str] = None):
        """
        Annotate rendered images.

        Args:
            object_ids: Optional list of specific object IDs to annotate
        """
        # Get objects to annotate
        if object_ids:
            objects = [self.db.get_object(oid) for oid in object_ids]
        else:
            objects = self.db.get_all_objects()

        logger.info(f"Annotating {len(objects)} objects")

        for obj in tqdm(objects, desc="Annotating objects"):
            # Get all renders for this object
            renders = self.db.get_renders_by_object(obj["id"])

            for render in renders:
                # Check if already annotated
                existing = self.db.get_annotations_by_render(render["id"])
                if existing:
                    logger.debug(f"Render {render['id']} already annotated")
                    continue

                # Generate annotation
                image_path = Path("/data") / render["image_path"]
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue

                try:
                    result = self.vision_model.describe_image(str(image_path))

                    # Store annotation in database
                    self.db.add_annotation(
                        render_id=render["id"],
                        description=result["description"],
                        category=obj.get("category"),
                        attributes={"prompt": result["prompt"]},
                        confidence=1.0,
                    )

                    logger.debug(f"Annotated render {render['id']}")

                except Exception as e:
                    logger.error(f"Error annotating {render['id']}: {e}")

        logger.info("Annotation complete")


def main():
    """Command-line interface for annotator."""
    parser = argparse.ArgumentParser(description="Annotate rendered images")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--object-ids",
        type=str,
        nargs="+",
        help="Specific object IDs to annotate",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()
    if args.batch_size:
        config._config.setdefault("vision", {})["batch_size"] = args.batch_size

    # Create annotator
    annotator = Annotator(config=config)

    # Run annotation
    annotator.annotate_renders(object_ids=args.object_ids)


if __name__ == "__main__":
    main()
