"""
Asset Downloader

Downloads 3D assets from various sources (Objaverse, ShapeNet, etc.)
and organizes them in the local file system.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import objaverse
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.database import Database
from src.utils.logger import get_logger


logger = get_logger("asset_downloader")


class AssetDownloader:
    """Download and organize 3D assets from various sources."""

    def __init__(self, config=None, db=None):
        """
        Initialize asset downloader.

        Args:
            config: Configuration object
            db: Database instance
        """
        self.config = config or load_config()
        self.db = db or Database(self.config.get("database.path", "/data/database/assets.db"))

        # Get paths from config
        self.assets_raw_dir = Path(
            self.config.get("paths.assets_raw", "/data/assets/raw")
        )
        self.assets_raw_dir.mkdir(parents=True, exist_ok=True)

        # Get asset settings
        self.asset_count = self.config.get("assets.count", 1000)
        self.batch_size = self.config.get("assets.batch_size", 10)
        self.download_workers = self.config.get("assets.download_workers", 4)

        logger.info(f"Asset downloader initialized")
        logger.info(f"Target count: {self.asset_count}")
        logger.info(f"Output directory: {self.assets_raw_dir}")

    def download_objaverse_assets(self, count: Optional[int] = None) -> List[str]:
        """
        Download assets from Objaverse.

        Args:
            count: Number of assets to download (uses config if not specified)

        Returns:
            List of downloaded object IDs
        """
        count = count or self.asset_count
        logger.info(f"Downloading {count} assets from Objaverse...")

        try:
            # Get UIDs from Objaverse
            logger.info("Fetching Objaverse annotations...")
            annotations = objaverse.load_annotations()

            # Filter by categories if specified
            enabled_sources = [
                s for s in self.config.get("assets.sources", [])
                if s.get("enabled", True) and s.get("name") == "objaverse"
            ]

            if enabled_sources and "categories" in enabled_sources[0]:
                categories = enabled_sources[0]["categories"]
                logger.info(f"Filtering by categories: {categories}")
                # Note: Objaverse doesn't have built-in category filtering
                # We'll download diverse objects and filter later
                uids = list(annotations.keys())[:count * 2]  # Download more to filter
            else:
                uids = list(annotations.keys())[:count]

            # Download objects
            logger.info(f"Downloading {len(uids)} objects...")
            objects = objaverse.load_objects(
                uids=uids[:count],
                download_processes=self.download_workers
            )

            # Organize downloaded files
            logger.info("Organizing downloaded files...")
            downloaded_ids = []
            for uid, file_path in tqdm(objects.items(), desc="Organizing"):
                if file_path and os.path.exists(file_path):
                    metadata = annotations.get(uid, {})
                    category = self._infer_category(metadata)

                    # Create organized directory
                    object_dir = self.assets_raw_dir / category / uid
                    object_dir.mkdir(parents=True, exist_ok=True)

                    # Move file
                    dest_path = object_dir / Path(file_path).name
                    shutil.copy2(file_path, dest_path)

                    # Save metadata
                    metadata_path = object_dir / "metadata.json"
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    # Add to database
                    self.db.add_object(
                        name=metadata.get("name", uid),
                        category=category,
                        source="objaverse",
                        file_path=str(dest_path.relative_to(self.assets_raw_dir.parent)),
                        metadata=metadata,
                    )

                    downloaded_ids.append(uid)

                    if len(downloaded_ids) >= count:
                        break

            logger.info(f"Successfully downloaded {len(downloaded_ids)} assets")
            return downloaded_ids

        except Exception as e:
            logger.error(f"Error downloading Objaverse assets: {e}")
            raise

    def _infer_category(self, metadata: Dict) -> str:
        """
        Infer object category from metadata.

        Args:
            metadata: Object metadata dictionary

        Returns:
            Category string
        """
        # Try to extract category from metadata
        if "category" in metadata:
            return metadata["category"].lower()

        if "tags" in metadata and metadata["tags"]:
            # Use first tag as category
            return metadata["tags"][0].lower()

        if "name" in metadata:
            name = metadata["name"].lower()
            # Simple heuristics
            if any(word in name for word in ["car", "vehicle", "truck"]):
                return "vehicle"
            elif any(word in name for word in ["chair", "table", "furniture"]):
                return "furniture"
            elif any(word in name for word in ["human", "person", "character"]):
                return "human"
            elif any(word in name for word in ["animal", "creature", "pet"]):
                return "animal"
            elif any(word in name for word in ["building", "house", "structure"]):
                return "building"

        return "misc"

    def get_download_stats(self) -> Dict:
        """
        Get statistics about downloaded assets.

        Returns:
            Dictionary with download statistics
        """
        all_objects = self.db.get_all_objects()
        categories = {}
        for obj in all_objects:
            cat = obj.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_objects": len(all_objects),
            "categories": categories,
            "storage_path": str(self.assets_raw_dir),
        }


def main():
    """Command-line interface for asset downloader."""
    parser = argparse.ArgumentParser(description="Download 3D assets")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of assets to download (default: from config)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="objaverse",
        choices=["objaverse"],
        help="Asset source",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create downloader
    downloader = AssetDownloader(config=config)

    # Download assets
    if args.source == "objaverse":
        downloaded = downloader.download_objaverse_assets(count=args.count)
        logger.info(f"Downloaded {len(downloaded)} assets")

    # Print stats
    stats = downloader.get_download_stats()
    logger.info(f"Download statistics:")
    logger.info(f"  Total objects: {stats['total_objects']}")
    logger.info(f"  Categories: {stats['categories']}")
    logger.info(f"  Storage: {stats['storage_path']}")


if __name__ == "__main__":
    main()
