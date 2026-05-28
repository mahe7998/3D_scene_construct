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
        logger.info(f"Output directory: {self.assets_raw_dir}")

    def _load_all_uids(self) -> List[str]:
        """
        Return the full Objaverse UID list, caching it on disk after the first
        call. Avoids re-parsing all 160 annotation chunks on every run (~1 min
        even when fully cached) when we only need the list of keys.
        """
        cache_path = self.assets_raw_dir.parent / "cache" / "objaverse_uids.json"
        if cache_path.exists():
            logger.info(f"Loading cached UID list from {cache_path}")
            with open(cache_path) as f:
                return json.load(f)

        logger.info(
            "Fetching Objaverse annotations (first run; subsequent runs will "
            "use the cached UID list)..."
        )
        annotations = objaverse.load_annotations()
        uids = list(annotations.keys())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(uids, f)
        logger.info(f"Cached {len(uids)} UIDs to {cache_path}")
        return uids

    def download_objaverse_assets(
        self, count: Optional[int] = None, offset: int = 0
    ) -> List[str]:
        """
        Download assets from Objaverse.

        Args:
            count: Number of assets to download (uses config if not specified)
            offset: Skip this many UIDs before selecting

        Returns:
            List of downloaded object IDs
        """
        count = count or self.asset_count
        logger.info(f"Downloading {count} assets from Objaverse (offset={offset})...")

        try:
            all_uids = self._load_all_uids()
            uids = all_uids[offset:offset + count]

            # Only fetch metadata for the selected UIDs (much faster than full load).
            logger.info(f"Fetching annotations for {len(uids)} selected UIDs...")
            annotations = objaverse.load_annotations(uids=uids)

            logger.info(f"Downloading {len(uids)} objects...")
            objects = objaverse.load_objects(
                uids=uids,
                download_processes=self.download_workers,
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

                    # Add to database; use the Objaverse UID as the row's id so
                    # repeated downloads update in place rather than duplicate.
                    self.db.add_object(
                        name=metadata.get("name", uid),
                        category=category,
                        source="objaverse",
                        file_path=str(dest_path.relative_to(self.assets_raw_dir.parent)),
                        metadata=metadata,
                        object_id=uid,
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
        # Objaverse annotations store tags/categories as lists of
        # {"name": ..., "slug": ...} dicts, not bare strings.
        def _name_of(item):
            if isinstance(item, dict):
                return item.get("name") or item.get("slug")
            if isinstance(item, str):
                return item
            return None

        for key in ("category", "categories"):
            val = metadata.get(key)
            if isinstance(val, list) and val:
                name = _name_of(val[0])
            else:
                name = _name_of(val)
            if name:
                return name.lower()

        if metadata.get("tags"):
            name = _name_of(metadata["tags"][0])
            if name:
                return name.lower()

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
        "--offset",
        type=int,
        default=0,
        help="Skip this many UIDs before selecting (lets you pick different assets across runs)",
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
        downloaded = downloader.download_objaverse_assets(
            count=args.count, offset=args.offset
        )
        logger.info(f"Downloaded {len(downloaded)} assets")

    # Print stats
    stats = downloader.get_download_stats()
    logger.info(f"Download statistics:")
    logger.info(f"  Total objects: {stats['total_objects']}")
    logger.info(f"  Categories: {stats['categories']}")
    logger.info(f"  Storage: {stats['storage_path']}")


if __name__ == "__main__":
    main()
