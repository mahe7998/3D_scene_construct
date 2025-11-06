"""Asset manager for querying and managing downloaded assets."""

from pathlib import Path
from typing import Dict, List, Optional
import trimesh

from src.utils.database import Database
from src.utils.logger import get_logger


logger = get_logger("asset_manager")


class AssetManager:
    """Manage and query downloaded 3D assets."""

    def __init__(self, db: Database = None):
        """
        Initialize asset manager.

        Args:
            db: Database instance
        """
        self.db = db or Database()

    def get_asset_by_id(self, object_id: str) -> Optional[Dict]:
        """Get asset by ID."""
        return self.db.get_object(object_id)

    def get_assets_by_category(self, category: str) -> List[Dict]:
        """Get all assets in a category."""
        return self.db.get_objects_by_category(category)

    def get_random_assets(self, count: int, category: Optional[str] = None) -> List[Dict]:
        """
        Get random assets.

        Args:
            count: Number of assets to return
            category: Optional category filter

        Returns:
            List of asset dictionaries
        """
        if category:
            assets = self.db.get_objects_by_category(category)
        else:
            assets = self.db.get_all_objects()

        import random
        return random.sample(assets, min(count, len(assets)))

    def load_mesh(self, object_id: str) -> Optional[trimesh.Trimesh]:
        """
        Load 3D mesh for an object.

        Args:
            object_id: Object ID

        Returns:
            Trimesh object or None
        """
        obj = self.get_asset_by_id(object_id)
        if not obj:
            logger.warning(f"Object {object_id} not found")
            return None

        file_path = Path("/data") / obj["file_path"]
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        try:
            mesh = trimesh.load(str(file_path), force="mesh")
            return mesh
        except Exception as e:
            logger.error(f"Error loading mesh {file_path}: {e}")
            return None

    def get_asset_info(self, object_id: str) -> Optional[Dict]:
        """
        Get detailed asset information including mesh statistics.

        Args:
            object_id: Object ID

        Returns:
            Asset info dictionary
        """
        obj = self.get_asset_by_id(object_id)
        if not obj:
            return None

        mesh = self.load_mesh(object_id)
        if mesh:
            obj["mesh_info"] = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
                "extents": mesh.extents.tolist(),
                "is_watertight": mesh.is_watertight,
            }

        return obj

    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories."""
        all_objects = self.db.get_all_objects()
        categories = set(obj.get("category", "unknown") for obj in all_objects)
        return sorted(list(categories))
