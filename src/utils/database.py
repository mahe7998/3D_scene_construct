"""Database management utilities."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import uuid


class Database:
    """SQLite database manager for assets, renders, annotations, and scenes."""

    def __init__(self, db_path: str = "/data/database/assets.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Create database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Objects table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS objects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                source TEXT,
                file_path TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Renders table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS renders (
                id TEXT PRIMARY KEY,
                object_id TEXT NOT NULL,
                view_id TEXT,
                image_path TEXT,
                camera_params TEXT,
                lighting TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (object_id) REFERENCES objects(id)
            )
        """
        )

        # Annotations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                render_id TEXT NOT NULL,
                description TEXT,
                category TEXT,
                attributes TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (render_id) REFERENCES renders(id)
            )
        """
        )

        # Scenes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scenes (
                id TEXT PRIMARY KEY,
                complexity_level INTEGER,
                objects TEXT,
                environment TEXT,
                image_path TEXT,
                ground_truth TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indices
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_objects_category ON objects(category)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_renders_object_id ON renders(object_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotations_render_id ON annotations(render_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenes_complexity ON scenes(complexity_level)"
        )

        self.conn.commit()

    # Object operations
    def add_object(
        self,
        name: str,
        category: str,
        source: str,
        file_path: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Add a new object to the database."""
        object_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata) if metadata else None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO objects (id, name, category, source, file_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (object_id, name, category, source, file_path, metadata_json),
        )
        self.conn.commit()
        return object_id

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get object by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM objects WHERE id = ?", (object_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)
        return None

    def get_objects_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all objects in a category."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM objects WHERE category = ?", (category,))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_all_objects(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all objects."""
        cursor = self.conn.cursor()
        if limit:
            cursor.execute("SELECT * FROM objects LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM objects")
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Render operations
    def add_render(
        self,
        object_id: str,
        view_id: str,
        image_path: str,
        camera_params: Dict[str, Any] = None,
        lighting: Dict[str, Any] = None,
    ) -> str:
        """Add a new render to the database."""
        render_id = str(uuid.uuid4())
        camera_json = json.dumps(camera_params) if camera_params else None
        lighting_json = json.dumps(lighting) if lighting else None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO renders (id, object_id, view_id, image_path, camera_params, lighting)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (render_id, object_id, view_id, image_path, camera_json, lighting_json),
        )
        self.conn.commit()
        return render_id

    def get_renders_by_object(self, object_id: str) -> List[Dict[str, Any]]:
        """Get all renders for an object."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM renders WHERE object_id = ?", (object_id,))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Annotation operations
    def add_annotation(
        self,
        render_id: str,
        description: str,
        category: str = None,
        attributes: Dict[str, Any] = None,
        confidence: float = None,
    ) -> str:
        """Add a new annotation to the database."""
        annotation_id = str(uuid.uuid4())
        attributes_json = json.dumps(attributes) if attributes else None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO annotations (id, render_id, description, category, attributes, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (annotation_id, render_id, description, category, attributes_json, confidence),
        )
        self.conn.commit()
        return annotation_id

    def get_annotations_by_render(self, render_id: str) -> List[Dict[str, Any]]:
        """Get all annotations for a render."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM annotations WHERE render_id = ?", (render_id,))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Scene operations
    def add_scene(
        self,
        complexity_level: int,
        objects: List[Dict[str, Any]],
        environment: Dict[str, Any],
        image_path: str,
        ground_truth: Dict[str, Any] = None,
    ) -> str:
        """Add a new scene to the database."""
        scene_id = str(uuid.uuid4())
        objects_json = json.dumps(objects)
        environment_json = json.dumps(environment)
        ground_truth_json = json.dumps(ground_truth) if ground_truth else None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO scenes (id, complexity_level, objects, environment, image_path, ground_truth)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (scene_id, complexity_level, objects_json, environment_json, image_path, ground_truth_json),
        )
        self.conn.commit()
        return scene_id

    def get_scenes_by_complexity(self, complexity_level: int) -> List[Dict[str, Any]]:
        """Get all scenes at a complexity level."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM scenes WHERE complexity_level = ?", (complexity_level,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Utility methods
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary."""
        result = dict(row)
        # Parse JSON fields
        if "metadata" in result and result["metadata"]:
            result["metadata"] = json.loads(result["metadata"])
        if "camera_params" in result and result["camera_params"]:
            result["camera_params"] = json.loads(result["camera_params"])
        if "lighting" in result and result["lighting"]:
            result["lighting"] = json.loads(result["lighting"])
        if "attributes" in result and result["attributes"]:
            result["attributes"] = json.loads(result["attributes"])
        if "objects" in result and result["objects"]:
            result["objects"] = json.loads(result["objects"])
        if "environment" in result and result["environment"]:
            result["environment"] = json.loads(result["environment"])
        if "ground_truth" in result and result["ground_truth"]:
            result["ground_truth"] = json.loads(result["ground_truth"])
        return result

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
