"""Tests for database management."""

import pytest
import tempfile
from pathlib import Path
from src.utils.database import Database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    yield db

    db.close()
    Path(db_path).unlink()


def test_add_object(temp_db):
    """Test adding an object to database."""
    object_id = temp_db.add_object(
        name="test_object",
        category="test",
        source="test_source",
        file_path="/path/to/file",
        metadata={"key": "value"},
    )

    assert object_id is not None

    # Retrieve object
    obj = temp_db.get_object(object_id)
    assert obj is not None
    assert obj["name"] == "test_object"
    assert obj["category"] == "test"
    assert obj["metadata"]["key"] == "value"


def test_get_objects_by_category(temp_db):
    """Test retrieving objects by category."""
    # Add multiple objects
    temp_db.add_object("obj1", "cat1", "src", "/path1")
    temp_db.add_object("obj2", "cat1", "src", "/path2")
    temp_db.add_object("obj3", "cat2", "src", "/path3")

    # Get category 1 objects
    cat1_objects = temp_db.get_objects_by_category("cat1")
    assert len(cat1_objects) == 2


def test_add_render(temp_db):
    """Test adding a render to database."""
    # Add object first
    object_id = temp_db.add_object("obj", "cat", "src", "/path")

    # Add render
    render_id = temp_db.add_render(
        object_id=object_id,
        view_id="001",
        image_path="/render/path.jpg",
        camera_params={"position": [0, 0, 5]},
        lighting={"type": "SUN"},
    )

    assert render_id is not None

    # Retrieve renders
    renders = temp_db.get_renders_by_object(object_id)
    assert len(renders) == 1
    assert renders[0]["view_id"] == "001"


def test_add_annotation(temp_db):
    """Test adding an annotation."""
    # Add object and render
    object_id = temp_db.add_object("obj", "cat", "src", "/path")
    render_id = temp_db.add_render(object_id, "001", "/render.jpg")

    # Add annotation
    annotation_id = temp_db.add_annotation(
        render_id=render_id,
        description="Test description",
        category="test_cat",
        attributes={"color": "red"},
        confidence=0.95,
    )

    assert annotation_id is not None

    # Retrieve annotations
    annotations = temp_db.get_annotations_by_render(render_id)
    assert len(annotations) == 1
    assert annotations[0]["description"] == "Test description"
    assert annotations[0]["confidence"] == 0.95
