"""
Vision Model API Server

Provides REST API for vision model inference.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import tempfile

from src.vision.vision_model import VisionModel
from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("vision_server")

app = FastAPI(title="Vision Model API")

# Global model instance
vision_model = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global vision_model
    config = load_config()
    vision_model = VisionModel(config)
    vision_model.load_model()
    logger.info("Vision model server started")


@app.post("/describe")
async def describe_image(file: UploadFile = File(...), prompt: str = None):
    """
    Describe an uploaded image.

    Args:
        file: Image file
        prompt: Optional custom prompt

    Returns:
        Description JSON
    """
    if vision_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Generate description
        result = vision_model.describe_image(tmp_path, prompt)

        # Clean up
        Path(tmp_path).unlink()

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error describing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": vision_model is not None}


def main():
    """Run the API server."""
    port = 8000
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
