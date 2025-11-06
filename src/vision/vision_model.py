"""
Vision LLM Model

Wrapper for LLaVA or other vision-language models for scene understanding.
"""

from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("vision_model")


class VisionModel:
    """Vision-Language Model for scene understanding."""

    def __init__(self, config=None):
        """
        Initialize vision model.

        Args:
            config: Configuration object
        """
        self.config = config or load_config()

        self.model_name = self.config.get("vision.model", "llava-1.6-vicuna-7b")
        self.model_path = self.config.get("vision.model_path", "/data/models/llava")
        self.batch_size = self.config.get("vision.batch_size", 8)
        self.max_length = self.config.get("vision.max_length", 512)
        self.temperature = self.config.get("vision.temperature", 0.7)

        self.model = None
        self.processor = None

        logger.info(f"Vision model initialized: {self.model_name}")

    def load_model(self):
        """Load the vision-language model."""
        try:
            # Import here to avoid loading during initialization
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading model on {device}...")

            # Load model and processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.model_path,
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def describe_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Generate description for an image.

        Args:
            image_path: Path to image
            prompt: Optional custom prompt

        Returns:
            Dictionary with description and metadata
        """
        if self.model is None:
            self.load_model()

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Use default prompt if not provided
        if prompt is None:
            prompt = self.config.get("vision.prompts.single_object", "Describe this image.")

        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate description
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
            )

        description = self.processor.decode(outputs[0], skip_special_tokens=True)

        return {
            "description": description,
            "image_path": image_path,
            "prompt": prompt,
        }

    def describe_scene(
        self,
        image_path: str,
    ) -> Dict[str, any]:
        """
        Generate detailed scene description.

        Args:
            image_path: Path to scene image

        Returns:
            Dictionary with scene description
        """
        prompt = self.config.get(
            "vision.prompts.scene_description",
            "Describe all objects in this 3D scene, their positions, and the environment."
        )

        return self.describe_image(image_path, prompt)

    def batch_describe(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None,
    ) -> List[Dict[str, any]]:
        """
        Generate descriptions for multiple images in batch.

        Args:
            image_paths: List of image paths
            prompts: Optional list of custom prompts

        Returns:
            List of description dictionaries
        """
        results = []
        for i, image_path in enumerate(image_paths):
            prompt = prompts[i] if prompts else None
            result = self.describe_image(image_path, prompt)
            results.append(result)

        return results


# For Python import compatibility
try:
    import torch
except ImportError:
    logger.warning("PyTorch not available - model loading will fail")
