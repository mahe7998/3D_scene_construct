"""
Vision LLM Model

Thin client for an OpenAI-compatible vision-language server (vLLM serving
Qwen2.5-VL). The heavy model runs in the `vllm` service with the weights
bind-mounted from the host (./data/models); this class only does HTTP, so it
carries no torch/transformers dependency and can run from any service image.
"""

import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("vision_model")


class VisionModel:
    """Vision-Language Model client (talks to a vLLM OpenAI-compatible server)."""

    def __init__(self, config=None):
        """
        Initialize the vision model client.

        Args:
            config: Configuration object
        """
        self.config = config or load_config()

        # Served model name (must match vLLM --served-model-name), not a path.
        self.model_name = self.config.get("vision.model", "qwen2.5-vl-7b")
        self.endpoint = self.config.get("vision.endpoint", "http://vllm:8000/v1").rstrip("/")
        self.api_key = self.config.get("vision.api_key", "EMPTY")
        self.batch_size = self.config.get("vision.batch_size", 8)
        self.max_length = self.config.get("vision.max_length", 512)
        # Default to deterministic decoding so the reward pipeline gets stable,
        # parseable output.
        self.temperature = self.config.get("vision.temperature", 0.0)
        self.timeout = self.config.get("vision.request_timeout", 180)

        logger.info(
            f"Vision client initialized: model={self.model_name} endpoint={self.endpoint}"
        )

    def load_model(self):
        """
        Compatibility shim: there is no in-process model to load. We just probe
        the vLLM server so callers (e.g. the FastAPI startup hook) fail fast if
        the server is not reachable.
        """
        url = f"{self.endpoint}/models"
        try:
            resp = requests.get(
                url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout
            )
            resp.raise_for_status()
            served = [m.get("id") for m in resp.json().get("data", [])]
            logger.info(f"Connected to vLLM. Served models: {served}")
            if self.model_name not in served:
                logger.warning(
                    f"Configured model '{self.model_name}' not in served list {served}"
                )
        except Exception as e:
            logger.error(f"Could not reach vLLM at {url}: {e}")
            raise

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Return a data: URI for the image (base64), inferring the MIME type."""
        suffix = Path(image_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Best-effort parse of a JSON object embedded in the model's reply."""
        # Strip ```json ... ``` fences if present.
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            # Fall back to the first {...} span.
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            candidate = brace.group(0) if brace else None
        if candidate is None:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def describe_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Generate a description for an image via the vLLM chat completions API.

        Args:
            image_path: Path to image
            prompt: Optional custom prompt

        Returns:
            Dictionary with the model's text, any parsed JSON, and metadata.
        """
        if prompt is None:
            prompt = self.config.get("vision.prompts.single_object", "Describe this image.")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._encode_image(image_path)},
                        },
                    ],
                }
            ],
            "max_tokens": self.max_length,
            "temperature": self.temperature,
        }

        url = f"{self.endpoint}/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        description = data["choices"][0]["message"]["content"]

        return {
            "description": description,
            "parsed": self._extract_json(description),
            "image_path": image_path,
            "prompt": prompt,
            "model": self.model_name,
            "usage": data.get("usage"),
        }

    def describe_scene(
        self,
        image_path: str,
    ) -> Dict[str, any]:
        """
        Generate a detailed scene description.

        Args:
            image_path: Path to scene image

        Returns:
            Dictionary with scene description
        """
        prompt = self.config.get(
            "vision.prompts.scene_description",
            "Describe all objects in this 3D scene, their positions, and the environment.",
        )

        return self.describe_image(image_path, prompt)

    def batch_describe(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None,
    ) -> List[Dict[str, any]]:
        """
        Generate descriptions for multiple images.

        Note: issued sequentially here, but vLLM batches concurrent requests
        server-side. For real throughput, fan these out concurrently.

        Args:
            image_paths: List of image paths
            prompts: Optional list of custom prompts

        Returns:
            List of description dictionaries
        """
        results = []
        for i, image_path in enumerate(image_paths):
            prompt = prompts[i] if prompts else None
            results.append(self.describe_image(image_path, prompt))
        return results
