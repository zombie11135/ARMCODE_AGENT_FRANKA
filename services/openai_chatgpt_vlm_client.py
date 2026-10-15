import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class OpenAIChatGPTVLMClient:
    """
    Minimal OpenAI-compatible VLM client used by verifier skills.

    It mirrors QwenVLClient.describe_images but also accepts response_format for
    JSON-schema style outputs when the provider supports it.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def describe_images(
        self,
        image_paths: List[str],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        image_labels: Optional[List[str]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        content = [{"type": "text", "text": prompt}]
        for index, image_path in enumerate(image_paths):
            if image_labels and index < len(image_labels) and image_labels[index]:
                content.append({"type": "text", "text": f"Image {index + 1}: {image_labels[index]}"})
            image_base64 = self._encode_image_to_base64(image_path)
            mime_type = self._guess_mime_type(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                }
            )

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "OpenAI-compatible VLM request failed. "
                f"status_code={response.status_code}, text={response.text}"
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected VLM response format: {data}") from e

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts).strip()
        return str(content).strip()

    def chat(
        self,
        text: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": str(text)}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "OpenAI-compatible chat request failed. "
                f"status_code={response.status_code}, text={response.text}"
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected chat response format: {data}") from e
        return str(content).strip()

    @staticmethod
    def _encode_image_to_base64(image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _guess_mime_type(image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        return mime_type or "image/png"
