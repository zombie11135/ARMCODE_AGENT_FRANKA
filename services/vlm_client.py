# services/vlm_client.py

import base64
import mimetypes
from pathlib import Path
from typing import List

import requests


class QwenVLClient:
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

    def chat(
        self,
        text: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": text,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        return self._parse_response(response)

    def describe_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        return self.describe_images(
            image_paths=[image_path],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def describe_images(
        self,
        image_paths: List[str],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """
        多图 VDM：
        输入多张 RGB 图像，例如 global_color.png + wrist_color.png，
        让 VLM 综合判断当前机器人场景。
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for image_path in image_paths:
            image_base64 = self._encode_image_to_base64(image_path)
            mime_type = self._guess_mime_type(image_path)

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    },
                }
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        return self._parse_response(response)

    def _parse_response(self, response: requests.Response) -> str:
        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen-VL request failed. "
                f"status_code={response.status_code}, text={response.text}"
            )

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected Qwen-VL response format: {data}") from e

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts).strip()

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

        if mime_type is None:
            return "image/png"

        return mime_type