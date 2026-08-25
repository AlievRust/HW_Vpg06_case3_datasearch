from __future__ import annotations

import base64
from typing import Any, Sequence

from openai import OpenAI

from app.config import Settings


class YandexClient:
    """Минимальный клиент Yandex AI Studio через OpenAI-совместимый API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.yandex_api_key,
            base_url=settings.yandex_base_url,
            timeout=settings.external_api_timeout,
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.settings.yandex_chat_model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise RuntimeError("YandexGPT вернул пустой ответ")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.settings.yandex_embedding_model,
            input=list(texts),
        )
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.settings.yandex_query_embedding_model,
            input=[text],
        )
        return list(response.data[0].embedding)

    def describe_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.settings.yandex_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Опиши фотографию собаки по-русски. Оцени породу осторожно, "
                                "не выдавай визуальную гипотезу за достоверный факт."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        content = response.choices[0].message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise RuntimeError("Yandex VLM вернула пустое описание")

