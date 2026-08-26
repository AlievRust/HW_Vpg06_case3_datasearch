from __future__ import annotations

import base64
import logging
from typing import Any, Sequence

from openai import OpenAI

from app.config import Settings

LOGGER = logging.getLogger(__name__)


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
        """Строит document-векторы через Yandex в формате обычных float.

        Yandex AI Studio не принимает base64-формат embedding-ответа. Кроме
        того, для совместимости с legacy-клиентом каждый текст отправляется
        отдельным запросом как строковый ``input``, а не как массив token IDs.
        """

        if not texts:
            return []
        vectors: list[list[float]] = []
        for text in texts:
            LOGGER.info(
                "EMBEDDING_REQUEST kind=document model=%s text_len=%s",
                self.settings.yandex_embedding_model,
                len(text),
            )
            response = self.client.embeddings.create(
                model=self.settings.yandex_embedding_model,
                input=text,
                encoding_format="float",
            )
            vectors.append(list(response.data[0].embedding))
        LOGGER.info("EMBEDDING_RESULT kind=document vectors=%s", len(vectors))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Строит query-вектор в совместимом с Yandex float-формате."""

        LOGGER.info(
            "EMBEDDING_REQUEST kind=query model=%s text_len=%s",
            self.settings.yandex_query_embedding_model,
            len(text),
        )
        response = self.client.embeddings.create(
            model=self.settings.yandex_query_embedding_model,
            input=text,
            encoding_format="float",
        )
        LOGGER.info("EMBEDDING_RESULT kind=query vectors=1")
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
            temperature=0.3,
            # Для описания изображения глубокое рассуждение не требуется.
            # Иначе модель может потратить весь лимит на reasoning_content,
            # оставив обычное message.content пустым.
            reasoning_effort=self.settings.yandex_vision_reasoning_effort,
            max_tokens=700,
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                for block in content
            )
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise RuntimeError("Yandex VLM вернула пустое описание")
