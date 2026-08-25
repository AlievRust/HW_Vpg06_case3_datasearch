from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    yandex_api_key: str
    yandex_folder_id: str
    yandex_base_url: str
    yandex_chat_model: str
    yandex_embedding_model: str
    yandex_query_embedding_model: str
    yandex_vision_model: str
    chroma_directory: Path
    memory_collection: str
    dialogue_collection: str
    document_collection: str
    memory_context_size: int
    memory_search_candidates: int
    memory_max_distance: float
    memory_min_confidence: float
    dialogue_context_size: int
    dialogue_recent_turns: int
    external_api_timeout: float
    max_file_size_bytes: int
    max_document_pages: int
    document_chunk_size: int
    document_chunk_overlap: int
    document_context_size: int
    log_level: str
    supported_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".pdf", ".docx", ".txt", ".md", ".html", ".htm"})
    )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        folder_id = os.getenv("YANDEX_FOLDER_ID", "")
        chat_model = os.getenv("YANDEXGPT_MODEL") or os.getenv(
            "YANDEX_CLOUD_MODEL", f"gpt://{folder_id}/yandexgpt/latest"
        )
        embedding_model = os.getenv(
            "YANDEX_EMBEDDING_MODEL", f"emb://{folder_id}/text-search-doc/latest"
        )
        query_embedding_model = os.getenv(
            "YANDEX_QUERY_EMBEDDING_MODEL",
            embedding_model.replace("text-search-doc", "text-search-query"),
        )
        vision_model = os.getenv(
            "YANDEX_VISION_MODEL", f"gpt://{folder_id}/qwen3.6-35b-a3b"
        )
        directory = os.getenv("CHROMA_PERSIST_DIRECTORY") or os.getenv(
            "CHROMA_DIR", "data/chroma"
        )
        # Старый Docker-конфиг использует /app/data/chroma. На Windows такой
        # путь был бы абсолютным путем от корня диска, поэтому локально
        # направляем его в каталог проекта; в Linux/Docker оставляем как есть.
        if os.name == "nt" and directory.replace("\\", "/").startswith("/app/"):
            directory = "data/" + directory.replace("\\", "/").split("/app/", 1)[1].removeprefix("data/")
        max_file_size_mb = _int("MAX_FILE_SIZE_MB", 20)
        return cls(
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", ""),
            yandex_api_key=os.getenv("YANDEX_API_KEY", ""),
            yandex_folder_id=folder_id,
            yandex_base_url=os.getenv(
                "YANDEX_OPENAI_BASE_URL", "https://ai.api.cloud.yandex.net/v1"
            ),
            yandex_chat_model=chat_model,
            yandex_embedding_model=embedding_model,
            yandex_query_embedding_model=query_embedding_model,
            yandex_vision_model=vision_model,
            chroma_directory=Path(directory),
            memory_collection=os.getenv("CHROMA_COLLECTION", "assistant_memory"),
            dialogue_collection=os.getenv("CHROMA_DIALOGUE_COLLECTION", "assistant_dialogue"),
            document_collection=os.getenv("CHROMA_DOCUMENT_COLLECTION", "assistant_documents"),
            memory_context_size=_int("MEMORY_CONTEXT_SIZE", _int("TOP_K", 5)),
            memory_search_candidates=_int("MEMORY_SEARCH_CANDIDATES", 5),
            memory_max_distance=_float("MEMORY_MAX_DISTANCE", 0.8),
            memory_min_confidence=_float("MEMORY_MIN_CONFIDENCE", 0.7),
            dialogue_context_size=_int("DIALOGUE_CONTEXT_SIZE", 6),
            dialogue_recent_turns=_int("DIALOGUE_RECENT_TURNS", 6),
            external_api_timeout=_float("EXTERNAL_API_TIMEOUT", 20),
            max_file_size_bytes=max_file_size_mb * 1024 * 1024,
            max_document_pages=_int("MAX_DOCUMENT_PAGES", 100),
            document_chunk_size=_int("DOCUMENT_CHUNK_SIZE", 400),
            document_chunk_overlap=_int("DOCUMENT_CHUNK_OVERLAP", 50),
            document_context_size=_int("DOCUMENT_CONTEXT_SIZE", 5),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

    def validate(self) -> None:
        missing = []
        if not self.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.yandex_api_key:
            missing.append("YANDEX_API_KEY")
        if missing:
            raise RuntimeError(f"Не заданы обязательные переменные окружения: {', '.join(missing)}")
        if self.max_document_pages < 1 or self.max_file_size_bytes < 1:
            raise ValueError("Ограничения размера файла и числа страниц должны быть положительными")
