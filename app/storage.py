from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import chromadb

LOGGER = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChromaStore:
    """Изолированный слой доступа к локальному ChromaDB."""

    def __init__(self, persist_directory: str, collections: Sequence[str]) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {
            name: self.client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
            for name in collections
        }

    @staticmethod
    def _metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
        result: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                result[key] = str(value)
        return result

    def add(
        self,
        collection: str,
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Добавляет записи в коллекцию и сообщает количество записанных строк."""

        if collection not in self.collections:
            raise KeyError(f"Неизвестная коллекция: {collection}")
        ids_list = list(ids or [uuid.uuid4().hex for _ in documents])
        self.collections[collection].add(
            ids=ids_list,
            documents=list(documents),
            embeddings=[list(item) for item in embeddings],
            metadatas=[self._metadata(item) for item in metadatas],
        )
        LOGGER.info("CHROMA_WRITE collection=%s records=%s", collection, len(ids_list))
        return ids_list

    def query(
        self,
        collection: str,
        embedding: Sequence[float],
        *,
        n_results: int,
        where: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Выполняет векторный поиск с необязательной фильтрацией metadata."""

        kwargs: dict[str, Any] = {
            "query_embeddings": [list(embedding)],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = dict(where)
        result = self.collections[collection].query(**kwargs)
        result["cosine_distances"] = result.get("distances")
        return result

    def get(
        self,
        collection: str,
        *,
        where: Mapping[str, Any] | None = None,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if where:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return self.collections[collection].get(**kwargs)

    def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> int:
        """Удаляет записи по ID или фильтру и возвращает количество удалённых строк."""

        target = self.collections[collection]
        if ids:
            target.delete(ids=list(ids))
            LOGGER.info("CHROMA_DELETE collection=%s records=%s", collection, len(ids))
            return len(ids)
        if where:
            records = target.get(where=dict(where), include=[])
            found = list(records.get("ids", []))
            if found:
                target.delete(ids=found)
            LOGGER.info("CHROMA_DELETE collection=%s records=%s", collection, len(found))
            return len(found)
        raise ValueError("Для удаления нужны ids или where")

    def add_texts(
        self,
        collection: str,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        return self.add(collection, texts, embeddings, metadatas)
