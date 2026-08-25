from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings
from app.llm import YandexClient
from app.storage import ChromaStore, utc_now

LOGGER = logging.getLogger(__name__)


class MemoryService:
    """Долговременная память пользователя с LLM-классификацией.

    Сервис не записывает каждое сообщение подряд. Сначала YandexGPT извлекает
    claims, затем каждый claim сравнивается с близкой записью в ChromaDB:

    - ``created`` — новая информация;
    - ``skipped`` — тот же факт уже известен или claim не прошёл порог;
    - ``updated`` — новое значение заменило состояние или инструкцию.

    Именно эти решения подробно выводятся в консоль через ``MEMORY_*``-логи.
    """

    MEMORY_KINDS = {"fact", "state", "instruction", "task", "note"}

    def __init__(self, store: ChromaStore, llm: YandexClient, settings: Settings) -> None:
        self.store = store
        self.llm = llm
        self.settings = settings

    @staticmethod
    def _json(raw: str) -> dict[str, Any]:
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start : end + 1] if start >= 0 and end > start else raw)

    def _analyze(self, text: str) -> list[dict[str, Any]]:
        """Извлекает из сообщения атомарные записи для долговременной памяти."""

        if text.strip().startswith("/") or text.strip().lower() in {
            "привет", "здравствуйте", "спасибо", "ок"
        }:
            return []
        prompt = (
            "Верни только JSON без markdown вида "
            '{"claims":[{"memory_kind":"fact|state|instruction|task|note",'
            '"subject":"...","predicate":"...","value":"...",'
            '"normalized_fact":"...","confidence":0.0}]}. '
            "Извлекай только устойчивые факты, состояния, инструкции, задачи и явные заметки. "
            "Не включай вопросы, приветствия и разовые просьбы."
        )
        try:
            payload = self._json(self.llm.generate(text, system_prompt=prompt))
            claims = []
            for claim in payload.get("claims", []):
                if not isinstance(claim, dict) or not claim.get("normalized_fact"):
                    continue
                kind = str(claim.get("memory_kind", "fact")).lower()
                claims.append(
                    {
                        "memory_kind": kind if kind in self.MEMORY_KINDS else "fact",
                        "subject": str(claim.get("subject", "")),
                        "predicate": str(claim.get("predicate", "")),
                        "value": str(claim.get("value", "")),
                        "normalized_fact": str(claim["normalized_fact"]),
                        "confidence": float(claim.get("confidence", 0)),
                    }
                )
            return claims
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError) as exc:
            LOGGER.warning("Не удалось разобрать анализ памяти: %s", exc)
            return []

    def _metadata(self, user_id: str, claim: dict[str, Any], *, updated: bool = False) -> dict[str, Any]:
        now = utc_now()
        metadata = {
            "user_id": user_id,
            "memory_type": claim["memory_kind"],
            "normalized_fact": claim["normalized_fact"],
            "semantic_text": claim["normalized_fact"],
            "status": "active",
            "updated_at": now,
        }
        if not updated:
            metadata["created_at"] = now
        for field in ("subject", "predicate", "value"):
            if claim.get(field):
                metadata[field] = claim[field]
        return metadata

    def _classify(self, old: dict[str, Any], new_fact: str) -> str:
        """Определяет, является ли новый claim повтором, заменой или новой записью."""

        try:
            payload = self._json(
                self.llm.generate(
                    f"СТАРОЕ: {old.get('metadata', {}).get('normalized_fact', '')}\nНОВОЕ: {new_fact}",
                    system_prompt=(
                        'Верни только JSON {"action":"same|contradiction|new"}. '
                        "same — тот же факт, contradiction — замена, new — самостоятельная информация."
                    ),
                )
            )
            action = payload.get("action", "new")
            normalized_action = action if action in {"same", "contradiction", "new"} else "new"
            LOGGER.info("MEMORY_CLASSIFY action=%s", normalized_action)
            return normalized_action
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError):
            LOGGER.info("MEMORY_CLASSIFY action=new reason=invalid_llm_response")
            return "new"

    @staticmethod
    def _candidate(result: dict[str, Any]) -> dict[str, Any] | None:
        ids = result.get("ids") or [[]]
        if not ids or not ids[0]:
            return None
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("cosine_distances") or [[]]
        return {
            "id": ids[0][0],
            "document": (documents[0] or [""])[0],
            "metadata": (metadatas[0] or [{}])[0],
            "distance": (distances[0] or [None])[0],
        }

    def remember_message(self, user_id: str | int, text: str) -> str:
        """Анализирует сообщение, принимает решение и записывает claims в ChromaDB."""

        user_id = str(user_id)
        extracted_claims = self._analyze(text)
        claims = [
            item
            for item in extracted_claims
            if item["confidence"] >= self.settings.memory_min_confidence
        ]
        LOGGER.info(
            "MEMORY_ANALYZE user_id=%s extracted=%s accepted=%s threshold=%s",
            user_id,
            len(extracted_claims),
            len(claims),
            self.settings.memory_min_confidence,
        )
        if not claims:
            LOGGER.info("MEMORY_DECISION user_id=%s action=skipped reason=no_accepted_claims", user_id)
            return "skipped"
        actions: list[str] = []
        for claim in claims:
            LOGGER.info(
                "MEMORY_CLAIM user_id=%s type=%s confidence=%.2f fact=%s",
                user_id,
                claim["memory_kind"],
                claim["confidence"],
                str(claim["normalized_fact"])[:120],
            )
            embedding = self.llm.embed_documents([claim["normalized_fact"]])[0]
            result = self.store.query(
                self.settings.memory_collection,
                embedding,
                n_results=self.settings.memory_search_candidates,
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"memory_type": claim["memory_kind"]},
                    ]
                },
            )
            candidate = self._candidate(result)
            LOGGER.info(
                "MEMORY_CANDIDATE user_id=%s type=%s found=%s distance=%s",
                user_id,
                claim["memory_kind"],
                candidate is not None,
                candidate["distance"] if candidate else None,
            )
            if candidate is None or candidate["distance"] is None or candidate["distance"] > self.settings.memory_max_distance:
                self.store.add(
                    self.settings.memory_collection,
                    [text], [embedding], [self._metadata(user_id, claim)],
                )
                actions.append("created")
                LOGGER.info(
                    "MEMORY_DECISION user_id=%s action=created reason=no_close_candidate type=%s",
                    user_id,
                    claim["memory_kind"],
                )
                continue
            action = self._classify(candidate, claim["normalized_fact"])
            if action == "same":
                actions.append("skipped")
                LOGGER.info("MEMORY_DECISION user_id=%s action=skipped reason=same_claim", user_id)
            elif action == "contradiction" and claim["memory_kind"] in {"state", "instruction"}:
                metadata = dict(candidate["metadata"])
                metadata.update(self._metadata(user_id, claim, updated=True))
                collection = self.store.collections[self.settings.memory_collection]
                collection.update(
                    ids=[candidate["id"]], documents=[text], embeddings=[embedding], metadatas=[metadata]
                )
                actions.append("updated")
                LOGGER.info(
                    "MEMORY_DECISION user_id=%s action=updated reason=contradiction type=%s",
                    user_id,
                    claim["memory_kind"],
                )
            else:
                self.store.add(
                    self.settings.memory_collection,
                    [text], [embedding], [self._metadata(user_id, claim)],
                )
                actions.append("created")
                LOGGER.info(
                    "MEMORY_DECISION user_id=%s action=created reason=new_claim type=%s",
                    user_id,
                    claim["memory_kind"],
                )
        final_action = "updated" if "updated" in actions else "created" if "created" in actions else "skipped"
        LOGGER.info("MEMORY_RESULT user_id=%s action=%s actions=%s", user_id, final_action, actions)
        return final_action

    def search(self, user_id: str, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Ищет релевантные записи памяти пользователя и сообщает размер выдачи."""

        result = self.store.query(
            self.settings.memory_collection,
            self.llm.embed_query(query),
            n_results=limit or self.settings.memory_context_size,
            where={"user_id": str(user_id)},
        )
        return _flatten_results(result)

    def active_instructions(self, user_id: str) -> list[str]:
        """Возвращает активные пользовательские инструкции в порядке свежести."""

        result = self.store.get(
            self.settings.memory_collection,
            where={"$and": [{"user_id": str(user_id)}, {"memory_type": "instruction"}]},
            include=["documents", "metadatas"],
        )
        values = []
        for document, metadata in zip(result.get("documents", []), result.get("metadatas", [])):
            if metadata.get("status", "active") == "active":
                values.append((str(metadata.get("updated_at", "")), str(metadata.get("normalized_fact", document))))
        return [item for _, item in sorted(values, reverse=True)]

    def clear_user(self, user_id: str) -> int:
        """Удаляет всю долговременную память конкретного пользователя."""

        return self.store.delete(self.settings.memory_collection, where={"user_id": str(user_id)})


def _flatten_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("cosine_distances") or [[]])[0]
    return [
        {
            "document": document,
            "metadata": metadatas[index] if index < len(metadatas) else {},
            "distance": distances[index] if index < len(distances) else None,
        }
        for index, document in enumerate(documents)
        if document
    ]


class DialogueStore:
    """Хранилище полных реплик пользователя и ассистента в отдельной коллекции."""

    def __init__(self, store: ChromaStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def add(self, user_id: str, role: str, text: str) -> None:
        embedding = self.settings  # keeps the method dependency explicit
        del embedding
        # Dialogue is embedded with the same Yandex document model by the caller.

    def add_with_embedding(self, user_id: str, role: str, text: str, embedding: list[float]) -> None:
        """Записывает одну реплику и её роль в семантический журнал диалога."""

        self.store.add(
            self.settings.dialogue_collection,
            [text], [embedding],
            [{"user_id": str(user_id), "role": role, "memory_type": "dialogue_turn", "created_at": utc_now()}],
        )

    def context(self, user_id: str, query: str, llm: YandexClient) -> list[dict[str, str]]:
        """Возвращает похожие реплики для понимания контекста текущего вопроса."""

        relevant = self.store.query(
            self.settings.dialogue_collection,
            llm.embed_query(query),
            n_results=self.settings.dialogue_context_size,
            where={"user_id": str(user_id)},
        )
        items = _flatten_results(relevant)
        return [
            {"role": str(item["metadata"].get("role", "assistant")), "text": str(item["document"])}
            for item in items
        ]

    def clear_user(self, user_id: str) -> int:
        """Удаляет журнал диалога конкретного пользователя."""

        return self.store.delete(self.settings.dialogue_collection, where={"user_id": str(user_id)})
