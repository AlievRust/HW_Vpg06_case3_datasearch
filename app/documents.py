from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

from haystack import Document, Pipeline, component
from haystack.components.preprocessors import DocumentSplitter

from app.config import Settings
from app.llm import YandexClient
from app.storage import ChromaStore, utc_now

LOGGER = logging.getLogger(__name__)


@component
class DoclingConverterComponent:
    """Haystack-компонент, превращающий файл в структурированный Document.

    Docling отвечает за разбор PDF/DOCX/TXT/MD/HTML и экспортирует единый
    Markdown-текст. Служебные metadata файла прикрепляются до splitter, чтобы
    Haystack скопировал их во все последующие чанки.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @component.output_types(documents=list[Document])
    def run(
        self,
        source_path: str,
        file_name: str,
        user_id: str,
        document_id: str,
        version: int,
    ) -> dict[str, list[Document]]:
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(source_path)
        pages = getattr(result.document, "pages", None)
        if pages is not None and len(pages) > self.settings.max_document_pages:
            raise ValueError(
                f"Документ превышает лимит в {self.settings.max_document_pages} страниц"
            )
        content = result.document.export_to_markdown().strip()
        if not content:
            raise ValueError("Docling не извлёк текст из файла")
        document = Document(
            content=content,
            meta={
                "user_id": user_id,
                "file_name": file_name,
                "document_id": document_id,
                "version": version,
                "created_at": utc_now(),
            },
        )
        return {"documents": [document]}


@component
class YandexDocumentEmbedder:
    """Добавляет каждому Haystack Document embedding от Yandex."""

    def __init__(self, llm: YandexClient) -> None:
        self.llm = llm

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        embeddings = self.llm.embed_documents([document.content for document in documents])
        result = []
        for document, embedding in zip(documents, embeddings):
            result.append(Document(content=document.content, meta=document.meta, embedding=embedding))
        return {"documents": result}


@component
class ChromaDocumentWriter:
    """Записывает чанки с embeddings и обязательными metadata в ChromaDB."""

    def __init__(self, store: ChromaStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    @component.output_types(documents_written=int, texts=list[str])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        texts = []
        embeddings = []
        metadatas = []
        ids = []
        for index, document in enumerate(documents, start=1):
            metadata = dict(document.meta or {})
            metadata["chunk_id"] = index
            metadata.setdefault("page_number", metadata.get("page_number", ""))
            texts.append(document.content)
            embeddings.append(document.embedding or [])
            metadatas.append(metadata)
            ids.append(f"doc-{metadata['document_id']}-{index}-{uuid.uuid4().hex[:8]}")
        self.store.add(self.settings.document_collection, texts, embeddings, metadatas, ids)
        return {"documents_written": len(documents), "texts": texts}


class DocumentService:
    """Сервис загрузки документов и генерации однопредложного резюме.

    Pipeline намеренно собран явно и последовательно для учебного проекта:
    ``converter → splitter → embedder → writer``. Это позволяет увидеть, на
    каком этапе файл анализируется, режется, векторизуется и сохраняется.
    """

    def __init__(self, settings: Settings, store: ChromaStore, llm: YandexClient) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.pipeline = Pipeline()
        self.pipeline.add_component("converter", DoclingConverterComponent(settings))
        self.pipeline.add_component(
            "splitter",
            DocumentSplitter(
                split_by="word",
                split_length=settings.document_chunk_size,
                split_overlap=settings.document_chunk_overlap,
            ),
        )
        self.pipeline.add_component("embedder", YandexDocumentEmbedder(llm))
        self.pipeline.add_component("writer", ChromaDocumentWriter(store, settings))
        self.pipeline.connect("converter.documents", "splitter.documents")
        self.pipeline.connect("splitter.documents", "embedder.documents")
        self.pipeline.connect("embedder.documents", "writer.documents")

    def existing(self, user_id: str, file_name: str) -> list[dict[str, Any]]:
        result = self.store.get(
            self.settings.document_collection,
            where={"$and": [{"user_id": str(user_id)}, {"file_name": file_name}]},
            include=["documents", "metadatas"],
        )
        return [
            {"document": document, "metadata": metadata}
            for document, metadata in zip(result.get("documents", []), result.get("metadatas", []))
        ]

    def next_version(self, user_id: str, file_name: str) -> int:
        versions = [int(item["metadata"].get("version", 1)) for item in self.existing(user_id, file_name)]
        return max(versions, default=0) + 1

    def replace(self, user_id: str, file_name: str) -> int:
        return self.store.delete(
            self.settings.document_collection,
            where={"$and": [{"user_id": str(user_id)}, {"file_name": file_name}]},
        )

    def ingest(
        self,
        source_path: Path,
        *,
        user_id: str,
        file_name: str,
        version: int,
    ) -> str:
        document_id = uuid.uuid4().hex
        result = self.pipeline.run(
            {
                "converter": {
                    "source_path": str(source_path),
                    "file_name": file_name,
                    "user_id": str(user_id),
                    "document_id": document_id,
                    "version": version,
                }
            }
        )
        texts = result["writer"]["texts"]
        source = "\n".join(texts)
        return self.summarize(file_name, source)

    def summarize(self, file_name: str, content: str) -> str:
        raw = self.llm.generate(
            content[:14000],
            system_prompt=(
                "Сделай краткое резюме документа на русском языке. Верни ровно одно предложение, "
                "без списков, заголовков и кавычек. Укажи главную тему и ключевой вывод."
            ),
            temperature=0.2,
            max_tokens=180,
        )
        text = " ".join(raw.split())
        boundary = re.search(r"(?<=[.!?。！？])\s+", text)
        if boundary:
            text = text[: boundary.start()]
        text = text.rstrip(".!?。！？")
        return (text or "Документ обработан") + "."
