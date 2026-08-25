from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.assistant import AssistantService
from app.config import Settings
from app.documents import DocumentService

LOGGER = logging.getLogger(__name__)


@dataclass
class PendingDocument:
    token: str
    user_id: str
    file_name: str
    path: Path


@dataclass
class HandlerContext:
    settings: Settings
    assistant: AssistantService
    documents: DocumentService
    pending: dict[str, PendingDocument]


def build_router(context: HandlerContext) -> Router:
    router = Router(name="telegram_handlers")

    async def process_document(message: Message, pending: PendingDocument, version: int) -> None:
        await message.answer(
            "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…"
        )
        try:
            summary = await asyncio.to_thread(
                context.documents.ingest,
                pending.path,
                user_id=pending.user_id,
                file_name=pending.file_name,
                version=version,
            )
            await message.answer("Готово. Я изучил этот файл, теперь можем его обсудить.")
            await message.answer(summary)
        except Exception:
            LOGGER.exception("Ошибка обработки документа user_id=%s file=%s", pending.user_id, pending.file_name)
            await message.answer("Не удалось обработать файл. Проверьте его формат и попробуйте ещё раз.")
        finally:
            context.pending.pop(pending.token, None)
            try:
                pending.path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Не удалось удалить временный файл %s", pending.path)

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(
            "Привет! Я ассистент с памятью и поиском по загруженным документам. "
            "Можно спросить о погоде, попросить фотографию собаки или прислать PDF/DOCX/TXT/MD/HTML.\n\n"
            "Для удаления памяти используйте /forget_me."
        )

    @router.message(Command("forget_me", "clear_memory"))
    async def request_memory_clear(message: Message) -> None:
        await message.answer(
            "Будут удалены факты, журнал общения и документы. "
            "Если уверены, отправьте /forget_me_confirm."
        )
        context.pending[f"clear:{message.from_user.id}"] = PendingDocument(
            token=f"clear:{message.from_user.id}", user_id=str(message.from_user.id), file_name="", path=Path()
        )

    @router.message(Command("forget_me_confirm"))
    async def confirm_memory_clear(message: Message) -> None:
        token = f"clear:{message.from_user.id}"
        if token not in context.pending:
            await message.answer("Сначала отправьте /forget_me.")
            return
        context.pending.pop(token, None)
        deleted = await asyncio.to_thread(context.assistant.clear_user, str(message.from_user.id))
        deleted += await asyncio.to_thread(
            context.documents.store.delete,
            context.settings.document_collection,
            where={"user_id": str(message.from_user.id)},
        )
        await message.answer(f"Готово. Удалено записей: {deleted}.")

    @router.message(F.document)
    async def document(message: Message, bot: Bot) -> None:
        telegram_document = message.document
        file_name = telegram_document.file_name or "document"
        extension = Path(file_name).suffix.lower()
        if extension not in context.settings.supported_extensions:
            await message.answer("Поддерживаются только PDF, DOCX, TXT, MD и HTML-файлы.")
            return
        if telegram_document.file_size and telegram_document.file_size > context.settings.max_file_size_bytes:
            limit_mb = context.settings.max_file_size_bytes // (1024 * 1024)
            await message.answer(f"Файл слишком большой. Максимальный размер — {limit_mb} МБ.")
            return

        handle = await bot.get_file(telegram_document.file_id)
        fd, raw_path = tempfile.mkstemp(suffix=extension, prefix="telegram-document-")
        os.close(fd)
        path = Path(raw_path)
        await bot.download_file(handle.file_path, destination=path)
        user_id = str(message.from_user.id)
        existing = await asyncio.to_thread(context.documents.existing, user_id, file_name)
        if existing:
            token = uuid.uuid4().hex
            pending = PendingDocument(token, user_id, file_name, path)
            context.pending[token] = pending
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(text="Создать версию", callback_data=f"doc:version:{token}"),
                        InlineKeyboardButton(text="Заменить", callback_data=f"doc:replace:{token}"),
                    ]
                ]
            )
            await message.answer(
                f"Файл «{file_name}» уже загружался. Что сделать с новой копией?",
                reply_markup=keyboard,
            )
            return
        pending = PendingDocument(uuid.uuid4().hex, user_id, file_name, path)
        await process_document(message, pending, version=1)

    @router.callback_query(F.data.startswith("doc:"))
    async def document_action(callback: CallbackQuery) -> None:
        parts = (callback.data or "").split(":", 2)
        if len(parts) != 3:
            await callback.answer("Некорректное действие", show_alert=True)
            return
        action, token = parts[1], parts[2]
        pending = context.pending.get(token)
        if pending is None or callback.from_user.id != int(pending.user_id):
            await callback.answer("Это действие больше недоступно", show_alert=True)
            return
        if not isinstance(callback.message, Message):
            await callback.answer()
            return
        await callback.answer()
        if action == "replace":
            await asyncio.to_thread(context.documents.replace, pending.user_id, pending.file_name)
            version = 1
        elif action == "version":
            version = await asyncio.to_thread(
                context.documents.next_version, pending.user_id, pending.file_name
            )
        else:
            await callback.message.answer("Неизвестное действие")
            return
        await process_document(callback.message, pending, version)

    @router.message(F.text & ~F.text.startswith("/"))
    async def text(message: Message) -> None:
        user_text = (message.text or "").strip()
        if not user_text:
            return
        try:
            response, dog = await asyncio.to_thread(
                context.assistant.answer, str(message.from_user.id), user_text
            )
            if dog:
                await message.answer_photo(
                    BufferedInputFile(dog.image_bytes, filename="dog.jpg"),
                    caption=dog.description,
                )
            await message.answer(response)
        except Exception:
            LOGGER.exception("Ошибка обработки сообщения user_id=%s", message.from_user.id)
            await message.answer("Извините, при обработке сообщения произошла ошибка.")

    return router

