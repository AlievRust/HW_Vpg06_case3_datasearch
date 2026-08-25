from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from app.assistant import AssistantService
from app.config import Settings
from app.documents import DocumentService
from app.llm import YandexClient
from app.memory import DialogueStore, MemoryService
from app.storage import ChromaStore
from app.telegram.handlers import HandlerContext, build_router


def build_application() -> tuple[Bot, Dispatcher]:
    settings = Settings.from_env()
    settings.validate()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings.chroma_directory.mkdir(parents=True, exist_ok=True)
    store = ChromaStore(
        str(settings.chroma_directory),
        [settings.memory_collection, settings.dialogue_collection, settings.document_collection],
    )
    llm = YandexClient(settings)
    memory = MemoryService(store, llm, settings)
    dialogue = DialogueStore(store, settings)
    assistant = AssistantService(settings, store, llm, memory, dialogue)
    documents = DocumentService(settings, store, llm)
    context = HandlerContext(settings, assistant, documents, pending={})
    bot = Bot(token=settings.telegram_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(context))
    return bot, dispatcher


async def main() -> None:
    bot, dispatcher = build_application()
    try:
        await dispatcher.start_polling(bot, skip_updates=True)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())

