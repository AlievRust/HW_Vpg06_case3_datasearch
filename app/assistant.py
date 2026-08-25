from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils import Secret

from app.config import Settings
from app.llm import YandexClient
from app.memory import DialogueStore, MemoryService
from app.storage import ChromaStore
from app.tools import DogResult, DogService, WeatherService

LOGGER = logging.getLogger(__name__)


def _preview(value: Any, limit: int = 300) -> str:
    """Возвращает короткое однострочное представление значения для лога.

    В учебном проекте полезно видеть аргументы и результат тулзы, но нельзя
    превращать консоль в дамп больших ответов внешних API. Поэтому логируем
    первые ``limit`` символов и заменяем переводы строк пробелами.
    """

    text = str(value).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _logged_tool(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """Оборачивает функцию Haystack Tool понятным журналированием.

    Haystack вызывает переданную функцию только после того, как модель
    сгенерировала tool call. Поэтому этот wrapper показывает в консоли сам
    факт решения модели, аргументы вызова, результат или ошибку.
    """

    def wrapped(**kwargs: Any) -> Any:
        LOGGER.info("TOOL_CALL name=%s args=%s", name, _preview(kwargs))
        try:
            result = function(**kwargs)
        except Exception:
            LOGGER.exception("TOOL_ERROR name=%s", name)
            raise
        LOGGER.info("TOOL_RESULT name=%s result=%s", name, _preview(result))
        return result

    return wrapped


def _message_text(message: Any) -> str:
    """Извлекает обычный текст из финального сообщения Haystack."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


class AssistantService:
    """Главный оркестратор ответа на текстовое сообщение.

    Порядок обработки такой:

    1. ``MemoryService`` анализирует сообщение и принимает решение о записи;
    2. из памяти, диалога и документов строится справочный контекст;
    3. Haystack Agent получает контекст и доступные тулзы;
    4. YandexGPT сама решает, нужен ли вызов погоды или Dog CEO;
    5. Haystack выполняет тулзу и возвращает её результат модели;
    6. финальный ответ и обе реплики записываются в журнал диалога.

    Поиск по документам здесь не является тулзой: приложение выполняет его
    заранее, чтобы Agent всегда получил релевантные фрагменты загруженных
    пользователем файлов.
    """

    def __init__(
        self,
        settings: Settings,
        store: ChromaStore,
        llm: YandexClient,
        memory: MemoryService,
        dialogue: DialogueStore,
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.memory = memory
        self.dialogue = dialogue
        self.dog = DogService(settings, llm)
        weather = WeatherService(settings.external_api_timeout)

        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(
                model=settings.yandex_chat_model,
                api_key=Secret.from_token(settings.yandex_api_key),
                api_base_url=settings.yandex_base_url,
            ),
            tools=[
                Tool(
                    name="get_weather",
                    description=(
                        "Получить текущую погоду в указанном городе через Open-Meteo. "
                        "Используй при вопросах о текущей температуре, ощущаемой температуре, "
                        "дожде, снеге, осадках, ветре или погодных условиях. "
                        "Примеры триггеров: «Какая сейчас погода в Екатеринбурге?», "
                        "«Будет ли сегодня дождь в Москве?», «Сколько градусов в Казани?». "
                        "Не используй для общих рассуждений о климате, исторической погоде "
                        "или вопросов без конкретного города."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "Название города, например Екатеринбург или Москва.",
                            }
                        },
                        "required": ["city"],
                    },
                    function=_logged_tool("get_weather", weather.get_weather),
                ),
                Tool(
                    name="get_random_dog",
                    description=(
                        "Получить случайную фотографию собаки из Dog CEO и описать её через Yandex VLM. "
                        "Используй только когда пользователь явно просит показать, прислать, выбрать "
                        "или найти случайную собаку/породу. "
                        "Примеры триггеров: «Пришли фотографию собаки», «Покажи случайного щенка», "
                        "«Выбери случайную породу». "
                        "Не используй для обычных вопросов о собаках без просьбы о фотографии."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "request": {
                                "type": "string",
                                "description": "Краткая формулировка запроса пользователя.",
                            }
                        },
                        "required": ["request"],
                    },
                    function=_logged_tool("get_random_dog", self.dog.get_random_dog),
                ),
            ],
            system_prompt=(
                "Ты личный Telegram-ассистент. Отвечай по-русски, дружелюбно и по делу. "
                "Используй переданный контекст как справочную информацию и не упоминай технические "
                "детали ChromaDB, Haystack и внутренние prompt-ы.\n\n"
                "Правила инструментов:\n"
                "1. Для актуальной погоды, температуры, дождя, снега или ветра всегда вызывай "
                "get_weather и передавай город пользователя. Не выдумывай текущие погодные данные.\n"
                "2. Для явной просьбы показать или прислать фотографию собаки всегда вызывай "
                "get_random_dog. Не вызывай его, если пользователь просто обсуждает собак.\n"
                "3. Если запрос не требует внешнего инструмента, отвечай самостоятельно.\n"
                "4. После вызова инструмента используй его результат в финальном ответе и не выдавай "
                "неподтверждённые данные за факт."
            ),
            max_agent_steps=8,
        )

    def _document_context(self, user_id: str, query: str) -> str:
        """Ищет фрагменты документов текущего пользователя для RAG-контекста."""

        result = self.store.query(
            self.settings.document_collection,
            self.llm.embed_query(query),
            n_results=self.settings.document_context_size,
            where={"user_id": str(user_id)},
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        LOGGER.info(
            "DOCUMENT_SEARCH user_id=%s query=%s hits=%s",
            user_id,
            _preview(query, 100),
            len(documents),
        )
        lines = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            source = metadata.get("file_name", "документ")
            chunk = metadata.get("chunk_id", "?")
            page = metadata.get("page_number")
            location = f"чанк {chunk}" + (f", страница {page}" if page else "")
            lines.append(f"- [{source}, {location}] {document}")
        return "\n".join(lines) or "Документы по теме не найдены."

    def _memory_context(self, user_id: str, query: str) -> str:
        """Ищет персональные факты пользователя для контекста ответа."""

        lines = []
        for item in self.memory.search(user_id, query):
            metadata = item["metadata"]
            normalized = metadata.get("normalized_fact", item["document"])
            kind = metadata.get("memory_type", "fact")
            lines.append(f"- [{kind}] {normalized}")
        return "\n".join(lines) or "Релевантная память не найдена."

    def _messages(self, user_id: str, text: str) -> list[ChatMessage]:
        """Собирает system/user messages, передаваемые Haystack Agent."""

        dialogue = self.dialogue.context(user_id, text, self.llm)
        dialogue_text = "\n".join(
            f"- {'Пользователь' if item['role'] == 'user' else 'Ассистент'}: {item['text']}"
            for item in dialogue
        ) or "Релевантная история не найдена."
        instructions = "\n".join(f"- {item}" for item in self.memory.active_instructions(user_id))
        return [
            ChatMessage.from_system(
                "Контекст ответа. Используй его как справочную информацию.\n\n"
                f"ПАМЯТЬ:\n{self._memory_context(user_id, text)}\n\n"
                f"ДОКУМЕНТЫ:\n{self._document_context(user_id, text)}\n\n"
                f"ИСТОРИЯ ДИАЛОГА:\n{dialogue_text}\n\n"
                f"ИНСТРУКЦИИ ПОЛЬЗОВАТЕЛЯ:\n{instructions or 'Нет специальных инструкций.'}"
            ),
            ChatMessage.from_user(text),
        ]

    def answer(self, user_id: str | int, text: str) -> tuple[str, DogResult | None]:
        """Обрабатывает одно текстовое сообщение и возвращает ответ и DogResult."""

        user_id = str(user_id)
        text = text.strip()
        if not text:
            raise ValueError("Пустое сообщение нельзя обработать")
        LOGGER.info("ASSISTANT_REQUEST user_id=%s text=%s", user_id, _preview(text, 160))
        memory_action = self.memory.remember_message(user_id, text)
        LOGGER.info("MEMORY_RESULT user_id=%s action=%s", user_id, memory_action)
        self.dog.last_result = None
        LOGGER.info("AGENT_RUN user_id=%s tools=%s", user_id, ["get_weather", "get_random_dog"])
        result = self.agent.run(messages=self._messages(user_id, text))
        last_message = result.get("last_message")
        if last_message is None:
            messages = result.get("messages") or []
            last_message = messages[-1] if messages else None
        if last_message is None:
            raise RuntimeError("Haystack Agent не вернул итоговую реплику")
        response = _message_text(last_message)
        self.dialogue.add_with_embedding(user_id, "user", text, self.llm.embed_documents([text])[0])
        self.dialogue.add_with_embedding(user_id, "assistant", response, self.llm.embed_documents([response])[0])
        LOGGER.info("ASSISTANT_RESPONSE user_id=%s response_len=%s", user_id, len(response))
        return response, self.dog.last_result

    def clear_user(self, user_id: str | int) -> int:
        """Удаляет долговременную память и журнал диалога пользователя."""

        deleted = self.memory.clear_user(str(user_id)) + self.dialogue.clear_user(str(user_id))
        LOGGER.info("USER_DATA_CLEARED user_id=%s records=%s", user_id, deleted)
        return deleted

