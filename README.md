# Telegram-бот-ассистент

Новая реализация использует `aiogram 3`, Haystack Agent, Docling, Yandex AI
Studio и ChromaDB. Папка `legacy` хранится отдельно как архив и справочник;
новый код её не импортирует.

## Возможности MVP

- текстовый ассистент с памятью пользователя;
- поиск по истории диалога и загруженным документам;
- погода через Open-Meteo;
- случайная собака через Dog CEO и описание Yandex VLM;
- документы PDF, DOCX, TXT, MD и HTML;
- разбор файлов Docling и сохранение чанков в `assistant_documents`;
- краткое резюме файла ровно одним предложением;
- выбор между новой версией и заменой при повторной загрузке;
- команды `/forget_me`, `/forget_me_confirm` и `/clear_memory`.

## Запуск локально

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```

В `.env` обязательны `TELEGRAM_BOT_TOKEN`, `YANDEX_API_KEY` и
`YANDEX_FOLDER_ID`. Поддерживаются старые имена конфигурации из архивного
проекта, включая `YANDEX_CLOUD_MODEL`, `CHROMA_DIR` и `TOP_K`.

## Запуск в Docker

```powershell
docker compose up --build
```

По умолчанию ограничение файла — 20 МБ, ограничение документа — 100 страниц.
Оба значения меняются через `.env`.
