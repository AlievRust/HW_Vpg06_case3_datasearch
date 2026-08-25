# Небольшой production-like образ для учебного синхронного Telegram-бота.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# bot.py — предыдущая реализация. Новый образ намеренно не содержит её:
# единственной точкой запуска является hay_bot.py.
COPY chroma_manager.py hay_bot.py ./

# Каталог data монтируется из docker-compose и содержит только локальную
# персистентную базу ChromaDB, а не секреты.
RUN mkdir -p /app/data

# Новая учебная версия на Haystack запускается отдельно от исходного bot.py.
CMD ["python", "hay_bot.py"]
