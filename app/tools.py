from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.llm import YandexClient

LOGGER = logging.getLogger(__name__)


class WeatherService:
    """Синхронный адаптер Open-Meteo для Haystack Tool.

    Сначала сервис переводит название города в координаты, затем получает
    текущие погодные показатели. Модель не должна угадывать погоду сама —
    актуальные значения появляются только после вызова этого сервиса.
    """

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def get_weather(self, city: str) -> dict[str, Any]:
        """Возвращает текущую погоду для одного конкретного города."""

        LOGGER.info("WEATHER_REQUEST city=%s", city)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                geo = client.get(
                    self.GEOCODING_URL,
                    params={"name": city, "count": 1, "language": "ru"},
                )
                geo.raise_for_status()
                results = geo.json().get("results") or []
                if not results:
                    LOGGER.info("WEATHER_RESULT city=%s status=not_found", city)
                    return {"error": f"Город не найден: {city}"}
                place = results[0]
                forecast = client.get(
                    self.FORECAST_URL,
                    params={
                        "latitude": place["latitude"],
                        "longitude": place["longitude"],
                        "current": (
                            "temperature_2m,apparent_temperature,precipitation,"
                            "rain,weather_code,wind_speed_10m"
                        ),
                        "timezone": "auto",
                    },
                )
                forecast.raise_for_status()
                current = forecast.json().get("current", {})
            result = {
                "city": place.get("name", city),
                "temperature": current.get("temperature_2m"),
                "feels_like": current.get("apparent_temperature"),
                "precipitation": current.get("precipitation"),
                "wind_speed": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
            }
            LOGGER.info("WEATHER_RESULT city=%s status=ok", result["city"])
            return result
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("WEATHER_ERROR city=%s error=%s", city, exc)
            return {"error": "Сервис погоды временно недоступен."}


@dataclass(frozen=True)
class DogResult:
    """Байты изображения и описание для последующей отправки в Telegram."""

    image_bytes: bytes
    description: str


class DogService:
    """Получает случайную собаку и передаёт изображение в Yandex VLM.

    Haystack получает компактный словарь с URL и описанием, а байты картинки
    сохраняются в ``last_result`` для Telegram-обработчика, который отправляет
    фотографию отдельным сообщением.
    """

    DOG_URL = "https://dog.ceo/api/breeds/image/random"

    def __init__(self, settings: Settings, vision: YandexClient) -> None:
        self.timeout = settings.external_api_timeout
        self.vision = vision
        self.last_result: DogResult | None = None

    def get_random_dog(self, request: str = "случайная собака") -> dict[str, str]:
        """Скачивает случайное изображение собаки и запрашивает его описание."""

        LOGGER.info("DOG_REQUEST request=%s", request)
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(self.DOG_URL)
                response.raise_for_status()
                image_url = str(response.json()["message"])
                image_response = client.get(image_url)
                image_response.raise_for_status()
                image_bytes = image_response.content
                mime_type = image_response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
            description = self.vision.describe_image(image_bytes, mime_type)
            self.last_result = DogResult(image_bytes=image_bytes, description=description)
            LOGGER.info("DOG_RESULT status=ok image_bytes=%s", len(image_bytes))
            return {"image_url": image_url, "description": description}
        except (httpx.HTTPError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            LOGGER.warning("DOG_ERROR error=%s", exc)
            return {"error": "Не удалось получить и описать фотографию собаки."}

