"""
WorkingMemory — слой рабочей памяти
====================================
Хранит текущее состояние только в RAM.
Обновляется из ROS2-топиков или напрямую.
Данные НЕ персистируются — при перезапуске сбрасываются.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


class WorkingMemory:
    """Текущее состояние мира и робота — всегда передаётся в LLM."""

    SEASONS = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }

    SEASON_RU = {
        "winter": "зима", "spring": "весна",
        "summer": "лето", "autumn": "осень",
    }

    DAY_RU = [
        "понедельник", "вторник", "среда", "четверг",
        "пятница", "суббота", "воскресенье",
    ]

    TIME_OF_DAY = [
        (5,  "night"),     # 00:00-04:59 → ночь
        (12, "morning"),   # 05:00-11:59 → утро
        (18, "afternoon"), # 12:00-17:59 → день
        (22, "evening"),   # 18:00-21:59 → вечер
        # >= 22 → stays default "night"
    ]

    def __init__(self) -> None:
        self.data: dict = {
            "time": {},
            "location": {
                "room": "unknown",
                "coordinates": [0.0, 0.0, 0.0],
                "landmark": "",
            },
            "robot_state": {
                "mode": "idle",          # idle | conversation | navigation | task
                "battery": 100,
                "current_task": None,
                "facing": None,          # имя человека, на которого смотрит робот
            },
            "environment": {
                "people_present": [],
                "ambient_noise": "normal",
                "lighting": "normal",
            },
        }
        self.refresh_time()

    # ------------------------------------------------------------------
    # Время
    # ------------------------------------------------------------------

    def refresh_time(self) -> None:
        """Обновляет временной блок по системным часам."""
        now = datetime.now()
        hour = now.hour
        season = self.SEASONS[now.month]

        tod = "night"
        for threshold, name in self.TIME_OF_DAY:
            if hour < threshold:
                tod = name
                break

        self.data["time"] = {
            "timestamp": now.isoformat(timespec="seconds"),
            "time_of_day": tod,
            "time_of_day_ru": {
                "night": "ночь", "morning": "утро",
                "afternoon": "день", "evening": "вечер",
            }.get(tod, tod),
            "day_of_week": now.strftime("%A"),
            "day_of_week_ru": self.DAY_RU[now.weekday()],
            "day": now.day,
            "month": now.month,
            "month_name": now.strftime("%B"),
            "year": now.year,
            "season": season,
            "season_ru": self.SEASON_RU[season],
        }

    # ------------------------------------------------------------------
    # Обновления из ROS2 / внешних источников
    # ------------------------------------------------------------------

    def update_location(self, room: str, landmark: str = "",
                        coordinates: Optional[list[float]] = None) -> None:
        self.data["location"]["room"] = room
        self.data["location"]["landmark"] = landmark
        if coordinates:
            self.data["location"]["coordinates"] = coordinates

    def update_robot_state(
        self,
        mode: Optional[str] = None,
        battery: Optional[int] = None,
        current_task: Optional[str] = None,
        facing: Optional[str] = None,
    ) -> None:
        s = self.data["robot_state"]
        if mode is not None:
            s["mode"] = mode
        if battery is not None:
            s["battery"] = battery
        if current_task is not None:
            s["current_task"] = current_task
        if facing is not None:
            s["facing"] = facing

    def update_environment(
        self,
        people: Optional[list[str]] = None,
        noise: Optional[str] = None,
        lighting: Optional[str] = None,
    ) -> None:
        e = self.data["environment"]
        if people is not None:
            e["people_present"] = people
        if noise is not None:
            e["ambient_noise"] = noise
        if lighting is not None:
            e["lighting"] = lighting

    # ------------------------------------------------------------------
    # Форматирование для системного промпта
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        """Компактный текст для вставки в system prompt."""
        self.refresh_time()
        t = self.data["time"]
        l = self.data["location"]
        s = self.data["robot_state"]
        e = self.data["environment"]

        people_str = ", ".join(e["people_present"]) if e["people_present"] else "никого"
        task_str = s["current_task"] if s["current_task"] else "нет активной задачи"
        facing_str = f", смотрю на {s['facing']}" if s["facing"] else ""

        return (
            f"Время: {t['timestamp']} ({t['day_of_week_ru']}, {t['time_of_day_ru']}, {t['season_ru']} {t['year']})\n"
            f"Расположение: {l['room']}"
            + (f" — {l['landmark']}" if l["landmark"] else "") + "\n"
            f"Режим: {s['mode']}{facing_str}, батарея: {s['battery']}%\n"
            f"Задача: {task_str}\n"
            f"Люди рядом: {people_str}"
        )
