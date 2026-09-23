# Telegram Bridge — InMoov

Удалённое управление роботом через Telegram. `/ask` и свободный текст проходят
через `llm_node` — все tool calls, умный дом и персональная память работают полностью.

## Возможности

| Команда | Действие |
|---------|----------|
| `/status` | Режим, человек перед роботом, CPU/RAM |
| `/photo` | Снимок с левой камеры (JPEG) |
| `/say текст` | TTS напрямую (без LLM) |
| `/ask текст` | llm_node — полный пайплайн (SmartHome, память, web_search) |
| `/wake` | Вывести из спящего режима |
| *любой текст* | Работает как `/ask` |

## Идентификация

Телеграм-пользователь идентифицируется по `telegram_id` из таблицы `persons` в `/home/artur/inmoov_memory.db`.
Поле заполняется вручную — `telegram_id` это числовой Telegram user_id (узнать через @userinfobot).

```sql
UPDATE persons SET telegram_id = 123456789 WHERE name = 'Артур';
```

Если `telegram_id` найден, LLM получает `person_ctx` (имя, заметки, meet_count) — робот
знает с кем общается и может сохранять факты в персональную память.

## Архитектура

```
Telegram (polling, asyncio, daemon thread)
   │
   ├─ /status   → /social_context + psutil
   ├─ /photo    → cv2.VideoCapture → JPEG
   ├─ /say      → queue → ROS timer → Speak ActionClient → tts_node
   ├─ /ask      → SQLite lookup telegram_id → /telegram_ask
   └─ /wake     → /robot_sleep False

ROS spin (main thread)
   ├─ /telegram_ask (out)       → llm_node
   ├─ /telegram_response (in)   ← llm_node (полный LLM + tool call ответ)
   ├─ /social_context (in)
   └─ /robot_sleep (in/out latched)

llm_node
   ├─ /telegram_ask (in)   → _query_llm (полный пайплайн)
   └─ /telegram_response (out) → telegram_bridge
```

## 1. Создание Telegram-бота

1. Открыть [@BotFather](https://t.me/BotFather) → `/newbot`
2. Скопировать токен вида `123456789:AAFxxxxxxxx`

## 2. Узнать chat_id

Написать [@userinfobot](https://t.me/userinfobot) — ответит вашим `id`.

## 3. Переменные окружения

```bash
# ~/.bashrc или ~/.zshrc
export TELEGRAM_BOT_TOKEN="123456789:AAFxxxxxxxx"
export TELEGRAM_ALLOWED_CHAT_ID="987654321"
source ~/.bashrc
```

> **Важно**: Токен только через env var, никогда в коде.

## 4. Привязать telegram_id к человеку в БД

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/home/artur/inmoov_memory.db')
# Посмотреть список людей:
for row in conn.execute('SELECT id, name, telegram_id FROM persons'):
    print(row)
# Привязать:
conn.execute('UPDATE persons SET telegram_id=987654321 WHERE name=\"Артур\"')
conn.commit()
conn.close()
print('done')
"
```

## 5. Запуск

```bash
# Вместе с полным стеком (lifecycle_manager поднимет мост в тире 6):
ros2 launch inmoov_bringup inmoov.launch.py \
  telegram:=true \
  allowed_chat_id:=$TELEGRAM_ALLOWED_CHAT_ID

# Только мост вручную (llm_node должен быть запущен отдельно):
ros2 run inmoov_cognition telegram_bridge_node --ros-args \
  --params-file src/inmoov_cognition/config/telegram_params.yaml
ros2 lifecycle set /telegram_bridge_node configure
ros2 lifecycle set /telegram_bridge_node activate
```

## 6. Сборка

```bash
cd ~/ros2_ws
colcon build --packages-select inmoov_cognition inmoov_memory
source install/setup.bash
```

## Безопасность

- Только `allowed_chat_id` получает ответы; все остальные молча игнорируются
- Если LLM занята активным диалогом с человеком перед роботом — Telegram получает `⚙️ занят`
- Polling — публичный IP не нужен
