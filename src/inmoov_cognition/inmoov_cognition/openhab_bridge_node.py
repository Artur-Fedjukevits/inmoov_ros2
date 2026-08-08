#!/usr/bin/env python3

"""
openhab_bridge_node.py
======================
Мост между OpenHAB и ROS2.

Загружает список устройств (тег ChatGPT) при старте и подписывается на
WebSocket-поток OpenHAB для получения обновлений состояния в реальном времени.

Публикует:
  /openhab_items  (String, JSON) — полный список с актуальными state.
                  Публикуется при старте и при каждом изменении состояния.
  /openhab_schema (String, JSON) — статичная схема: name/label/type/options.
                  Публикуется один раз при старте (и по таймеру для опоздавших подписчиков).
  /telegram_push  (String, JSON) — push-уведомления о критических сенсорах.
                  Rate limit: не чаще одного раза в ALERT_RATE_LIMIT_SEC (900 с = 15 мин)
                  на один сенсор. При нормализации — однократное ✅ сообщение.

Параметры:
  openhab_url       — базовый URL OpenHAB (default: http://192.168.10.118:8080)
  items_tag         — тег фильтрации Items (default: ChatGPT)
  reconnect_sec     — пауза перед переподключением WS (default: 15.0)
  schema_repeat_sec — интервал повтора публикации schema (default: 10.0)
  ws_ping_interval  — интервал WS ping keepalive в секундах (default: 30)

Мониторинг окружающей среды (→ Telegram push, НЕ в базу напоминаний):
  Температура: > 25°C или < 16°C
  Влажность:   > 70%  или < 30%
  CO₂:         > 1200 ppm
  VOC:         > 300 ppb
  Батарея:     < 20%
"""

import json
import threading
import time

import requests
import websocket
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import String

from inmoov_memory.openhab_alerts import (
    classify_sensor, parse_value, evaluate_threshold,
)

# ── Константы мониторинга ──────────────────────────────────────────────────────

ALERT_RATE_LIMIT_SEC = 900.0  # 15 минут между повторными алертами одного сенсора


class OpenHABBridgeNode(LifecycleNode):
    def __init__(self):
        super().__init__('openhab_bridge_node')
        self._items_pub     = None
        self._schema_pub    = None
        self._telegram_push = None
        self._timer         = None
        self._cache         = {}
        self._lock          = threading.Lock()
        self._stop_event    = threading.Event()
        self._ws            = None
        self._alerted_items = set()
        self._alert_last_sent = {}

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('openhab_url',       'http://192.168.10.118:8080')
        self._dp('items_tag',         'ChatGPT')
        self._dp('reconnect_sec',     15.0)
        self._dp('schema_repeat_sec', 10.0)
        self._dp('ws_ping_interval',  30)

        self.openhab_url        = self.get_parameter('openhab_url').value
        self.items_tag          = self.get_parameter('items_tag').value
        self.reconnect_sec      = self.get_parameter('reconnect_sec').value
        self._ping_interval     = self.get_parameter('ws_ping_interval').value
        self._schema_repeat_sec = self.get_parameter('schema_repeat_sec').value

        self._items_pub     = self.create_lifecycle_publisher(String, 'openhab_items',  10)
        self._schema_pub    = self.create_lifecycle_publisher(String, 'openhab_schema', 10)
        self._telegram_push = self.create_lifecycle_publisher(String, '/telegram_push', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._items_pub.on_activate(state)
        self._schema_pub.on_activate(state)
        self._telegram_push.on_activate(state)

        self._timer = self.create_timer(self._schema_repeat_sec, self._publish_schema)

        if self._load_items():
            self._publish_items()
            self._publish_schema()
            self._check_all_alerts()
        else:
            self.get_logger().error('Не удалось загрузить items из OpenHAB при старте')

        self._stop_event.clear()
        threading.Thread(target=self._ws_worker, daemon=True).start()

        self.get_logger().info(
            f'OpenHAB bridge активирован. URL: {self.openhab_url}, '
            f'tag: {self.items_tag}, устройств: {len(self._cache)}')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._stop_event.set()
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        self._items_pub.on_deactivate(state)
        self._schema_pub.on_deactivate(state)
        self._telegram_push.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._stop_event.set()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._stop_event.set()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._stop_event.set()
        return TransitionCallbackReturn.SUCCESS

    # ── Загрузка Items ─────────────────────────────────────────────────────────

    def _load_items(self) -> bool:
        url = f'{self.openhab_url}/rest/items?tags={self.items_tag}'
        try:
            r = requests.get(url, timeout=10.0)
            r.raise_for_status()
            with self._lock:
                self._cache.clear()
                for item in r.json():
                    self._cache[item['name']] = self._normalize(item)
            return True
        except Exception as e:
            self.get_logger().warn(f'Ошибка загрузки items: {e}')
            return False

    def _normalize(self, item: dict) -> dict:
        entry = {
            'name':   item['name'],
            'label':  item.get('label', item['name']),
            'type':   item['type'],
            'state':  item.get('state', 'NULL'),
            'groups': item.get('groupNames', []),
        }
        opts = item.get('stateDescription', {}).get('options', [])
        if opts:
            entry['options'] = [o['value'] for o in opts]
        return entry

    # ── WebSocket Worker ───────────────────────────────────────────────────────

    def _ws_worker(self):
        ws_url = (self.openhab_url
                  .replace('http://', 'ws://')
                  .replace('https://', 'wss://') + '/ws')

        while not self._stop_event.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                )
                self._ws.run_forever(ping_interval=self._ping_interval, ping_timeout=10)
            except Exception as e:
                if not self._stop_event.is_set():
                    self.get_logger().warn(f'WS: исключение в run_forever ({e})')

            if not self._stop_event.is_set():
                self.get_logger().info(f'WS: переподключение через {self.reconnect_sec}с...')
                time.sleep(self.reconnect_sec)
                if self._load_items():
                    self._publish_items()
                    self._check_all_alerts()

    def _on_ws_open(self, ws):
        self.get_logger().info('WS: подключён к OpenHAB')

    def _on_ws_message(self, ws, message):
        try:
            event = json.loads(message)
        except Exception:
            return

        if event.get('type') != 'ItemStateChangedEvent':
            return

        topic = event.get('topic', '')
        parts = topic.split('/')
        if len(parts) < 4 or parts[-1] != 'statechanged':
            return

        item_name = parts[2]
        try:
            payload   = json.loads(event.get('payload', '{}'))
            new_state = payload.get('value', '')
        except Exception:
            return

        with self._lock:
            if item_name not in self._cache:
                return
            if self._cache[item_name]['state'] == new_state:
                return
            self._cache[item_name]['state'] = new_state
            item_type = self._cache[item_name]['type']

        self.get_logger().info(f'WS: {item_name} → {new_state}')
        self._check_alert(item_name, item_type, new_state)
        self._publish_items()

    def _on_ws_error(self, ws, error):
        if not self._stop_event.is_set():
            self.get_logger().warn(f'WS: ошибка ({error})')

    def _on_ws_close(self, ws, close_status_code, close_msg):
        if not self._stop_event.is_set():
            self.get_logger().warn(
                f'WS: соединение закрыто (код={close_status_code}, msg={close_msg})'
            )

    # ── Мониторинг окружающей среды ────────────────────────────────────────────

    def _send_telegram(self, text: str):
        """Публикует push-уведомление в /telegram_push (→ telegram_bridge_node)."""
        msg = String()
        msg.data = json.dumps({'text': text}, ensure_ascii=False)
        self._telegram_push.publish(msg)

    def _check_all_alerts(self):
        """Проверяет все закэшированные items при старте и после переподключения."""
        with self._lock:
            snapshot = [(v['name'], v['type'], v['state']) for v in self._cache.values()]
        for name, typ, state in snapshot:
            self._check_alert(name, typ, state)

    def _check_alert(self, item_name: str, item_type: str, state_str: str):
        """Проверяет один item на критические пороги и отправляет Telegram push."""
        sensor_type = classify_sensor(item_name, item_type)
        if sensor_type is None:
            return

        # Switch BatteryLow: ON/OFF — не требует числового парсинга
        if sensor_type == 'battery_low':
            if state_str not in ('ON', 'OFF'):
                return  # NULL/UNDEF — пропускаем
            is_critical = (state_str == 'ON')
            if is_critical:
                raw = item_name[: item_name.lower().rfind('_batterylow')]
                alert_msg = f'Садится батарейка: {raw.replace("_", " ")}'
            else:
                alert_msg = ''
        else:
            # Числовые датчики
            value = parse_value(state_str)
            if value is None:
                self._alerted_items.discard(item_name)
                return
            is_critical, alert_msg = evaluate_threshold(item_name, sensor_type, value)

        if is_critical:
            self._alerted_items.add(item_name)
            now = time.time()
            last = self._alert_last_sent.get(item_name, 0.0)
            if now - last >= ALERT_RATE_LIMIT_SEC:
                self._alert_last_sent[item_name] = now
                self._send_telegram(f'⚠️ {alert_msg}')
                self.get_logger().warn(
                    f'ALERT [{item_name}]: {alert_msg} (значение: {state_str}) → Telegram'
                )
            else:
                self.get_logger().debug(
                    f'ALERT [{item_name}]: rate limit, пропускаем '
                    f'({int(ALERT_RATE_LIMIT_SEC - (now - last))}с до следующего)'
                )
        else:
            if item_name in self._alerted_items:
                self._alerted_items.discard(item_name)
                self._alert_last_sent.pop(item_name, None)
                with self._lock:
                    label = self._cache.get(item_name, {}).get('label', item_name)
                self._send_telegram(f'✅ {label}: значение вернулось к норме ({state_str})')
                self.get_logger().info(
                    f'ALERT СНЯТ [{item_name}]: значение вернулось к норме ({state_str})'
                )

    # ── Публикация ─────────────────────────────────────────────────────────────

    def _publish_items(self):
        with self._lock:
            items = list(self._cache.values())
        msg      = String()
        msg.data = json.dumps(items, ensure_ascii=False)
        self._items_pub.publish(msg)

    def _publish_schema(self):
        with self._lock:
            if not self._cache:
                return
            schema = []
            for it in self._cache.values():
                entry = {
                    'name':  it['name'],
                    'label': it['label'],
                    'type':  it['type'],
                }
                if 'options' in it:
                    entry['options'] = it['options']
                schema.append(entry)

        msg      = String()
        msg.data = json.dumps(schema, ensure_ascii=False)
        self._schema_pub.publish(msg)




def main():
    rclpy.init()
    node = OpenHABBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
