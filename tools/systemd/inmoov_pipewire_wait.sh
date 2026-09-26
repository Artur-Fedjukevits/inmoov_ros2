#!/bin/bash
# inmoov_pipewire_wait.sh — инициализация Jabra Speak2 40 MS перед запуском робота
#
# Jabra используется в профиле output:analog-stereo+input:mono-fallback (НЕ pro-audio).
# pro-audio вызывает нестабильность microphone capture: source периодически
# суспендируется PipeWire и возвращает нули (флипание).
# output:analog-stereo+input:mono-fallback — стабильный нормальный микширующий
# путь, поддерживает одновременный capture и playback без блокировок.
#
# WirePlumber сохраняет этот профиль в default-profile state — скрипт ждёт
# готовности sink И source. Если source не появился (WirePlumber применил
# неверный профиль при гонке старта) — перезапускаем WirePlumber один раз.
# Это лечит типичную проблему после перезагрузки: Jabra sink готов, но
# комбинированный профиль ещё не применён → source отсутствует → mic = нули.
#
# Вызывается как ExecStartPre в inmoov.service. Всегда завершается с кодом 0.
# Логи: journalctl --user -u inmoov -f

MAX_WAIT=30
INTERVAL=1
JABRA_SOURCE_PATTERN="alsa_input.usb-QTIL_Jabra_Speak2_40_MS"
WP_RESTARTED=0

jabra_source_ready() {
    pactl list sources short 2>/dev/null \
        | grep -v "\.monitor" \
        | grep -q "$JABRA_SOURCE_PATTERN"
}

jabra_sink_ready() {
    local sink
    sink=$(pactl info 2>/dev/null | grep "^Default Sink:" | awk '{print $3}')
    [ -n "$sink" ] && [ "$sink" != "auto_null" ] && [[ "$sink" != "@"* ]]
}

echo "[pipewire_wait] Ожидаю Jabra sink+source (max ${MAX_WAIT}s)..."

for i in $(seq 1 $MAX_WAIT); do
    if jabra_sink_ready && jabra_source_ready; then
        echo "[pipewire_wait] Jabra sink+source готовы (${i}s)"

        jabra_idx=$(arecord -l 2>/dev/null | grep -i jabra | grep -oP 'card \K\d+' | head -1)
        if [ -n "$jabra_idx" ]; then
            amixer -c "$jabra_idx" set PCM 100% >/dev/null 2>&1 \
                && echo "[pipewire_wait] Jabra PCM volume = 100%" \
                || echo "[pipewire_wait] WARNING: PCM volume не установлен"
        fi

        profile=$(pactl list cards 2>/dev/null \
            | grep -A 100 "QTIL_Jabra" \
            | grep "Active Profile:" | head -1 \
            | awk -F': ' '{print $2}' | tr -d ' ')
        echo "[pipewire_wait] Jabra profile: $profile"
        exit 0
    fi

    # Sink есть, но source нет → WirePlumber применил неверный профиль.
    # Перезапускаем его один раз чтобы он применил state из default-profile.
    if jabra_sink_ready && ! jabra_source_ready && [ "$WP_RESTARTED" -eq 0 ]; then
        echo "[pipewire_wait] Jabra sink готов, но source отсутствует — перезапускаю WirePlumber (${i}s)"
        systemctl --user restart wireplumber
        WP_RESTARTED=1
        sleep 2
        continue
    fi

    sleep $INTERVAL
done

echo "[pipewire_wait] WARNING: Jabra source не появился за ${MAX_WAIT}s — продолжаю"
exit 0
