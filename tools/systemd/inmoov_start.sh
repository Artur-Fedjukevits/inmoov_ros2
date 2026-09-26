#!/bin/bash
# Wrapper для запуска InMoov через systemd.
# Явно sourcing ROS2 — systemd не читает ~/.bashrc.

set -e

export PATH="$HOME/.local/bin:$PATH"

source /opt/ros/jazzy/setup.bash
source "$(cd "$(dirname "$0")/../.." && pwd)/install/setup.bash"   # <workspace>/install

# Включить Telegram если задано в окружении
TELEGRAM_ARG="${INMOOV_TELEGRAM:-false}"

exec ros2 launch inmoov_bringup inmoov.launch.py \
    telegram:="${TELEGRAM_ARG}" \
    allowed_chat_id:="${TELEGRAM_ALLOWED_CHAT_ID:-0}"
