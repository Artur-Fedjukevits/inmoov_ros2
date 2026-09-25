#!/usr/bin/env bash
# Installs the build + test dependencies into a ros:jazzy container (CI).
# Same steps as README "Build", non-interactive. Run from the workspace root.
set -euxo pipefail
cd "$(dirname "$0")/.."

export DEBIAN_FRONTEND=noninteractive
# Ubuntu 24.04 marks the system Python as externally managed; CI is a throwaway
# container, so pip (and rosdep's pip keys) may install into it.
export PIP_BREAK_SYSTEM_PACKAGES=1

apt-get update
apt-get install -y --no-install-recommends python3-pip python3-pytest git
rosdep update --rosdistro "$ROS_DISTRO"
# The pip-based rosdep keys (onnxruntime, insightface) are installed from
# requirements.txt below instead: insightface pulls numpy 2, and pip can't
# uninstall Debian's numpy 1.x — rosdep would abort half-way.
rosdep install --from-paths src --ignore-src -y --rosdistro "$ROS_DISTRO" \
    --skip-keys "python3-onnxruntime-pip python3-insightface-pip"

# Like the robot (where they live in ~/.local): pip packages go on top of the
# Debian ones instead of replacing them — pip can't uninstall Debian packages
# (numpy 1.x, PyYAML, …: "RECORD file not found"). --ignore-installed for all.
export PIP_IGNORE_INSTALLED=1
pip install "numpy>=2.0" scipy
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r src/inmoov_voice/requirements.txt \
            -r src/inmoov_vision/requirements.txt \
            -r src/inmoov_cognition/requirements.txt \
            -r src/inmoov_memory/requirements.txt
