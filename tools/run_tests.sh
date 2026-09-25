#!/usr/bin/env bash
# Hardware-free unit tests of every package (run from the workspace root, after
# `colcon build && source install/setup.bash`).
# Skips ament's style linters (test_copyright/flake8/pep257) and the scripts
# that need a real board (test_i2c_pca9685.py). Each package runs separately:
# same-named test modules in different packages would clash in one session.
set -uo pipefail
cd "$(dirname "$0")/.."

status=0
for dir in src/*/test; do
    files=$(find "$dir" -maxdepth 1 -name 'test_*.py' \
              ! -name test_copyright.py ! -name test_flake8.py ! -name test_pep257.py \
              ! -name test_i2c_pca9685.py | sort)
    [ -z "$files" ] && continue
    echo "== ${dir%/test}"
    python3 -m pytest -q -p no:warnings $files || status=1
done
exit $status
