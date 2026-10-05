"""
gesture_store.py — validation and on-disk storage of gestures recorded in the
Android app (format "inmoov_gesture", version 1).

Pure Python (no rclpy), shared by urdf_bridge_node and the tests.

A gesture is one JSON file per (kind, name):

  <root>/<kind>/<name>.json        kind = "body" | "face"
  default root: ~/.config/inmoov/gestures  (or $INMOOV_GESTURES_DIR)

`positions` (URDF radians) is the source of truth; `servo_deg` is recomputed
here from the calibration table, so it always matches the current table and the
client's own copy is ignored. Saving a gesture with an existing name overwrites it.

Author: Artur Fedjukevits
Assisted by: Claude (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import math
import os
import re
import tempfile

FORMAT = 'inmoov_gesture'
VERSION = 1
KINDS = ('body', 'face')
MAX_BYTES = 64 * 1024
MAX_NAME_LEN = 64

GESTURES_DIR = os.path.expanduser(
    os.environ.get('INMOOV_GESTURES_DIR', '~/.config/inmoov/gestures'))

_BAD_CHARS = re.compile(r'[^\w\-]+', re.UNICODE)


class GestureError(ValueError):
    """The message is not a usable gesture."""


def safe_name(name: str) -> str:
    """File-system-safe stem: letters (any script), digits, '_' and '-' only."""
    stem = _BAD_CHARS.sub('_', name.strip()).strip('_')[:MAX_NAME_LEN]
    if not stem:
        raise GestureError(f'gesture name {name!r} has no usable characters')
    return stem


def parse(text: str, expected_kind: str) -> dict:
    """Validate a published JSON string; returns the gesture dict (normalised)."""
    if len(text.encode('utf-8')) > MAX_BYTES:
        raise GestureError(f'message larger than {MAX_BYTES} bytes')
    try:
        g = json.loads(text)
    except json.JSONDecodeError as e:
        raise GestureError(f'not valid JSON: {e}') from e
    if not isinstance(g, dict):
        raise GestureError('top level must be an object')
    if g.get('format') != FORMAT:
        raise GestureError(f'format must be "{FORMAT}"')
    if g.get('version') != VERSION:
        raise GestureError(f'unsupported version {g.get("version")!r}')
    if g.get('kind') != expected_kind:
        raise GestureError(f'kind {g.get("kind")!r} sent to the {expected_kind} topic')
    if g.get('units', 'urdf_rad') != 'urdf_rad':
        raise GestureError(f'unsupported units {g.get("units")!r}')
    if not isinstance(g.get('name'), str):
        raise GestureError('name must be a string')
    safe_name(g['name'])
    pos = g.get('positions')
    if not isinstance(pos, dict) or not pos:
        raise GestureError('positions must be a non-empty object')
    for k, v in pos.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise GestureError(f'position of {k!r} is not a finite number')
    return g


def add_servo_deg(g: dict, urdf_map) -> list:
    """Replace g['servo_deg'] with the table's view of g['positions'].
    Returns the joint names the table does not know (kept in `positions`)."""
    deg, unknown = {}, []
    for name, rad in g['positions'].items():
        j = urdf_map.by_urdf.get(name)
        if j is None:
            unknown.append(name)
        else:
            deg[j.servo] = round(j.urdf_to_servo(rad), 2)
    g['servo_deg'] = deg
    return unknown


def gesture_path(kind: str, name: str, root: str = GESTURES_DIR) -> str:
    return os.path.join(root, kind, safe_name(name) + '.json')


def save(g: dict, root: str = GESTURES_DIR) -> str:
    path = gesture_path(g['kind'], g['name'], root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(g, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)       # atomic: a reader never sees a half-written file
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def load(kind: str, name: str, root: str = GESTURES_DIR) -> dict:
    with open(gesture_path(kind, name, root), encoding='utf-8') as f:
        return json.load(f)


def list_gestures(kind: str, root: str = GESTURES_DIR) -> list:
    """Names (as the user entered them) of the saved gestures of this kind."""
    d = os.path.join(root, kind)
    names = []
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.json'):
                try:
                    with open(os.path.join(d, fn), encoding='utf-8') as f:
                        names.append(json.load(f).get('name', fn[:-5]))
                except (OSError, ValueError):
                    names.append(fn[:-5])
    return names
