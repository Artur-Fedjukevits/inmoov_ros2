#!/usr/bin/env python3
"""
face_expression_calibrator.py — InMoov face expression calibrator.

Left panel: list of emotions — click one to load it.
Right panel: sliders for the 16 face servos, grouped.
  Checkbox "✓" = the servo is part of this expression (will be saved).
  Slider moves the servo on the robot in real time.

Saving: the "Сохранить" (Save) button writes the user calibration file
(face_expressions_node.USER_CALIB_FILE: $INMOOV_FACE_CALIBRATION or
~/.config/inmoov/face_expressions_calibration.json). face_expressions_node picks
it up on its next start; it survives rebuilds. Servo limits/rests are imported
from face_expressions_node, so they are defined in one place.

Run (ROS2 must be initialised):
  cd ~/ros2_ws && source install/setup.bash
  ros2 run inmoov_control face_expression_calibrator

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import math
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from inmoov_msgs.msg import JointCommand

from inmoov_control import face_expressions_node as _fen

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_FILE = _fen.USER_CALIB_FILE   # where Save writes
_LOAD_FILE       = _fen._CALIB_FILE       # user file if present, else the packaged one

# ─────────────────────────────────────────────────────────────────────────────
# Face servo definitions
# (name, min_deg, max_deg, rest_deg, label, group)
# Limits/rests: face_expressions_node (which mirrors InMoovLeft.ino / InMoovRight.ino)
# ─────────────────────────────────────────────────────────────────────────────

# (name, label, group); min/max/rest come from face_expressions_node (_MN/_MX/FACE_REST)
_SERVO_LABELS: list[tuple] = [
    ('eyelid_L_Upper',  'Eyelid L Upper',  'Веки'),
    ('eyelid_L_Lower',  'Eyelid L Lower',  'Веки'),
    ('eyelid_R_Upper',  'Eyelid R Upper',  'Веки'),
    ('eyelid_R_Lower',  'Eyelid R Lower',  'Веки'),
    ('eyebrow_L',       'Eyebrow Left',    'Брови'),
    ('eyebrow_R',       'Eyebrow Right',   'Брови'),
    ('cheek_L',         'Cheek Left',      'Щёки'),
    ('cheek_R',         'Cheek Right',     'Щёки'),
    ('forhead_L',       'Forehead Left',   'Лоб'),
    ('forhead_R',       'Forehead Right',  'Лоб'),
    ('eye_lr_L',        'Eye LR Left',     'Глаза'),
    ('eye_ud_L',        'Eye UD Left',     'Глаза'),
    ('eye_lr_R',        'Eye LR Right',    'Глаза'),
    ('eye_ud_R',        'Eye UD Right',    'Глаза'),
    ('upperLip',        'Upper Lip',       'Рот'),
    ('jaw',             'Jaw',             'Рот'),
]

SERVO_DEFS: list[tuple] = [
    (name, _fen._MN[name], _fen._MX[name], _fen.FACE_REST[name], label, group)
    for name, label, group in _SERVO_LABELS
]

SERVO_INFO   = {name: (mn, mx, rest, label, grp) for name, mn, mx, rest, label, grp in SERVO_DEFS}
FACE_REST    = {name: rest for name, _, _, rest, _, _ in SERVO_DEFS}
SERVO_LIMITS = {name: (mn, mx) for name, mn, mx, _, _, _ in SERVO_DEFS}
SERVO_NAMES  = [name for name, *_ in SERVO_DEFS]

# ─────────────────────────────────────────────────────────────────────────────
# Default expression positions (before calibration)
# Ported from face_expressions_node.py: _MIN → min_angle, _MAX → max_angle
# ─────────────────────────────────────────────────────────────────────────────


def _build_defaults() -> dict[str, dict[str, int]]:
    mn = {n: v[0] for n, v in SERVO_LIMITS.items()}
    mx = {n: v[1] for n, v in SERVO_LIMITS.items()}
    return {
        'neutral': {},
        'angry': {
            'forhead_L':        mx['forhead_L'],
            'forhead_R':        mx['forhead_R'],
            'eyelid_L_Upper':   70,
            'eyelid_L_Lower':   70,
            'eyelid_R_Upper':   70,
            'eyelid_R_Lower':   70,
            'upperLip':         mx['upperLip'],
            'cheek_L':          mn['cheek_L'],
            'cheek_R':          mn['cheek_R'],
            'eyebrow_L':        mn['eyebrow_L'],
            'eyebrow_R':        mn['eyebrow_R'],
        },
        'wink': {
            'eyelid_L_Upper':   mn['eyelid_L_Upper'],
            'eyelid_L_Lower':   mn['eyelid_L_Lower'],
        },
        'disgust': {
            'upperLip':         mx['upperLip'],
            'forhead_L':        mn['forhead_L'],
            'forhead_R':        mn['forhead_R'],
            'eyelid_L_Lower':   mn['eyelid_L_Lower'],
            'eyelid_R_Lower':   mn['eyelid_R_Lower'],
            'cheek_R':          mx['cheek_R'],
            'eyebrow_L':        mn['eyebrow_L'],
            'eyebrow_R':        mn['eyebrow_R'],
        },
        'fear': {
            'eyelid_L_Upper':   mx['eyelid_L_Upper'],
            'eyelid_L_Lower':   mx['eyelid_L_Lower'],
            'eyelid_R_Upper':   mx['eyelid_R_Upper'],
            'eyelid_R_Lower':   mx['eyelid_R_Lower'],
            'cheek_L':          mn['cheek_L'],
            'cheek_R':          mn['cheek_R'],
            'eyebrow_L':        mx['eyebrow_L'],
            'eyebrow_R':        mx['eyebrow_R'],
            'forhead_L':        mx['forhead_L'],
            'forhead_R':        mx['forhead_R'],
        },
        'happy': {
            'eyebrow_L':        mx['eyebrow_L'],
            'eyebrow_R':        mx['eyebrow_R'],
            'cheek_L':          mx['cheek_L'],
            'cheek_R':          mx['cheek_R'],
            'upperLip':         90,
            'eyelid_L_Lower':   mn['eyelid_L_Lower'],
            'eyelid_R_Lower':   mn['eyelid_R_Lower'],
            'jaw':              mx['jaw'],
        },
        'smile': {
            'cheek_L':          mx['cheek_L'],
            'cheek_R':          mx['cheek_R'],
            'eyelid_L_Lower':   mn['eyelid_L_Lower'],
            'eyelid_R_Lower':   mn['eyelid_R_Lower'],
            'jaw':              mx['jaw'],
        },
        'sad': {
            'eyelid_L_Upper':   mn['eyelid_L_Upper'],
            'eyelid_R_Upper':   mn['eyelid_R_Upper'],
            'cheek_L':          mn['cheek_L'],
            'cheek_R':          mn['cheek_R'],
        },
        'sigh': {
            'eye_lr_L':         mx['eye_lr_L'],
            'eye_ud_L':         mx['eye_ud_L'],
            'eye_lr_R':         mx['eye_lr_R'],
            'eye_ud_R':         mx['eye_ud_R'],
        },
        'sorry': {
            'eyebrow_L':        mx['eyebrow_L'],
            'eyebrow_R':        mx['eyebrow_R'],
            'cheek_L':          mn['cheek_L'],
            'cheek_R':          mn['cheek_R'],
        },
        'suspicious': {
            'upperLip':         mn['upperLip'],
            'forhead_R':        mx['forhead_R'],
            'forhead_L':        mn['forhead_L'],
            'cheek_L':          mx['cheek_L'],
            'eyebrow_R':        mn['eyebrow_R'],
            'eyebrow_L':        mx['eyebrow_L'],
            'eyelid_L_Upper':   mx['eyelid_L_Upper'],
            'eyelid_L_Lower':   mx['eyelid_L_Lower'],
            'eyelid_R_Upper':   70,
            'eyelid_R_Lower':   70,
        },
        'thinking': {
            'eyebrow_L':        mn['eyebrow_L'],
            'eyebrow_R':        mn['eyebrow_R'],
            'forhead_L':        mx['forhead_L'],
            'forhead_R':        mx['forhead_R'],
            'eyelid_L_Lower':   mn['eyelid_L_Lower'],
            'eyelid_R_Lower':   mn['eyelid_R_Lower'],
            'eye_lr_L':         mx['eye_lr_L'],
            'eye_ud_L':         mx['eye_ud_L'],
            'eye_lr_R':         mx['eye_lr_R'],
            'eye_ud_R':         mx['eye_ud_R'],
        },
        'unamused': {
            'eyebrow_R':        mn['eyebrow_R'],
            'forhead_L':        mx['forhead_L'],
            'forhead_R':        mn['forhead_R'],
            'eyelid_L_Upper':   70,
            'eyelid_L_Lower':   70,
            'eyelid_R_Upper':   70,
            'eyelid_R_Lower':   70,
            'eye_lr_L':         mn['eye_lr_L'],
            'eye_lr_R':         mn['eye_lr_R'],
            'cheek_L':          mn['cheek_L'],
            'cheek_R':          mn['cheek_R'],
        },
        'surprise': {
            'eyebrow_L':        mx['eyebrow_L'],
            'eyebrow_R':        mx['eyebrow_R'],
            'eyelid_L_Upper':   mx['eyelid_L_Upper'],
            'eyelid_R_Upper':   mx['eyelid_R_Upper'],
            'forhead_L':        mn['forhead_L'],
            'forhead_R':        mn['forhead_R'],
            'jaw':              mx['jaw'],
        },
        'sleeping': {
            'eyelid_L_Upper':   mn['eyelid_L_Upper'],
            'eyelid_L_Lower':   mn['eyelid_L_Lower'],
            'eyelid_R_Upper':   mn['eyelid_R_Upper'],
            'eyelid_R_Lower':   mn['eyelid_R_Lower'],
        },
    }


EXPRESSIONS_DEFAULTS = _build_defaults()
EXPRESSION_NAMES = list(EXPRESSIONS_DEFAULTS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# JSON — load / save
# ─────────────────────────────────────────────────────────────────────────────


def load_calibration() -> dict[str, dict[str, int]]:
    """Load the calibration JSON file; return defaults if it does not exist."""
    if os.path.exists(_LOAD_FILE):
        with open(_LOAD_FILE, encoding='utf-8') as f:
            data = json.load(f)
        # Merge: defaults as base, loaded values override
        merged = {expr: dict(EXPRESSIONS_DEFAULTS.get(expr, {})) for expr in EXPRESSION_NAMES}
        for expr, positions in data.items():
            if expr in merged:
                merged[expr] = positions
        return merged
    return {expr: dict(v) for expr, v in EXPRESSIONS_DEFAULTS.items()}


def save_calibration(data: dict[str, dict[str, int]]) -> None:
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────────────────────────────────────────


def _r(deg: float) -> float:
    return (deg - 90.0) * math.pi / 180.0


class FaceCalibNode(Node):
    def __init__(self):
        super().__init__('face_expression_calibrator')
        self._cmd_pub = self.create_publisher(JointCommand, '/joint_cmd', 10)

    def _publish(self, msg: JointState) -> None:
        # Calibration priority: overrides tracker/expressions while tuning (2 s lease)
        self._cmd_pub.publish(JointCommand(
            source='calibration', priority=JointCommand.PRIORITY_CALIBRATION,
            lease_sec=2.0, cmd=msg))

    def send_single(self, name: str, deg: int) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [name]
        msg.position = [_r(float(deg))]
        self._publish(msg)

    def send_batch(self, positions: dict[str, int]) -> None:
        if not positions:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(positions.keys())
        msg.position = [_r(float(v)) for v in positions.values()]
        self._publish(msg)

# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────


# Group colours
GROUP_COLORS = {
    'Веки':   '#4a6fa5',
    'Брови':  '#6a994e',
    'Щёки':   '#bc4749',
    'Лоб':    '#a7c957',
    'Глаза':  '#386641',
    'Рот':    '#c77dff',
}
ACTIVE_BG   = '#d6eaf8'   # background of an active row (part of the expression)
INACTIVE_BG = '#f5f5f5'   # background of an inactive row


class FaceExpressionCalibrator:
    def __init__(self, root: tk.Tk, node: FaceCalibNode):
        self.root = root
        self.node = node
        self.root.title('InMoov — Face Expression Calibrator')
        self.root.geometry('900x680')
        self.root.resizable(True, True)

        # State
        self._calib_data: dict[str, dict[str, int]] = load_calibration()
        self._current_expr: str = EXPRESSION_NAMES[0]
        self._auto_send    = tk.BooleanVar(value=True)

        # Per-servo widgets
        self._vars:       dict[str, tk.IntVar]      = {}  # current slider value
        self._checks:     dict[str, tk.BooleanVar]  = {}  # whether it is part of the expression
        self._val_labels: dict[str, tk.Label]       = {}
        self._rows:       dict[str, tk.Frame]       = {}  # row frame

        self._build_ui()
        self._load_expression(self._current_expr)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        top = ttk.Frame(self.root)
        top.pack(side='top', fill='x', padx=8, pady=(6, 0))

        ttk.Label(top, text='Face Expression Calibrator',
                  font=('', 13, 'bold')).pack(side='left')

        ttk.Checkbutton(top, text='Авто-отправка', variable=self._auto_send
                        ).pack(side='right', padx=6)
        ttk.Button(top, text='Отправить всё',
                   command=self._send_all).pack(side='right', padx=4)
        ttk.Button(top, text='Все → REST',
                   command=self._all_rest).pack(side='right', padx=4)

        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=8, pady=4)

        # ── Main area ─────────────────────────────────────────────────────────
        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=8)

        # Left column — emotion list
        left = ttk.Frame(main, width=150)
        left.pack(side='left', fill='y', padx=(0, 8))
        left.pack_propagate(False)

        ttk.Label(left, text='Эмоции', font=('', 10, 'bold')).pack(pady=(4, 2))

        self._expr_listbox = tk.Listbox(
            left, selectmode='single', activestyle='none',
            font=('', 10), width=14, exportselection=False,
            bg='#1e1e1e', fg='#ffffff', selectbackground='#3a86ff',
            selectforeground='#ffffff', relief='flat', borderwidth=1,
        )
        self._expr_listbox.pack(fill='both', expand=True)
        for name in EXPRESSION_NAMES:
            self._expr_listbox.insert('end', f'  {name}')
        self._expr_listbox.select_set(0)
        self._expr_listbox.bind('<<ListboxSelect>>', self._on_expr_select)

        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=6)

        ttk.Button(left, text='↩  REST все',
                   command=self._all_rest).pack(fill='x', pady=2)

        # Right column — sliders
        right = ttk.Frame(main)
        right.pack(side='left', fill='both', expand=True)

        # Header
        hdr = ttk.Frame(right)
        hdr.pack(fill='x', pady=(0, 2))
        tk.Label(hdr, text='✓',  width=3,  anchor='center', font=('', 9, 'bold')).pack(side='left')
        tk.Label(hdr, text='Servo',        width=16, anchor='w',      font=('', 9, 'bold')).pack(side='left')
        tk.Label(hdr, text='min',          width=4,  anchor='e',      font=('', 9, 'bold')).pack(side='left')
        tk.Label(hdr, text='Позиция',      width=36, anchor='center', font=('', 9, 'bold')).pack(side='left')
        tk.Label(hdr, text='max',          width=4,  anchor='w',      font=('', 9, 'bold')).pack(side='left')
        tk.Label(hdr, text='Знач',         width=5,  anchor='center', font=('', 9, 'bold')).pack(side='left')
        tk.Label(hdr, text='REST',         width=6,  anchor='center', font=('', 9, 'bold')).pack(side='left')
        ttk.Separator(right, orient='horizontal').pack(fill='x', pady=1)

        # Scrollable canvas
        canvas = tk.Canvas(right, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(right, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        self._slider_frame = ttk.Frame(canvas)
        canvas_win = canvas.create_window((0, 0), window=self._slider_frame, anchor='nw')

        def _on_frame_configure(_e):
            canvas.configure(scrollregion=canvas.bbox('all'))
        self._slider_frame.bind('<Configure>', _on_frame_configure)

        def _on_canvas_configure(e):
            canvas.itemconfig(canvas_win, width=e.width)
        canvas.bind('<Configure>', _on_canvas_configure)

        canvas.bind('<MouseWheel>',
                    lambda e: canvas.yview_scroll(-1 * (e.delta // 120), 'units'))
        canvas.bind('<Button-4>',
                    lambda e: canvas.yview_scroll(-1, 'units'))
        canvas.bind('<Button-5>',
                    lambda e: canvas.yview_scroll(1, 'units'))

        self._build_sliders()

        # ── Bottom bar ────────────────────────────────────────────────────────
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=8, pady=4)

        bot = ttk.Frame(self.root)
        bot.pack(side='bottom', fill='x', padx=8, pady=(0, 6))

        ttk.Button(bot, text='💾  Сохранить экспрессию',
                   command=self._save_expression).pack(side='left', padx=4)
        ttk.Button(bot, text='⟲  Сброс к defaults',
                   command=self._reset_to_defaults).pack(side='left', padx=4)
        ttk.Button(bot, text='📂  Показать файл',
                   command=self._show_file_path).pack(side='left', padx=4)

        self._status = tk.StringVar(value='Готов.')
        tk.Label(bot, textvariable=self._status, anchor='e',
                 font=('', 9), fg='#555').pack(side='right', padx=4)

    def _build_sliders(self):
        parent = self._slider_frame
        current_group = None

        for name, mn, mx, rest, label, group in SERVO_DEFS:
            # Group separator
            if group != current_group:
                current_group = group
                color = GROUP_COLORS.get(group, '#888')
                sep_frame = tk.Frame(parent, bg=color)
                sep_frame.pack(fill='x', pady=(6, 2))
                tk.Label(sep_frame, text=f'  {group}',
                         bg=color, fg='white',
                         font=('', 9, 'bold'), anchor='w').pack(fill='x')

            row = tk.Frame(parent, bg=INACTIVE_BG)
            row.pack(fill='x', pady=1, padx=2)
            self._rows[name] = row

            # Checkbox
            chk_var = tk.BooleanVar(value=False)
            self._checks[name] = chk_var
            chk = tk.Checkbutton(row, variable=chk_var, bg=INACTIVE_BG,
                                 activebackground=INACTIVE_BG,
                                 command=lambda n=name: self._on_check_toggle(n))
            chk.pack(side='left', padx=(4, 0))

            # Label
            tk.Label(row, text=label, width=16, anchor='w',
                     bg=INACTIVE_BG, font=('', 9)).pack(side='left')

            # Min label
            tk.Label(row, text=str(mn), width=3, anchor='e',
                     bg=INACTIVE_BG, fg='#888', font=('', 8)).pack(side='left', padx=(2, 0))

            # Slider
            var = tk.IntVar(value=rest)
            self._vars[name] = var
            slider = ttk.Scale(row, from_=mn, to=mx, orient='horizontal',
                               variable=var, length=260)
            slider.pack(side='left', padx=4, fill='x', expand=True)

            # Max label
            tk.Label(row, text=str(mx), width=3, anchor='w',
                     bg=INACTIVE_BG, fg='#888', font=('', 8)).pack(side='left', padx=(0, 2))

            # Value label
            val_lbl = tk.Label(row, text=f'{rest:3d}°', width=5,
                               bg=INACTIVE_BG, font=('', 9, 'bold'), fg='#222')
            val_lbl.pack(side='left', padx=2)
            self._val_labels[name] = val_lbl

            # REST button
            tk.Button(row, text=f'↩{rest}', width=5, relief='flat',
                      bg='#e8e8e8', activebackground='#d0d0d0',
                      font=('', 8),
                      command=lambda n=name, r=rest: self._set_servo(n, r)
                      ).pack(side='left', padx=(0, 4))

            # Trace slider
            def _on_slider(vname, idx, mode,
                           _var=var, _lbl=val_lbl, _name=name, _chk=chk_var):
                v = int(_var.get())
                _lbl.config(text=f'{v:3d}°')
                if self._auto_send.get():
                    self.node.send_single(_name, v)

            var.trace_add('write', _on_slider)

    # ── Expression logic ──────────────────────────────────────────────────────

    def _load_expression(self, expr_name: str):
        """Load an expression into the sliders and highlight the active servos."""
        self._current_expr = expr_name
        expr_data = self._calib_data.get(expr_name, {})
        active_count = 0

        for name in SERVO_NAMES:
            mn, mx, rest, _, _ = SERVO_INFO[name][0:5] if False else \
                                  (*SERVO_LIMITS[name], FACE_REST[name], None, None)
            mn, mx = SERVO_LIMITS[name]

            in_expr = name in expr_data
            value   = expr_data[name] if in_expr else FACE_REST[name]
            value   = max(mn, min(mx, value))

            self._checks[name].set(in_expr)
            self._vars[name].set(value)
            self._update_row_style(name, in_expr)

            if in_expr:
                active_count += 1

        n_active_str = f'{active_count} серво активно' if active_count else 'все серво → REST'
        self._status.set(f'{expr_name.upper()} — {n_active_str}')

        # Send the full state to the robot
        self._send_all()

    def _update_row_style(self, name: str, active: bool):
        bg = ACTIVE_BG if active else INACTIVE_BG
        row = self._rows[name]
        row.config(bg=bg)
        for child in row.winfo_children():
            try:
                child.config(bg=bg)
            except tk.TclError:
                pass

    def _on_check_toggle(self, name: str):
        self._update_row_style(name, self._checks[name].get())

    def _set_servo(self, name: str, deg: int):
        """Set the slider and send the command."""
        self._vars[name].set(deg)  # the trace callback calls send_single if auto_send is on

    def _send_all(self):
        """Send all current slider positions to the robot."""
        positions = {name: self._vars[name].get() for name in SERVO_NAMES}
        self.node.send_batch(positions)

    def _all_rest(self):
        """Reset all sliders to REST and clear the checkboxes."""
        for name in SERVO_NAMES:
            self._checks[name].set(False)
            self._update_row_style(name, False)
            self._vars[name].set(FACE_REST[name])

    # ── Save / Reset ──────────────────────────────────────────────────────────

    def _save_expression(self):
        """Save only the checked servos for the current expression."""
        expr = self._current_expr
        positions = {
            name: self._vars[name].get()
            for name in SERVO_NAMES
            if self._checks[name].get()
        }
        self._calib_data[expr] = positions
        save_calibration(self._calib_data)

        n = len(positions)
        self._status.set(f'✓ Сохранено: {expr} ({n} серво) → {os.path.basename(CALIBRATION_FILE)}')

    def _reset_to_defaults(self):
        """Reset the current expression to the MRL defaults."""
        expr = self._current_expr
        if messagebox.askyesno('Сброс', f'Сбросить "{expr}" к defaults (MRL)?'):
            self._calib_data[expr] = dict(EXPRESSIONS_DEFAULTS.get(expr, {}))
            self._load_expression(expr)
            self._status.set(f'↩ {expr} сброшено к defaults')

    def _show_file_path(self):
        messagebox.showinfo('Файл калибровки', CALIBRATION_FILE)

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_expr_select(self, _event=None):
        sel = self._expr_listbox.curselection()
        if not sel:
            return
        name = EXPRESSION_NAMES[sel[0]]
        if name != self._current_expr:
            self._load_expression(name)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = FaceCalibNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    root = tk.Tk()
    app = FaceExpressionCalibrator(root, node)   # noqa: F841
    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
