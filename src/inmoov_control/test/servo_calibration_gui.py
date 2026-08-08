#!/usr/bin/env python3
"""
InMoov Servo Calibration GUI — новый протокол.

Публикует sensor_msgs/JointState на:
  /joint_command  — все серво тела (26 joints)
  /face_command   — серво лица (13 joints)

Значения слайдеров в градусах (как в firmware).
Конвертация: rad = (deg - 90) * pi / 180  (center_deg=90 для всех серво).
"""

import math
import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ─────────────────────────────────────────────────────────────────────────────
# Определения серво: (joint_name, min_deg, max_deg, rest_deg)
# Имена должны совпадать с arduino_right_node.py и arduino_left_node.py
# ─────────────────────────────────────────────────────────────────────────────

SERVOS = {
    "Right Hand": [
        # joint_name     min  max  rest    → /joint_command
        ("thumb_R",       0, 180,   0),
        ("index_R",       0, 180,   0),
        ("middle_R",      0, 180,   0),
        ("ring_R",        0, 180,   0),
        ("pinky_R",       0, 180,   0),
        ("wrist_R",       0, 180,   0),
    ],
    "Right Arm": [
        ("bicep_R",       0,  90,   0),
        ("rotate_R",     40, 180,  90),
        ("shoulder_R",    0, 180,  30),
        ("omoplate_R",   10,  80,  10),
    ],
    "Left Hand": [
        ("thumb_L",       0, 180,   0),
        ("index_L",       0, 180,   0),
        ("majeure_L",     0, 180,   0),
        ("ring_L",        0, 180,   0),
        ("pinky_L",       0, 180,   0),
        ("wrist_L",       0, 180,   0),
    ],
    "Left Arm": [
        ("bicep_L",       0,  90,   0),
        ("rotate_L",     40, 180,  90),
        ("shoulder_L",    0, 180,  30),
        ("omoplate_L",   25,  90,  25),
    ],
    "Head Movement": [
        ("rollneck",     50, 115,  80),   # Right Arduino → /joint_command
        ("neck",          0, 100,  40),   # Left Arduino  → /joint_command
        ("rothead",      30, 140,  90),
    ],
    "Stomach": [
        ("topstom",      60, 110,  83),
        ("midstom",      60, 120,  90),
        ("lowstom",       0, 180,  90),
    ],
    "Eyes & Jaw": [
        # Синхронные — слайдер двигает оба глаза одновременно
        ("eye_lr_L",     80, 100,  90),   # двигает eye_lr_L + eye_lr_R
        ("eye_ud_L",     80, 110, 100),   # двигает eye_ud_L + eye_ud_R
        ("jaw",          10,  90,  10),
        ("upperLip",     90, 105,  90),
    ],
    "Face (PCA9685)": [
        # Веки: один слайдер на глаз — Upper тянет Lower (инверсия в firmware)
        # Больший угол = закрытие
        ("eyelid_L_Upper", 75,  95,  85),   # синхронизует eyelid_L_Lower
        ("eyelid_R_Upper", 70,  95,  85),   # синхронизует eyelid_R_Lower
        ("eyebrow_L",      60, 110,  90),
        ("eyebrow_R",      70, 105,  80),
        ("cheek_L",        75, 115, 100),
        ("cheek_R",        68, 105,  87),
        ("forhead_L",      90, 110,  90),
        ("forhead_R",      85, 105,  85),
    ],
}

# Какие joint_name идут на /face_command (остальные → /joint_command)
FACE_JOINTS = {
    # Right Arduino FACE_JOINTS
    "eye_lr_R", "eye_ud_R", "upperLip",
    # Left Arduino GPIO face (eye_lr_L, eye_ud_L, jaw теперь тоже face_command)
    "eye_lr_L", "eye_ud_L", "jaw",
    # Left Arduino PCA9685
    "eyelid_L_Upper", "eyelid_L_Lower",
    "eyelid_R_Upper", "eyelid_R_Lower",
    "eyebrow_L", "eyebrow_R",
    "cheek_L",   "cheek_R",
    "forhead_L", "forhead_R",
}

# Пары для синхронизации: при движении одного — второй двигается вместе
# Формат: joint_name → список joint_name которые дублируют это движение
EYE_SYNC = {
    # Горизонталь — оба глаза синхронно
    "eye_lr_L": ["eye_lr_R"],
    "eye_lr_R": ["eye_lr_L"],
    # Вертикаль — оба глаза синхронно
    "eye_ud_L": ["eye_ud_R"],
    "eye_ud_R": ["eye_ud_L"],
    # Веки — Upper двигает Lower того же глаза (инверсия в firmware)
    "eyelid_L_Upper": ["eyelid_L_Lower"],
    "eyelid_R_Upper": ["eyelid_R_Lower"],
}

CENTER_DEG = 90.0  # 0 rad = 90°


def deg_to_rad(deg: float) -> float:
    return (deg - CENTER_DEG) * math.pi / 180.0


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────────────────────────────────────────

class ServoCalibNode(Node):
    def __init__(self):
        super().__init__("servo_calib_gui_node")
        self._joint_pub = self.create_publisher(JointState, "/joint_command", 10)
        self._face_pub  = self.create_publisher(JointState, "/face_command",  10)

    def send(self, joint_name: str, deg: int):
        """Send a single joint command, syncing paired joints if needed."""
        names = [joint_name]
        degs  = [deg]

        # Add sync partners (e.g. both eyes move together)
        for partner in EYE_SYNC.get(joint_name, []):
            names.append(partner)
            degs.append(deg)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = names
        msg.position = [deg_to_rad(float(d)) for d in degs]

        if joint_name in FACE_JOINTS:
            self._face_pub.publish(msg)
        else:
            self._joint_pub.publish(msg)

    def send_all(self, values: dict[str, int]):
        """Send all joints in two batched messages, expanding sync partners."""
        now = self.get_clock().now().to_msg()

        expanded = dict(values)
        for name, deg in values.items():
            for partner in EYE_SYNC.get(name, []):
                if partner not in expanded:
                    expanded[partner] = deg

        body_names, body_pos = [], []
        face_names, face_pos = [], []

        for name, deg in expanded.items():
            rad = deg_to_rad(float(deg))
            if name in FACE_JOINTS:
                face_names.append(name)
                face_pos.append(rad)
            else:
                body_names.append(name)
                body_pos.append(rad)

        if body_names:
            msg = JointState()
            msg.header.stamp = now
            msg.name     = body_names
            msg.position = body_pos
            self._joint_pub.publish(msg)

        if face_names:
            msg = JointState()
            msg.header.stamp = now
            msg.name     = face_names
            msg.position = face_pos
            self._face_pub.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class CalibGUI:
    def __init__(self, root: tk.Tk, node: ServoCalibNode):
        self.root = root
        self.node = node
        root.title("InMoov Servo Calibration  (/joint_command · /face_command)")
        root.resizable(True, True)

        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", padx=6, pady=(6, 0))

        ttk.Button(toolbar, text="All → Rest",
                   command=self._all_rest).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Publish All (batch)",
                   command=self._publish_all).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Close",
                   command=root.destroy).pack(side="right", padx=2)

        # Topic indicators
        ttk.Label(toolbar, text="▶ /joint_command  |  ▶ /face_command",
                  foreground="#555").pack(side="right", padx=10)

        # ── Notebook ─────────────────────────────────────────────────────────
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        self._vars:  dict[str, tk.IntVar] = {}
        self._rests: dict[str, int]       = {}

        for group_name, servos in SERVOS.items():
            frame = ttk.Frame(nb)
            nb.add(frame, text=group_name)
            self._build_group(frame, servos)

    # ── Group builder ────────────────────────────────────────────────────────

    def _build_group(self, parent: ttk.Frame, servos: list):
        canvas = tk.Canvas(parent, borderwidth=0)
        vsb    = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Header
        headers = ["Servo", "Topic", "Min", "Angle", "Max", "", "Rest"]
        for col, text in enumerate(headers):
            ttk.Label(inner, text=text, font=("", 9, "bold")).grid(
                row=0, column=col, padx=4, pady=2, sticky="w")

        for row, (name, mn, mx, rest) in enumerate(servos, start=1):
            self._rests[name] = rest
            var = tk.IntVar(value=rest)
            self._vars[name] = var

            topic_lbl = "/face_command" if name in FACE_JOINTS else "/joint_command"
            color     = "#7a3" if name in FACE_JOINTS else "#38a"

            ttk.Label(inner, text=name, width=18, anchor="w").grid(
                row=row, column=0, padx=4, pady=3, sticky="w")

            tk.Label(inner, text=topic_lbl, fg=color, font=("", 8)).grid(
                row=row, column=1, padx=4)

            ttk.Label(inner, text=str(mn), width=4, anchor="e").grid(
                row=row, column=2, padx=(4, 0))

            slider = ttk.Scale(inner, from_=mn, to=mx, orient="horizontal",
                               variable=var, length=280)
            slider.grid(row=row, column=3, padx=4, sticky="ew")

            ttk.Label(inner, text=str(mx), width=4, anchor="w").grid(
                row=row, column=4, padx=(0, 4))

            angle_lbl = ttk.Label(inner, text=f"{rest:3d}°", width=5)
            angle_lbl.grid(row=row, column=5, padx=4)

            ttk.Button(inner, text=f"↩ {rest}",
                       command=lambda n=name, r=rest: self._set_rest(n, r),
                       width=7).grid(row=row, column=6, padx=4)

            def _on_change(vname, idx, mode,
                           _var=var, _lbl=angle_lbl, _name=name):
                v = int(_var.get())
                _lbl.config(text=f"{v:3d}°")
                self.node.send(_name, v)

            var.trace_add("write", _on_change)

        inner.columnconfigure(3, weight=1)

    # ── Actions ──────────────────────────────────────────────────────────────

    def _set_rest(self, name: str, rest: int):
        self._vars[name].set(rest)           # triggers _on_change → publishes

    def _all_rest(self):
        for name, rest in self._rests.items():
            self._vars[name].set(rest)

    def _publish_all(self):
        """Send all current slider values in two batched JointState messages."""
        values = {name: var.get() for name, var in self._vars.items()}
        self.node.send_all(values)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ServoCalibNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    root = tk.Tk()
    CalibGUI(root, node)
    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
