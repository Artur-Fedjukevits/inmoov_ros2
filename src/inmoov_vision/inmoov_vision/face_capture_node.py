#!/usr/bin/env python3
"""
face_capture_node.py
====================
The only node that opens the USB cameras in the eyes.
Publishes compressed JPEG frames — no raw Image topics.

Topics:
  /camera/eye_left/compressed   (sensor_msgs/CompressedImage)
  /camera/eye_right/compressed  (sensor_msgs/CompressedImage)
  header.frame_id = the eye the picture really comes from ('eye_left'/'eye_right');
  on a camera failure the other eye's last frame is republished with ITS OWN
  frame_id and stamp, so consumers can tell the mirror (and a frozen source) apart.
  /diagnostics — per eye: OK / WARN (mirroring the other eye) / ERROR (no frames),
  with frame age, measured fps and capture mode.

Each eye is read in its own thread: cap.read() blocks until the camera's next
frame, and reading both eyes one after the other in one timer added the two
waits up — in dim light (exposure_dynamic_framerate drops the camera to 15 fps)
that halved the rate to ~7.5 fps per eye (measured 2026-10-02).

Capture path: the cameras are opened in MJPG and their JPEG is published as-is
(CAP_PROP_CONVERT_RGB=0) — no decode/encode on the NUC (~0.2 % CPU for both eyes
vs ~70 % for YUYV → BGR → cv2.imencode). A horizontal flip, or a camera that
refuses MJPG, falls back to decode + re-encode with jpeg_quality.

Parameters:
  cam_left, cam_right  — paths to the V4L2 devices (by-path)
  fps, width, height   — capture parameters
  jpeg_quality         — JPEG quality 1-100 (default 85)
  flip_h_left/right    — horizontal flip
  v4l2_controls        — UVC controls set with v4l2-ctl on every open
                         (default 'power_line_frequency=1' — 50 Hz)
  reopen_after_fails   — frame failures before a reopen attempt (default 10)
  blink_hold_sec       — keep dropping frames this long after the eyelids reopen
                         (/eyes/blink False; default 0.22 s)
  blink_max_sec        — cap on a blink window if the reopen never arrives (0.6 s)

Blinks: the eyelids pass in front of the eye cameras. Frames read between
/eyes/blink True (close) and False (reopen) + blink_hold_sec are not published,
so no consumer (detection, head tracker, lips, gallery, LLM photos) sees a
half-covered frame. Measured 2026-10-02: the eyelid is in the picture from the
close command up to ~0.29 s after it (reopen is sent 0.12 s after the close).

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import subprocess
import threading
import time

import cv2

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_updater import Updater
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

_LEFT_PATH  = '/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0'
_RIGHT_PATH = '/dev/v4l/by-path/pci-0000:c8:00.3-usb-0:1.3:1.0-video-index0'

_REOPEN_AFTER = 10   # consecutive failures before reopen attempt
_REOPEN_WAIT  = 2.0  # seconds between reopen attempts


class FaceCaptureNode(LifecycleNode):
    def __init__(self):
        super().__init__('face_capture_node')
        self._pubs        = {}
        self._caps        = {}
        self._fails       = {'left': 0, 'right': 0}
        self._last_reopen = {'left': 0.0, 'right': 0.0}
        self._last_frame  = {'left': None, 'right': None}   # (jpeg bytes, stamp)
        self._passthrough = {'left': False, 'right': False}  # publishing camera MJPEG as-is
        self._threads     = {}
        self._active      = False
        # Diagnostics: last own frame (monotonic) and frames since the last report
        self._last_ok_t   = {'left': 0.0, 'right': 0.0}
        self._n_frames    = {'left': 0, 'right': 0}
        self._diag        = None
        self._blink_until = 0.0     # monotonic: drop frames read before this
        self._n_blink_drop = 0

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores repeated declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('cam_left',           _LEFT_PATH)
        self._dp('cam_right',          _RIGHT_PATH)
        self._dp('fps',                15)
        self._dp('width',              640)
        self._dp('height',             480)
        self._dp('jpeg_quality',       85)
        self._dp('flip_h_left',        False)
        self._dp('flip_h_right',       False)
        self._dp('reopen_after_fails', _REOPEN_AFTER)
        # UVC controls applied on every open (they reset when the camera re-enumerates),
        # v4l2-ctl syntax: 'power_line_frequency=1,backlight_compensation=1'
        self._dp('v4l2_controls',      'power_line_frequency=1')
        self._dp('blink_hold_sec',     0.22)
        self._dp('blink_max_sec',      0.6)

        self._devs = {
            'left':  self.get_parameter('cam_left').value,
            'right': self.get_parameter('cam_right').value,
        }
        self._fps          = self.get_parameter('fps').value
        self._width        = self.get_parameter('width').value
        self._height       = self.get_parameter('height').value
        self._quality      = self.get_parameter('jpeg_quality').value
        self._flips        = {
            'left':  self.get_parameter('flip_h_left').value,
            'right': self.get_parameter('flip_h_right').value,
        }
        self._reopen_after = self.get_parameter('reopen_after_fails').value
        self._v4l2_ctrls   = self.get_parameter('v4l2_controls').value.strip()
        self._blink_hold   = float(self.get_parameter('blink_hold_sec').value)
        self._blink_max    = float(self.get_parameter('blink_max_sec').value)
        self.create_subscription(Bool, '/eyes/blink', self._blink_cb, 10)

        self._pubs = {
            'left':  self.create_lifecycle_publisher(
                CompressedImage, 'camera/eye_left/compressed',  5),
            'right': self.create_lifecycle_publisher(
                CompressedImage, 'camera/eye_right/compressed', 5),
        }
        if self._diag is None:
            self._diag = Updater(self, period=1.0)
            self._diag.setHardwareID('eye cameras (UVC)')
            for side in ('left', 'right'):
                self._diag.add(f'eye camera {side}',
                               lambda stat, side=side: self._diagnose(stat, side))
        return TransitionCallbackReturn.SUCCESS

    def _blink_cb(self, msg: Bool):
        now = time.monotonic()
        self._blink_until = now + (self._blink_max if msg.data else self._blink_hold)

    def _diagnose(self, stat, side: str):
        now = time.monotonic()
        fps = self._n_frames[side] / 1.0   # Updater period = 1 s
        self._n_frames[side] = 0
        age = now - self._last_ok_t[side] if self._last_ok_t[side] else float('inf')
        stat.add('device', self._devs[side])
        stat.add('mode', 'MJPEG passthrough' if self._passthrough[side] else 'decode+re-encode')
        stat.add('fps', f'{fps:.1f}')
        stat.add('last_frame_age_sec', f'{age:.1f}')
        stat.add('consecutive_failures', str(self._fails[side]))
        stat.add('frames_dropped_on_blinks', str(self._n_blink_drop))
        if not self._active:
            stat.summary(DiagnosticStatus.OK, 'inactive')
        elif age > 3.0:
            other = 'right' if side == 'left' else 'left'
            level = (DiagnosticStatus.WARN if now - self._last_ok_t[other] < 3.0
                     else DiagnosticStatus.ERROR)
            stat.summary(level, f'no frames for {age:.0f} s'
                         + (f' — mirroring the {other} eye' if level == DiagnosticStatus.WARN else ''))
        else:
            stat.summary(DiagnosticStatus.OK, f'{fps:.0f} fps')
        return stat

    def on_activate(self, state):
        for pub in self._pubs.values():
            pub.on_activate(state)

        opened_any = False
        for side in ('left', 'right'):
            cap = self._open(side)
            self._caps[side] = cap
            if cap.isOpened():
                opened_any = True

        if not opened_any:
            for pub in self._pubs.values():
                pub.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._active = True
        for side in ('left', 'right'):
            th = threading.Thread(target=self._reader, args=(side,), daemon=True,
                                  name=f'eye_{side}_reader')
            self._threads[side] = th
            th.start()
        self.get_logger().info(
            f'FaceCapture active: left={self._devs["left"]}, '
            f'right={self._devs["right"]}, '
            f'{self._width}x{self._height}@{self._fps}fps')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._stop_readers()
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        for pub in self._pubs.values():
            pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._stop_readers()
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._stop_readers()
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        return TransitionCallbackReturn.SUCCESS

    # ── Camera open ──────────────────────────────────────────────────────────

    def _open(self, side: str):
        dev = self._devs[side]
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS,          self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            self.get_logger().error(f'Failed to open camera {side} ({dev})')
            return cap
        self._apply_controls(side, dev)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')
        passthrough = fourcc == 'MJPG' and not self._flips[side]
        if passthrough:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)   # read() returns the camera's JPEG bytes
        self._passthrough[side] = passthrough
        mode = 'MJPEG passthrough' if passthrough else f'{fourcc} decode+re-encode'
        self.get_logger().info(f'Camera {side} opened: {dev} ({mode})')
        return cap

    def _apply_controls(self, side: str, dev: str) -> None:
        """power_line_frequency etc. — OpenCV has no properties for these.
        Default 1 = 50 Hz (Norway): the camera shipped with 60 Hz, which makes
        lamps flicker as horizontal bands."""
        if not self._v4l2_ctrls:
            return
        try:
            res = subprocess.run(['v4l2-ctl', '-d', dev, f'--set-ctrl={self._v4l2_ctrls}'],
                                 capture_output=True, text=True, timeout=3.0)
            if res.returncode != 0:
                self.get_logger().warn(
                    f'Camera {side}: v4l2-ctl {self._v4l2_ctrls} failed: {res.stderr.strip()}')
        except (OSError, subprocess.TimeoutExpired) as e:
            self.get_logger().warn(f'Camera {side}: v4l2-ctl unavailable ({e})')

    def _reopen(self, side: str):
        now = time.time()
        if now - self._last_reopen[side] < _REOPEN_WAIT:
            return
        self._last_reopen[side] = now

        self.get_logger().warn(f'Reopening camera {side}...')
        old = self._caps.get(side)
        if old is not None:
            old.release()
        self._caps[side] = self._open(side)
        self._fails[side] = 0

    # ── Frame capture ────────────────────────────────────────────────────────

    def _reader(self, side: str):
        """One eye's capture loop, paced to fps (the read itself blocks until
        the camera's next frame)."""
        period = 1.0 / self._fps
        while self._active:
            t0 = time.monotonic()
            try:
                self._grab(side)
            except Exception as e:   # never let one bad frame kill the eye's thread
                self.get_logger().error(f'Camera {side}: capture error: {e}',
                                        throttle_duration_sec=5.0)
            rest = period - (time.monotonic() - t0)
            if rest > 0:
                time.sleep(rest)

    def _stop_readers(self):
        self._active = False
        for th in self._threads.values():
            th.join(timeout=6.0)   # a read on a dead camera returns after the uvc timeout (5 s)
        self._threads.clear()

    def _grab(self, side: str):
        cap = self._caps[side]

        if not cap.isOpened():
            self._fails[side] += 1
            if self._fails[side] >= self._reopen_after:
                self._reopen(side)
            self._publish_fallback(side)
            return

        ret, frame = cap.read()
        # Stamp right after the frame arrives — closest to when it was taken
        # (lip-audio sync in face_tracker relies on it)
        stamp = self.get_clock().now().to_msg()
        if ret and self._passthrough[side]:
            jpeg = frame.tobytes() if frame is not None else b''
            ret = jpeg[:2] == b'\xff\xd8'   # truncated/corrupt MJPEG frame → treat as a miss

        if not ret:
            self._fails[side] += 1
            if self._fails[side] >= self._reopen_after:
                self.get_logger().warn(
                    f'Camera {side}: {self._fails[side]} failures → reopening',
                    throttle_duration_sec=5.0)
                self._reopen(side)
            self._publish_fallback(side)
            return

        # Successful frame
        self._fails[side] = 0
        if time.monotonic() < self._blink_until:
            self._n_blink_drop += 1   # eyelid in front of the lens — don't publish
            return
        if not self._passthrough[side]:
            if self._flips[side]:
                frame = cv2.flip(frame, 1)
            ok, buf = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            if not ok:
                return
            jpeg = buf.tobytes()
        self._last_frame[side] = (jpeg, stamp)
        self._last_ok_t[side] = time.monotonic()
        self._n_frames[side] += 1
        self._publish_jpeg(jpeg, self._pubs[side], stamp, f'eye_{side}')

    def _publish_fallback(self, side: str):
        """Own camera unavailable — republish the other eye's last frame.

        Keeps that frame's own stamp and frame_id: a re-stamped copy would look
        "fresh" to face_detection's frozen-camera check even when the other eye
        itself stopped updating.
        """
        other = 'right' if side == 'left' else 'left'
        last = self._last_frame.get(other)
        if last is None:
            return
        jpeg, other_stamp = last
        self.get_logger().debug(
            f'Camera {side}: falling back to {other}', throttle_duration_sec=10.0)
        self._publish_jpeg(jpeg, self._pubs[side], other_stamp, f'eye_{other}')

    @staticmethod
    def _publish_jpeg(jpeg: bytes, pub, stamp, frame_id: str):
        msg = CompressedImage()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.format          = 'jpeg'
        msg.data            = jpeg
        pub.publish(msg)

    # destroy_node replaced by on_shutdown / on_deactivate (lifecycle)


def main():
    rclpy.init()
    node = FaceCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
