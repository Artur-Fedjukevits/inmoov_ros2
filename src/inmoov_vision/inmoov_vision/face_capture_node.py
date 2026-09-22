#!/usr/bin/env python3
"""
face_capture_node.py
====================
The only node that opens the USB cameras in the eyes.
Publishes compressed JPEG frames — no raw Image topics.

Topics:
  /camera/eye_left/compressed   (sensor_msgs/CompressedImage)
  /camera/eye_right/compressed  (sensor_msgs/CompressedImage)

Parameters:
  cam_left, cam_right  — paths to the V4L2 devices (by-path)
  fps, width, height   — capture parameters
  jpeg_quality         — JPEG quality 1-100 (default 85)
  flip_h_left/right    — horizontal flip
  reopen_after_fails   — frame failures before a reopen attempt (default 10)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import time

import cv2

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import CompressedImage

_LEFT_PATH  = '/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0'
_RIGHT_PATH = '/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.2:1.0-video-index0'

_REOPEN_AFTER = 10   # consecutive failures before reopen attempt
_REOPEN_WAIT  = 2.0  # seconds between reopen attempts


class FaceCaptureNode(LifecycleNode):
    def __init__(self):
        super().__init__('face_capture_node')
        self._pubs        = {}
        self._caps        = {}
        self._fails       = {'left': 0, 'right': 0}
        self._last_reopen = {'left': 0.0, 'right': 0.0}
        self._last_frame  = {'left': None, 'right': None}
        self._timer       = None

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

        self._pubs = {
            'left':  self.create_lifecycle_publisher(
                CompressedImage, 'camera/eye_left/compressed',  5),
            'right': self.create_lifecycle_publisher(
                CompressedImage, 'camera/eye_right/compressed', 5),
        }
        return TransitionCallbackReturn.SUCCESS

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

        self._timer = self.create_timer(1.0 / self._fps, self._capture)
        self.get_logger().info(
            f'FaceCapture active: left={self._devs["left"]}, '
            f'right={self._devs["right"]}, '
            f'{self._width}x{self._height}@{self._fps}fps')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
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
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS,          self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            self.get_logger().error(f'Failed to open camera {side} ({dev})')
        else:
            self.get_logger().info(f'Camera {side} opened: {dev}')
        return cap

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

    def _capture(self):
        now = self.get_clock().now().to_msg()
        self._grab('left',  now)
        self._grab('right', now)

    def _grab(self, side: str, stamp):
        cap = self._caps[side]

        if not cap.isOpened():
            self._fails[side] += 1
            if self._fails[side] >= self._reopen_after:
                self._reopen(side)
            self._publish_fallback(side, stamp)
            return

        ret, frame = cap.read()

        if not ret:
            self._fails[side] += 1
            if self._fails[side] >= self._reopen_after:
                self.get_logger().warn(
                    f'Camera {side}: {self._fails[side]} failures → reopening',
                    throttle_duration_sec=5.0)
                self._reopen(side)
            self._publish_fallback(side, stamp)
            return

        # Successful frame
        self._fails[side] = 0
        if self._flips[side]:
            frame = cv2.flip(frame, 1)
        self._last_frame[side] = frame
        self._publish_frame(frame, self._pubs[side], stamp)

    def _publish_fallback(self, side: str, stamp):
        """If our own camera is unavailable — publish a mirror of the other one."""
        other = 'right' if side == 'left' else 'left'
        frame = self._last_frame.get(other)
        if frame is None:
            return
        self.get_logger().debug(
            f'Camera {side}: falling back to {other}', throttle_duration_sec=10.0)
        self._publish_frame(frame, self._pubs[side], stamp)

    def _publish_frame(self, frame, pub, stamp):
        ok, buf = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.format       = 'jpeg'
        msg.data         = buf.tobytes()
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
