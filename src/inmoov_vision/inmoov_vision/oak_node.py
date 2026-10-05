#!/usr/bin/env python3
"""
oak_node.py — OAK-D Lite: YOLO detection + stereo depth (depthai v3)

Pipeline follows the official example:
  https://docs.luxonis.com/software-v3/depthai/examples/spatial_detection_network/spatial_detection/

Publishes:
  /objects/detections  (String JSON) — all objects with XYZ coordinates in mm,
      ROS optical convention (x right, y DOWN, z forward) = oak_rgb_camera_optical_frame;
      depthai's own spatial Y points up, so it is negated here
  /objects/nearest     (String JSON) — the nearest object
  /oak/rgb/image_raw/compressed (sensor_msgs/CompressedImage, format 'jpeg')
      — hardware MJPEG from the OAK VideoEncoder (no host-side compression),
        frame_id = oak_rgb_camera_optical_frame (the URDF link), as in depthai-ros

Parameters:
  model_name      — model name in Luxonis HubAI (default: yolov6-nano)
  conf_threshold  — confidence threshold (default 0.5)
  fps             — FPS of all cameras (default 15)
  stream_enabled  — publish the MJPEG stream (default true)
  stream_width / stream_height — MJPEG frame size (default 640x480, full 4:3 FOV)
  stream_fps      — MJPEG frame rate, <= fps (default 15)
  stream_quality  — JPEG quality 1-100 (default 80)
  rgb_frame_id    — frame_id of the stream (default oak_rgb_camera_optical_frame)

Robustness: the OAK thread is a supervisor — if pipeline creation fails, the
pipeline stops or the device disconnects, it rebuilds the pipeline with a
growing back-off (_RESTART_MIN_SEC → _RESTART_MAX_SEC) while the node is active.
/diagnostics ('oak pipeline'): ERROR when no detection packet arrived for 3 s —
distinguishes an empty scene (packets keep coming) from a dead pipeline.

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_updater import Updater
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

_depthai_available = False
try:
    import depthai as dai
    _depthai_available = True
except ImportError:
    pass


COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


_RESTART_MIN_SEC = 5.0    # first pipeline rebuild after an OAK failure
_RESTART_MAX_SEC = 60.0   # back-off cap
_STABLE_RUN_SEC  = 60.0   # a run this long resets the back-off


class OakNode(LifecycleNode):
    def __init__(self):
        super().__init__('oak_node')
        self._running    = False
        self._oak_thread = None
        self._det_pub    = None
        self._near_pub   = None
        self._img_pub    = None
        self._n_frames   = 0
        # Diagnostics: the pipeline delivers a detection packet every frame,
        # even with no objects — so packet age tells "empty scene" from "dead OAK"
        self._last_packet_t = 0.0
        self._n_packets     = 0
        self._restarts      = 0
        self._diag          = None

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('model_name',     'yolov6-nano')
        self._dp('conf_threshold', 0.5)
        self._dp('fps',            15)
        self._dp('stream_enabled', True)
        self._dp('stream_width',   640)
        self._dp('stream_height',  480)
        self._dp('stream_fps',     15)
        self._dp('stream_quality', 80)
        self._dp('rgb_frame_id',   'oak_rgb_camera_optical_frame')

        self._model_name = self.get_parameter('model_name').value
        self._conf       = self.get_parameter('conf_threshold').value
        self._fps        = self.get_parameter('fps').value
        self._stream     = bool(self.get_parameter('stream_enabled').value)
        self._stream_wh  = (int(self.get_parameter('stream_width').value),
                            int(self.get_parameter('stream_height').value))
        self._stream_fps = float(self.get_parameter('stream_fps').value)
        self._stream_q   = int(self.get_parameter('stream_quality').value)
        self._frame_id   = self.get_parameter('rgb_frame_id').value

        self._det_pub  = self.create_lifecycle_publisher(String, 'objects/detections', 10)
        self._near_pub = self.create_lifecycle_publisher(String, 'objects/nearest', 10)
        # Sensor QoS (best effort, shallow): a late JPEG is worthless
        self._img_pub  = self.create_lifecycle_publisher(
            CompressedImage, 'oak/rgb/image_raw/compressed', qos_profile_sensor_data)
        if self._diag is None:
            self._diag = Updater(self, period=1.0)
            self._diag.setHardwareID('OAK-D Lite')
            self._diag.add('oak pipeline', self._diagnose)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._det_pub.on_activate(state)
        self._near_pub.on_activate(state)
        self._img_pub.on_activate(state)

        if not _depthai_available:
            self.get_logger().error('depthai is not installed — degraded (oak_node)')
            self._det_pub.on_deactivate(state)
            self._near_pub.on_deactivate(state)
            self._img_pub.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._running    = True
        self._oak_thread = threading.Thread(target=self._run_oak, daemon=True)
        self._oak_thread.start()
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._running = False
        if self._oak_thread and self._oak_thread.is_alive():
            self._oak_thread.join(timeout=5.0)
        self._oak_thread = None
        self._det_pub.on_deactivate(state)
        self._near_pub.on_deactivate(state)
        self._img_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._running = False
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._running = False
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._running = False
        return TransitionCallbackReturn.SUCCESS

    def _diagnose(self, stat):
        age = time.monotonic() - self._last_packet_t if self._last_packet_t else float('inf')
        rate = self._n_packets / 1.0   # Updater period = 1 s
        self._n_packets = 0
        stat.add('model', self._model_name)
        stat.add('packets_per_sec', f'{rate:.1f}')
        stat.add('jpeg_frames_total', str(self._n_frames))
        stat.add('last_packet_age_sec', f'{age:.1f}')
        stat.add('pipeline_restarts', str(self._restarts))
        if not self._running:
            stat.summary(DiagnosticStatus.OK, 'inactive')
        elif age > 3.0:
            stat.summary(DiagnosticStatus.ERROR, f'no detection packets for {age:.0f} s')
        else:
            stat.summary(DiagnosticStatus.OK, f'{rate:.0f} packets/s')
        return stat

    def _run_oak(self):
        """Supervisor: (re)builds the pipeline until the node is deactivated."""
        attempt = 0
        while self._running:
            t_start = time.monotonic()
            self._run_pipeline_once()
            if not self._running:
                break
            if time.monotonic() - t_start >= _STABLE_RUN_SEC:
                attempt = 0
            delay = min(_RESTART_MIN_SEC * (2 ** attempt), _RESTART_MAX_SEC)
            attempt += 1
            self._restarts += 1
            self.get_logger().warn(
                f'OAK pipeline stopped — rebuilding in {delay:.0f}s (attempt {attempt})')
            deadline = time.monotonic() + delay
            while self._running and time.monotonic() < deadline:
                time.sleep(0.1)

    def _run_pipeline_once(self):
        """Builds and runs the pipeline; returns when it stops or fails."""
        # Official pattern: https://docs.luxonis.com/software-v3/depthai/examples/
        # spatial_detection_network/spatial_detection/
        size = (640, 400)

        try:
            pipeline = dai.Pipeline()

            # ── Cameras ───────────────────────────────────────────────────────
            camRgb    = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A, sensorFps=self._fps)
            monoLeft  = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B, sensorFps=self._fps)
            monoRight = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C, sensorFps=self._fps)

            # ── StereoDepth — only what the official example uses ─────────────
            depthSource = pipeline.create(dai.node.StereoDepth)
            depthSource.setExtendedDisparity(True)
            monoLeft.requestOutput(size).link(depthSource.left)
            monoRight.requestOutput(size).link(depthSource.right)

            # ── SpatialDetectionNetwork.build() — official pattern ────────────
            # build() itself requests the needed output from the camera and links depth.
            modelDescription = dai.NNModelDescription(self._model_name)
            spatialNet = pipeline.create(dai.node.SpatialDetectionNetwork).build(
                camRgb, depthSource, modelDescription)

            spatialNet.setConfidenceThreshold(self._conf)
            spatialNet.setDepthLowerThreshold(100)
            spatialNet.setDepthUpperThreshold(8000)
            spatialNet.input.setBlocking(False)

            # ── Output queue ──────────────────────────────────────────────────
            det_queue = spatialNet.out.createOutputQueue(maxSize=4, blocking=False)

            # ── Hardware MJPEG: the encoder on the OAK, the host only forwards bytes
            jpg_queue = None
            if self._stream:
                enc = pipeline.create(dai.node.VideoEncoder).build(
                    camRgb.requestOutput(self._stream_wh, dai.ImgFrame.Type.NV12,
                                         fps=self._stream_fps),
                    frameRate=self._stream_fps,
                    profile=dai.VideoEncoderProperties.Profile.MJPEG,
                    quality=self._stream_q)
                jpg_queue = enc.out.createOutputQueue(maxSize=2, blocking=False)

        except Exception as e:
            self.get_logger().error(f'Failed to create pipeline: {e}')
            return

        try:
            with pipeline:
                pipeline.start()
                self.get_logger().info('OAK-D Lite connected (depthai v3)')

                while self._running and pipeline.isRunning():
                    if jpg_queue is not None:
                        self._publish_jpeg(jpg_queue.tryGet())
                    packet = det_queue.tryGet()
                    if packet is None:
                        time.sleep(0.005)
                        continue
                    self._last_packet_t = time.monotonic()
                    self._n_packets += 1
                    self._process(packet.detections)
                if self._running:
                    self.get_logger().error('OAK pipeline is no longer running')

        except Exception as e:
            self.get_logger().error(f'OAK error: {e}')

    def _publish_jpeg(self, frame):
        if frame is None:
            return
        self._n_frames += 1
        if self._img_pub.get_subscription_count() == 0:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.format = 'jpeg'
        msg.data = frame.getData().tobytes()
        self._img_pub.publish(msg)

    def _process(self, detections):
        objects = []
        for det in detections:
            # v3: labelName is built into the model; fall back to COCO_LABELS by index
            label = getattr(det, 'labelName', None)
            if not label:
                label = (COCO_LABELS[det.label]
                         if det.label < len(COCO_LABELS) else str(det.label))
            obj = {
                'label':      label,
                'confidence': round(det.confidence, 3),
                'x_mm':       round(det.spatialCoordinates.x, 1),
                # depthai spatial Y points up; ROS optical frame Y points down
                'y_mm':       round(-det.spatialCoordinates.y, 1),
                'z_mm':       round(det.spatialCoordinates.z, 1),
                'bbox':       [round(det.xmin, 3), round(det.ymin, 3),
                               round(det.xmax, 3), round(det.ymax, 3)],
            }
            objects.append(obj)

        if not objects:
            return

        msg = String()
        msg.data = json.dumps({'objects': objects}, ensure_ascii=False)
        self._det_pub.publish(msg)

        nearest = min(objects, key=lambda o: abs(o['z_mm']))
        near_msg = String()
        near_msg.data = json.dumps(nearest, ensure_ascii=False)
        self._near_pub.publish(near_msg)

        self.get_logger().debug(
            f'Objects: {[o["label"] for o in objects]}, '
            f'nearest: {nearest["label"]} @ {nearest["z_mm"]:.0f} mm')

    # destroy_node replaced by on_shutdown / on_deactivate (lifecycle)


def main():
    rclpy.init()
    node = OakNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
