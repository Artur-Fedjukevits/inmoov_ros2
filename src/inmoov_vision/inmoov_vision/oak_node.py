#!/usr/bin/env python3
"""
oak_node.py — OAK-D Lite: YOLO detection + stereo depth (depthai v3)

Pipeline follows the official example:
  https://docs.luxonis.com/software-v3/depthai/examples/spatial_detection_network/spatial_detection/

Publishes:
  /objects/detections  (String JSON) — all objects with XYZ coordinates
  /objects/nearest     (String JSON) — the nearest object

Parameters:
  model_name      — model name in Luxonis HubAI (default: yolov6-nano)
  conf_threshold  — confidence threshold (default 0.5)
  fps             — FPS of all cameras (default 15)

Robustness: the OAK thread is a supervisor — if pipeline creation fails, the
pipeline stops or the device disconnects, it rebuilds the pipeline with a
growing back-off (_RESTART_MIN_SEC → _RESTART_MAX_SEC) while the node is active.

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import threading
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
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

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('model_name',     'yolov6-nano')
        self._dp('conf_threshold', 0.5)
        self._dp('fps',            15)

        self._model_name = self.get_parameter('model_name').value
        self._conf       = self.get_parameter('conf_threshold').value
        self._fps        = self.get_parameter('fps').value

        self._det_pub  = self.create_lifecycle_publisher(String, 'objects/detections', 10)
        self._near_pub = self.create_lifecycle_publisher(String, 'objects/nearest', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._det_pub.on_activate(state)
        self._near_pub.on_activate(state)

        if not _depthai_available:
            self.get_logger().error('depthai is not installed — degraded (oak_node)')
            self._det_pub.on_deactivate(state)
            self._near_pub.on_deactivate(state)
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

        except Exception as e:
            self.get_logger().error(f'Failed to create pipeline: {e}')
            return

        try:
            with pipeline:
                pipeline.start()
                self.get_logger().info('OAK-D Lite connected (depthai v3)')

                while self._running and pipeline.isRunning():
                    packet = det_queue.tryGet()
                    if packet is None:
                        time.sleep(0.01)
                        continue
                    self._process(packet.detections)
                if self._running:
                    self.get_logger().error('OAK pipeline is no longer running')

        except Exception as e:
            self.get_logger().error(f'OAK error: {e}')

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
                'y_mm':       round(det.spatialCoordinates.y, 1),
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
