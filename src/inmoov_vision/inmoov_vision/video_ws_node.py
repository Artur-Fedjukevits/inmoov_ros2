#!/usr/bin/env python3
"""
video_ws_node.py — JPEG topic -> WebSocket binary frames (Android / Godot viewer)

Subscribes to a sensor_msgs/CompressedImage topic (default: oak_node's hardware
MJPEG stream) and sends every frame as one binary WebSocket message holding the
raw JPEG bytes — nothing is decoded or re-encoded on the host.

Client side (Godot):
  var ws := WebSocketPeer.new(); ws.connect_to_url("ws://<robot>:9091")
  ... ws.poll(); while ws.get_available_packet_count() > 0:
        var img := Image.new(); img.load_jpg_from_buffer(ws.get_packet())
        texture_rect.texture = ImageTexture.create_from_image(img)

Slow clients never stall the others: each client always gets the *latest* frame,
frames it was too slow for are dropped. No frame is sent while nobody is
connected, and the topic subscription stays (cheap: oak_node skips the publish
only when it has no subscribers at all).

Parameters:
  topic  — CompressedImage topic (default /oak/rgb/image_raw/compressed)
  port   — WebSocket port (default 9091)
  host   — bind address (default '' = all interfaces)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import asyncio
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


class VideoWsNode(Node):
    def __init__(self):
        super().__init__('video_ws_node')
        self.declare_parameter('topic', '/oak/rgb/image_raw/compressed')
        self.declare_parameter('port', 9091)
        self.declare_parameter('host', '')
        self._port = int(self.get_parameter('port').value)
        self._host = self.get_parameter('host').value or None

        self._loop = asyncio.new_event_loop()
        self._frame = None            # latest JPEG (bytes)
        self._seq = 0                 # frame counter — tells a client "something new"
        self._waiters = set()         # per-client asyncio.Event, set on a new frame
        self._stop = None             # asyncio.Event, created inside the loop
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self.create_subscription(
            CompressedImage, self.get_parameter('topic').value,
            self._on_image, qos_profile_sensor_data)

    # ── ROS thread ────────────────────────────────────────────────────────────
    def _on_image(self, msg):
        if not self._waiters:        # nobody connected: skip the copy into the loop
            return
        self._loop.call_soon_threadsafe(self._set_frame, bytes(msg.data))

    # ── asyncio thread ────────────────────────────────────────────────────────
    def _set_frame(self, data):
        self._frame = data
        self._seq += 1
        for ev in self._waiters:
            ev.set()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            self.get_logger().error(f'WebSocket server failed: {e}')

    async def _serve(self):
        self._stop = asyncio.Event()
        async with serve(self._client, self._host, self._port,
                         max_size=None, compression=None):
            self.get_logger().info(f'Video WebSocket on ws://*:{self._port}')
            await self._stop.wait()

    async def _client(self, ws):
        peer = ws.remote_address
        ev = asyncio.Event()
        self._waiters.add(ev)
        self.get_logger().info(f'Video client connected: {peer}')
        sent = self._seq
        try:
            while True:
                await ev.wait()
                ev.clear()
                if self._seq == sent or self._frame is None:
                    continue
                sent = self._seq
                await ws.send(self._frame)   # a slow client blocks only itself
        except ConnectionClosed:
            pass
        finally:
            self._waiters.discard(ev)
            self.get_logger().info(f'Video client disconnected: {peer}')

    def shutdown(self):
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=2.0)


def main():
    rclpy.init()
    node = VideoWsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
