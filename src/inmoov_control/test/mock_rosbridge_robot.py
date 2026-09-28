#!/usr/bin/env python3
"""
mock_rosbridge_robot.py — a fake InMoov behind a fake rosbridge, no ROS needed.

Speaks the subset of the rosbridge v2 protocol the Android app uses
(subscribe / unsubscribe / advertise / publish / call_service) and emulates
urdf_bridge + arduino nodes + joint_state_publisher:

  /urdf_joint_cmd  (inmoov_msgs/msg/JointCommand, URDF)  -> arbitration -> servo targets
  /joint_cmd       (inmoov_msgs/msg/JointCommand, servo) -> arbitration -> servo targets
  servos move toward the targets at the firmware default speed (~33 deg/s)
  /urdf_joint_states (sensor_msgs/msg/JointState) published at 25 Hz
  /urdf_bridge/get_map (std_srvs/srv/Trigger), /urdf_bridge/set_calibration,
  /urdf_bridge/status

  pip install websockets pyyaml
  python3 mock_rosbridge_robot.py [--port 9090] [--demo] [--map ../config/servo_urdf_map.yaml]

--demo: a fake "head_tracker" (priority 40) slowly turns the head, so the
mirror mode of the app has something to show.

Author: Artur Fedjukevits
Assisted by: Claude (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from inmoov_control.servo_urdf_map import (JointMap, ServoUrdfMap,  # noqa: E402
                                           servo_deg_to_rad, servo_rad_to_deg)

SPEED_DEG_S = 2 * 1000 / 60      # firmware step 2 per 60 ms tick
PRIORITY_REMOTE = 80             # urdf_bridge caps /urdf_joint_cmd priority at this
# arduino_comm_node mirrors the eyes; urdf_bridge drives only the left (leading) one
EYE_SYNC = {'eye_lr_L': 'eye_lr_R', 'eye_lr_R': 'eye_lr_L',
            'eye_ud_L': 'eye_ud_R', 'eye_ud_R': 'eye_ud_L'}
EYE_FOLLOWERS = {'eye_lr_R', 'eye_ud_R'}


class MockRobot:

    def __init__(self, m: ServoUrdfMap, user_map: str):
        self.map = m
        self.user_map = user_map
        self.pos = {j.servo: j.servo_rest for j in m.joints}      # servo deg
        self.target = dict(self.pos)
        self.speed = {j.servo: SPEED_DEG_S for j in m.joints}
        self.owner = {}      # servo -> (source, priority, until)
        self.clients = {}    # ws -> {topic: {'throttle': s, 'last': t}}
        self.status = {'map_source': 'mock', 'joints': len(m.joints),
                       'calibrated': 0, 'last_calibration': None}
        self.log = []

    # ------------------------------------------------------------ robot
    def _arbitrate(self, servo, source, prio, lease, now):
        o = self.owner.get(servo)
        if o and o[2] > now and o[0] != source and prio <= o[1]:
            return False
        if lease > 0:
            self.owner[servo] = (source, prio, now + min(lease, 30.0))
        elif o and o[0] == source:
            self.owner.pop(servo, None)
        return True

    def apply_servo_cmd(self, source, prio, lease, release, names, positions, velocities):
        now = time.monotonic()
        rejected = 0
        for i, n in enumerate(names):
            if n not in self.pos:
                continue
            if release:
                if self.owner.get(n, ('',))[0] == source:
                    self.owner.pop(n)
                continue
            if not self._arbitrate(n, source, prio, lease, now):
                rejected += 1
                continue
            v = velocities[i] if i < len(velocities) else 0.0
            for m in (n, EYE_SYNC.get(n)):
                if m not in self.pos:
                    continue
                j = self.map.by_servo[m]
                self.target[m] = j.clamp_servo(round(servo_rad_to_deg(positions[i])))
                self.speed[m] = math.degrees(abs(v)) if v else SPEED_DEG_S
        return rejected

    def on_urdf_cmd(self, msg):
        cmd = msg.get('cmd', {})
        names, pos, vel = [], [], []
        for i, un in enumerate(cmd.get('name', [])):
            if un in self.map.by_urdf and self.map.by_urdf[un].servo in EYE_FOLLOWERS:
                continue
            r = (self.map.urdf_to_servo_rad(un, cmd['position'][i])
                 if not msg.get('release') else (self.map.by_urdf[un].servo, 0.0)
                 if un in self.map.by_urdf else None)
            if r is None:
                continue
            names.append(r[0])
            pos.append(r[1])
            v = cmd.get('velocity') or []
            vel.append(self.map.urdf_velocity_to_servo(un, cmd['position'][i], v[i])
                       if i < len(v) and v[i] and not msg.get('release') else 0.0)
        rej = self.apply_servo_cmd(msg.get('source', 'remote'),
                                   min(msg.get('priority', 0), PRIORITY_REMOTE),
                                   msg.get('lease_sec', 0.0), msg.get('release', False),
                                   names, pos, vel)
        self.log.append(('urdf_cmd', msg.get('source'), len(names), rej))

    def on_servo_cmd(self, msg):
        cmd = msg.get('cmd', {})
        rej = self.apply_servo_cmd(msg.get('source', ''), msg.get('priority', 0),
                                   msg.get('lease_sec', 0.0), msg.get('release', False),
                                   cmd.get('name', []), cmd.get('position', []),
                                   cmd.get('velocity', []))
        self.log.append(('joint_cmd', msg.get('source'), len(cmd.get('name', [])), rej))

    def on_calibration(self, msg):
        try:
            d = json.loads(msg['data'])
            old = self.map.by_servo[d['servo']]
            new = JointMap(old.servo, old.urdf, [old.servo_min, old.servo_max], old.servo_rest,
                           d['points'], old.urdf_limits, d.get('verified', True), old.note,
                           old.board)
            self.map = ServoUrdfMap([new if j.servo == new.servo else j for j in self.map.joints],
                                    self.map.unmapped_servos, 'mock')
            if self.user_map:
                ServoUrdfMap([j for j in self.map.joints if j.verified]).save(self.user_map)
            self.status['last_calibration'] = {'ok': True, 'servo': new.servo,
                                               'points': new.to_dict()['points'],
                                               'persisted': bool(self.user_map) and new.verified}
        except Exception as e:  # noqa: BLE001
            self.status['last_calibration'] = {'ok': False, 'error': str(e)}
        self.status['calibrated'] = sum(j.verified for j in self.map.joints)

    def step(self, dt):
        for n, t in self.target.items():
            p = self.pos[n]
            d = self.speed[n] * dt
            self.pos[n] = t if abs(t - p) <= d else p + math.copysign(d, t - p)

    def joint_states(self):
        now = time.time()
        return {
            'header': {'stamp': {'sec': int(now), 'nanosec': int((now % 1) * 1e9)},
                       'frame_id': ''},
            'name': [j.urdf for j in self.map.joints],
            'position': [j.servo_to_urdf(self.pos[j.servo]) for j in self.map.joints],
            'velocity': [], 'effort': [],
        }

    def get_map(self):
        d = self.map.to_dict()
        d['source'] = 'mock'
        d['servo_deg'] = {k: round(v, 2) for k, v in self.pos.items()}
        return d

    # ------------------------------------------------------- rosbridge
    async def send(self, ws, topic, msg):
        try:
            await ws.send(json.dumps({'op': 'publish', 'topic': topic, 'msg': msg}))
        except websockets.ConnectionClosed:
            pass

    async def handler(self, ws):
        subs = self.clients[ws] = {}
        print(f'client connected: {ws.remote_address}')
        try:
            async for raw in ws:
                m = json.loads(raw)
                op = m.get('op')
                if op == 'subscribe':
                    subs[m['topic']] = {'throttle': m.get('throttle_rate', 0) / 1000.0, 'last': 0}
                    if m['topic'] == '/urdf_bridge/status':
                        await self.send(ws, m['topic'], {'data': json.dumps(self.status)})
                elif op == 'unsubscribe':
                    subs.pop(m['topic'], None)
                elif op == 'publish':
                    t, msg = m['topic'], m['msg']
                    if t == '/urdf_joint_cmd':
                        self.on_urdf_cmd(msg)
                    elif t == '/joint_cmd':
                        self.on_servo_cmd(msg)
                    elif t == '/urdf_bridge/set_calibration':
                        self.on_calibration(msg)
                        await self.broadcast('/urdf_bridge/status',
                                             {'data': json.dumps(self.status)})
                elif op == 'call_service':
                    resp = {'op': 'service_response', 'service': m['service'],
                            'id': m.get('id'), 'result': True}
                    if m['service'] == '/urdf_bridge/get_map':
                        resp['values'] = {'success': True, 'message': json.dumps(self.get_map())}
                    else:
                        resp['result'] = False
                        resp['values'] = f'service {m["service"]} does not exist'
                    await ws.send(json.dumps(resp))
                # advertise / unadvertise: nothing to do
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.pop(ws, None)
            print('client disconnected')

    async def broadcast(self, topic, msg):
        for ws, subs in list(self.clients.items()):
            if topic in subs:
                await self.send(ws, topic, msg)

    async def loop(self, demo):
        t0 = last = time.monotonic()
        while True:
            await asyncio.sleep(0.04)
            now = time.monotonic()
            if demo:
                a = 0.6 * math.sin((now - t0) * 0.5)
                r = self.map.urdf_to_servo_rad('i01_head_rothead_joint', a)
                self.apply_servo_cmd('head_tracker', 40, 0.5, False, [r[0]], [r[1]], [0.0])
            self.step(now - last)
            last = now
            js = None
            for ws, subs in list(self.clients.items()):
                s = subs.get('/urdf_joint_states')
                if s and now - s['last'] >= s['throttle']:
                    s['last'] = now
                    js = js or self.joint_states()
                    await self.send(ws, '/urdf_joint_states', js)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9090)
    ap.add_argument('--map', default=os.path.join(HERE, '..', 'config', 'servo_urdf_map.yaml'))
    ap.add_argument('--user-map', default='', help='save calibrations here (default: nowhere)')
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    robot = MockRobot(ServoUrdfMap.load(a.map), a.user_map)
    async with websockets.serve(robot.handler, '0.0.0.0', a.port):
        print(f'mock rosbridge robot on ws://0.0.0.0:{a.port}  ({len(robot.map.joints)} joints)')
        await robot.loop(a.demo)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
