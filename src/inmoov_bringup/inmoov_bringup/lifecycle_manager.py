#!/usr/bin/env python3
"""
InmoovLifecycleManager — tier-based lifecycle orchestrator for the InMoov Robot.

Brings nodes up tier by tier (0..6) sequentially, in parallel within each tier.
On failure: 3 attempts, then degraded (if the node isn't critical) or ABORT.

Execution model: every operation — the commands below, SLEEP/WAKE from
/robot_sleep, watchdog cascade/recovery — is put on ONE queue and executed by
ONE worker thread, one operation at a time. Operations never interleave (no
WAKE racing a cascade, no manual restart racing a recovery). Each operation
gets a generation number; per-node activation threads stop retrying once their
operation is no longer current, and SHUTDOWN pre-empts whatever is running.

Tier activation has ONE deadline (tier_advance_timeout_sec) for the whole tier.
A node still activating after it is marked degraded('timeout'); when its thread
finishes, a RECONCILE operation brings it to the state the system wants *now*
(e.g. deactivates it if the robot went to sleep meanwhile).

Watchdog: periodically polls get_state on all nodes (detection only). On process death:
  - marks degraded_reason='watchdog'
  - if the node is critical — queues a cascade deactivation of all tiers above it
  - on respawn (ros2 launch respawn=True) — queues re-activation of the node
    and of the cascaded tiers

Control via the /lifecycle/command topic:
  ACTIVATE        — bring the whole system up (automatic on startup); after
                    DEACTIVATE re-activates tiers 1..N, after a fault retries
                    the full activation
  DEACTIVATE      — deactivate everything except Foundation (tier 0); the
                    watchdog and SLEEP/WAKE transitions are paused until ACTIVATE
  SLEEP           — deactivate vision, keep voice/LLM
  WAKE            — reactivate after SLEEP
  SHUTDOWN        — shutdown all nodes (pre-empts the running operation)
  RESTART_TIER N  — restart a specific tier

Status is published on /lifecycle/status (JSON), including degraded_nodes[].

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import queue
import threading
import time
import yaml
from typing import Optional

import rclpy
from rclpy.node import Node
from lifecycle_msgs.msg import Transition, State
from lifecycle_msgs.srv import ChangeState, GetState
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


# lifecycle_msgs transitions
_CONFIGURE  = Transition.TRANSITION_CONFIGURE
_ACTIVATE   = Transition.TRANSITION_ACTIVATE
_DEACTIVATE = Transition.TRANSITION_DEACTIVATE
_CLEANUP    = Transition.TRANSITION_CLEANUP
_SHUTDOWN   = Transition.TRANSITION_UNCONFIGURED_SHUTDOWN

# Expected states
_STATE_INACTIVE     = State.PRIMARY_STATE_INACTIVE
_STATE_ACTIVE       = State.PRIMARY_STATE_ACTIVE
_STATE_UNCONFIGURED = State.PRIMARY_STATE_UNCONFIGURED

# Nodes deactivated on SLEEP (vision). voice_detector_node stays ACTIVE: the
# phrase right after the wake word must reach STT before WAKE finishes, and it
# costs nothing while idle (VAD runs only on activation / while recording).
_SLEEP_DEACTIVATE = {
    'face_capture_node', 'face_detection_node_left', 'face_detection_node_right',
    'face_tracker_node_left', 'face_tracker_node_right',
    'face_recognition_node', 'face_gallery_node',
    'emotion_recognition_node', 'vision_head_tracker_node',
    'human_detection_node', 'oak_node',
}

# Degradation reasons
_REASON_ACTIVATION = 'activation'   # failure during initial activation
_REASON_TIMEOUT    = 'timeout'      # still activating when the tier deadline passed
_REASON_WATCHDOG   = 'watchdog'     # process died (SIGKILL, OOM, segfault)
_REASON_CASCADE    = 'cascade'      # cascade deactivation due to a node dying below


class _Stale(Exception):
    """The operation this work belongs to is no longer current."""


class ManagedNode:
    """State of a single managed node."""

    def __init__(self, name: str, critical: bool, tier: int):
        self.name = name
        self.critical = critical
        self.tier = tier
        self.degraded = False
        self.degraded_reason = ''   # _REASON_* constant
        self.respawn_count = 0      # how many times the watchdog has brought the node back up
        self.recovering = False     # recovery queued or in progress
        self.attempts = 0
        self._change_state_cli = None
        self._get_state_cli = None

    def __repr__(self):
        status = f'DEGRADED({self.degraded_reason})' if self.degraded else 'ok'
        crit = 'CRIT' if self.critical else 'opt'
        return f'{self.name}[{crit},{status}]'


class InmoovLifecycleManager(Node):

    def __init__(self, **kwargs):
        super().__init__('lifecycle_manager', **kwargs)

        # Parameters
        self.declare_parameter('retry_count',                3)
        self.declare_parameter('retry_interval_sec',         10.0)
        self.declare_parameter('transition_timeout_sec',     30.0)
        self.declare_parameter('tier_advance_timeout_sec',   90.0)
        self.declare_parameter('config_file',                '')
        self.declare_parameter('autostart_delay_sec',        5.0)
        self.declare_parameter('watchdog_interval_sec',      5.0)
        self.declare_parameter('watchdog_startup_delay_sec', 15.0)
        self.declare_parameter('max_respawn_count',          5)
        # Comma-separated node names that the launch file didn't start (e.g. vision:=false,
        # telegram:=false). They are dropped from the tiers so activation doesn't sit
        # through retry/timeout waits for services that will never appear.
        self.declare_parameter('disabled_nodes',             '')

        self._retry_count             = self.get_parameter('retry_count').value
        self._retry_interval          = self.get_parameter('retry_interval_sec').value
        self._trans_timeout           = self.get_parameter('transition_timeout_sec').value
        self._tier_timeout            = self.get_parameter('tier_advance_timeout_sec').value
        self._watchdog_interval       = self.get_parameter('watchdog_interval_sec').value
        self._watchdog_startup_delay  = self.get_parameter('watchdog_startup_delay_sec').value
        self._max_respawn_count       = self.get_parameter('max_respawn_count').value
        autostart_delay               = self.get_parameter('autostart_delay_sec').value
        self._disabled_nodes = {
            n.strip() for n in self.get_parameter('disabled_nodes').value.split(',')
            if n.strip()
        }

        # Status publishing
        self._status_pub = self.create_publisher(String, '/lifecycle/status', 10)

        # Control commands
        self._cmd_sub = self.create_subscription(
            String, '/lifecycle/command', self._command_cb, 10)

        # Subscription to robot_sleep (latched)
        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._sleep_sub = self.create_subscription(
            Bool, '/robot_sleep', self._sleep_cb, _latched)

        # System state
        self._tiers: list[list[ManagedNode]] = []
        self._all_nodes: dict[str, ManagedNode] = {}
        self._system_state = 'starting'
        self._sleep_mode = False
        self._sleep_nodes = set(_SLEEP_DEACTIVATE)
        self._startup_timer = None
        self._watchdog_stop = threading.Event()
        self._watchdog_started = False

        # Serial execution: one queue, one worker
        self._ops: queue.Queue = queue.Queue()
        self._op_gen = 0                     # bumped at the start of every operation
        self._gen_lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                        name='lifecycle_worker')
        self._worker.start()

        # Status publishing timer
        self._status_timer = self.create_timer(2.0, self._publish_status)

        config_file = self.get_parameter('config_file').value
        if config_file:
            self._load_config_from_yaml(config_file)

        if self._tiers and autostart_delay > 0:
            self._startup_timer = self.create_timer(autostart_delay, self._autostart_cb)
            self.get_logger().info(
                f'InmoovLifecycleManager ready — autostart in {autostart_delay:.0f}s '
                f'({len(self._tiers)} tiers, {len(self._all_nodes)} nodes)')
        else:
            self._system_state = 'idle'   # no autostart — waits for ACTIVATE
            self.get_logger().info(
                'InmoovLifecycleManager ready — waiting for '
                + ('ACTIVATE' if self._tiers else 'tier configuration'))

    # ──────────────────────────────────────────────────────────────────────────
    # Config loading

    def _load_config_from_yaml(self, config_file: str):
        try:
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            params = cfg.get('lifecycle_manager', {}).get('ros__parameters', {})

            # YAML key → attribute actually read by the code
            attr_map = {
                'retry_count':                '_retry_count',
                'retry_interval_sec':         '_retry_interval',
                'transition_timeout_sec':     '_trans_timeout',
                'tier_advance_timeout_sec':   '_tier_timeout',
                'watchdog_interval_sec':      '_watchdog_interval',
                'watchdog_startup_delay_sec': '_watchdog_startup_delay',
                'max_respawn_count':          '_max_respawn_count',
            }
            for key, attr in attr_map.items():
                if key in params:
                    setattr(self, attr, params[key])

            tiers_cfg = params.get('tiers', [])
            self.configure_tiers(tiers_cfg)
            self.get_logger().info(
                f'Config loaded from {config_file}: '
                f'{len(self._tiers)} tiers, {len(self._all_nodes)} nodes')
        except Exception as e:
            self.get_logger().error(f'Error loading config {config_file}: {e}')

    def _autostart_cb(self):
        self.destroy_timer(self._startup_timer)
        self._startup_timer = None
        self.start_activation()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API

    def configure_tiers(self, tiers_cfg: list[dict]):
        """Initialize tiers from the config. Immediately creates all service clients."""
        for tier_cfg in tiers_cfg:
            tier_idx = len(self._tiers)
            tier_nodes = []
            critical_set = set(tier_cfg.get('critical', []))
            for node_name in tier_cfg['nodes']:
                if node_name in self._disabled_nodes:
                    continue
                mn = ManagedNode(node_name, critical=(node_name in critical_set), tier=tier_idx)
                self._all_nodes[node_name] = mn
                tier_nodes.append(mn)
            self._tiers.append(tier_nodes)

        # Pre-create clients from the main thread — thread-safe for create_client
        for mn in self._all_nodes.values():
            self._get_change_state_client(mn.name)
            self._get_get_state_client(mn.name)

        self.get_logger().info(
            f'Tier configuration: {len(self._tiers)} tiers, {len(self._all_nodes)} nodes')
        if self._disabled_nodes:
            self.get_logger().info(
                f'Disabled by launch (not managed): {sorted(self._disabled_nodes)}')

    def start_activation(self):
        self._enqueue('ACTIVATE_ALL')

    # ──────────────────────────────────────────────────────────────────────────
    # Operation queue

    def _enqueue(self, op: str, *args):
        self._ops.put((op, args))

    def _worker_loop(self):
        handlers = {
            'ACTIVATE_ALL': self._activate_all,
            'ACTIVATE':     self._do_activate,
            'DEACTIVATE':   self._do_deactivate,
            'SLEEP':        self._do_sleep,
            'WAKE':         self._do_wake,
            'SHUTDOWN':     self._do_shutdown,
            'RESTART_TIER': self._restart_tier,
            'CASCADE':      self._cascade_deactivate,
            'RECOVER':      self._recover_node,
            'RECONCILE':    self._reconcile_node,
        }
        while True:
            op, args = self._ops.get()
            if op == '_STOP':
                return
            gen = self._next_gen()
            if self._system_state == 'shutdown' and op != 'SHUTDOWN':
                continue
            try:
                handlers[op](gen, *args)
            except _Stale:
                self.get_logger().info(f'{op}: pre-empted')
            except Exception as e:
                self.get_logger().error(f'{op} failed: {e}')
            self._publish_status()

    def _next_gen(self) -> int:
        with self._gen_lock:
            self._op_gen += 1
            return self._op_gen

    def _is_stale(self, gen: int) -> bool:
        return gen != self._op_gen

    def _check(self, gen: int):
        if self._is_stale(gen):
            raise _Stale()

    def _sleep_unless_stale(self, sec: float, gen: int):
        deadline = time.monotonic() + sec
        while time.monotonic() < deadline:
            self._check(gen)
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _desired_active(self, mn: ManagedNode) -> bool:
        """Should this node be ACTIVE given the current system state?"""
        if self._system_state in ('shutdown', 'fault'):
            return False
        if self._system_state == 'deactivated' and mn.tier > 0:
            return False
        if self._sleep_mode and mn.name in self._sleep_nodes:
            return False
        return True

    def _settled_state(self) -> str:
        """System state once an operation completes normally."""
        if any(mn.degraded and mn.degraded_reason not in (_REASON_ACTIVATION, _REASON_TIMEOUT)
               for mn in self._all_nodes.values()):
            return 'degraded'   # watchdog / cascade damage still outstanding
        return 'sleep' if self._sleep_mode else 'active'

    # ──────────────────────────────────────────────────────────────────────────
    # Service clients

    def _get_change_state_client(self, node_name: str):
        mn = self._all_nodes[node_name]
        if mn._change_state_cli is None:
            mn._change_state_cli = self.create_client(
                ChangeState, f'/{node_name}/change_state')
        return mn._change_state_cli

    def _get_get_state_client(self, node_name: str):
        mn = self._all_nodes[node_name]
        if mn._get_state_cli is None:
            mn._get_state_cli = self.create_client(
                GetState, f'/{node_name}/get_state')
        return mn._get_state_cli

    # ──────────────────────────────────────────────────────────────────────────
    # Transitions

    def _send_transition(self, node_name: str, transition_id: int) -> bool:
        cli = self._get_change_state_client(node_name)
        if not cli.wait_for_service(timeout_sec=self._trans_timeout):
            self.get_logger().error(f'{node_name}: change_state service unavailable')
            return False
        req = ChangeState.Request()
        req.transition.id = transition_id
        future = cli.call_async(req)
        deadline = time.monotonic() + self._trans_timeout
        while not future.done():
            time.sleep(0.05)
            if time.monotonic() > deadline:
                self.get_logger().error(f'{node_name}: transition {transition_id} timed out')
                return False
        return future.result() is not None and future.result().success

    def _get_state(self, node_name: str, timeout_sec: float = 5.0) -> Optional[int]:
        cli = self._get_get_state_client(node_name)
        if not cli.wait_for_service(timeout_sec=timeout_sec):
            return None
        future = cli.call_async(GetState.Request())
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            time.sleep(0.05)
            if time.monotonic() > deadline:
                return None
        if future.result() is None:
            return None
        return future.result().current_state.id

    def _configure_and_activate(self, mn: ManagedNode, gen: Optional[int] = None) -> bool:
        """gen: stop before each transition if that operation is no longer current.
        A transition already sent always runs to completion."""
        state = self._get_state(mn.name)
        if state == _STATE_ACTIVE:
            return True
        if state != _STATE_INACTIVE:
            if gen is not None:
                self._check(gen)
            if not self._send_transition(mn.name, _CONFIGURE):
                return False
        if gen is not None:
            self._check(gen)
        if not self._send_transition(mn.name, _ACTIVATE):
            return False
        return self._get_state(mn.name) == _STATE_ACTIVE

    def _deactivate_if_active(self, node_name: str) -> bool:
        """DEACTIVATE only a node that is ACTIVE.

        An invalid transition (e.g. DEACTIVATE of an already INACTIVE node) makes
        rclpy raise inside the node's change_state service and kills the process.
        Only the worker thread calls this, so two DEACTIVATEs can't race.
        """
        if self._get_state(node_name) != _STATE_ACTIVE:
            return True
        return self._send_transition(node_name, _DEACTIVATE)

    def _shutdown_node(self, node_name: str) -> None:
        """State-aware DEACTIVATE → CLEANUP → SHUTDOWN (only valid transitions)."""
        state = self._get_state(node_name)
        if state == _STATE_ACTIVE and self._send_transition(node_name, _DEACTIVATE):
            state = _STATE_INACTIVE
        if state == _STATE_INACTIVE and self._send_transition(node_name, _CLEANUP):
            state = _STATE_UNCONFIGURED   # on_cleanup runs (nodes close DBs/devices there)
        if state == _STATE_UNCONFIGURED:
            self._send_transition(node_name, _SHUTDOWN)

    def _activate_node_with_retry(self, mn: ManagedNode, gen: int) -> bool:
        """Raises _Stale if the operation that asked for it is superseded."""
        for attempt in range(1, self._retry_count + 1):
            self._check(gen)
            mn.attempts = attempt
            self.get_logger().info(f'{mn.name}: attempt {attempt}/{self._retry_count}...')
            if self._configure_and_activate(mn, gen):
                self.get_logger().info(f'{mn.name}: ACTIVE ✓')
                return True
            if attempt < self._retry_count:
                wait = self._retry_interval * attempt
                self.get_logger().warn(f'{mn.name}: failed, waiting {wait:.0f}s...')
                self._sleep_unless_stale(wait, gen)
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # System activation

    def _activate_all(self, gen: int):
        self._system_state = 'starting'
        self._publish_status()

        for tier_idx, tier in enumerate(self._tiers):
            self._check(gen)
            self.get_logger().info(f'═══ Tier {tier_idx}: activating {len(tier)} nodes ═══')
            if not self._activate_tier(tier_idx, tier, gen):
                self.get_logger().error(
                    f'Tier {tier_idx}: critical node failed activation — ABORT')
                self._system_state = 'fault'
                return

        self._system_state = self._settled_state()
        self.get_logger().info('══ System fully activated ══')
        if self._sleep_mode:
            # /robot_sleep arrived during startup — honour it now
            self._deactivate_sleep_nodes()

        # Start the watchdog after full activation + startup_delay (once — ACTIVATE
        # after a fault re-runs this method)
        if self._watchdog_interval > 0 and not self._watchdog_started:
            self._watchdog_started = True
            threading.Thread(target=self._watchdog_start, daemon=True).start()

    def _activate_tier(self, tier_idx: int, tier: list[ManagedNode], gen: int) -> bool:
        """Parallel activation of a tier's nodes under ONE deadline. False = critical failure.

        A node still activating at the deadline is marked degraded('timeout') and
        the tier moves on; its thread keeps going (or stops if the operation is
        superseded) and then queues RECONCILE for that node.
        """
        results: dict[str, Optional[bool]] = {}
        late: set[str] = set()
        late_lock = threading.Lock()

        def activate_one(mn):
            try:
                ok = self._activate_node_with_retry(mn, gen)
            except _Stale:
                ok = None
            except Exception as e:
                self.get_logger().error(f'{mn.name}: exception during activation: {e}')
                ok = False
            with late_lock:
                results[mn.name] = ok
                is_late = mn.name in late
            if is_late:
                self._enqueue('RECONCILE', mn)

        threads = []
        for mn in tier:
            t = threading.Thread(target=activate_one, args=(mn,), daemon=True)
            threads.append((mn, t))
            t.start()
        deadline = time.monotonic() + self._tier_timeout
        for _, t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))

        stale = False
        with late_lock:
            for mn, _ in threads:
                if mn.name not in results:
                    late.add(mn.name)
                    mn.degraded = True
                    mn.degraded_reason = _REASON_TIMEOUT
                    self.get_logger().warn(
                        f'{mn.name}: still activating after the {self._tier_timeout:.0f}s '
                        f'tier deadline — degraded(timeout), will reconcile when it finishes')
                elif results[mn.name] is None:
                    stale = True
                elif not results[mn.name]:
                    mn.degraded = True
                    mn.degraded_reason = _REASON_ACTIVATION
        if stale:
            raise _Stale()

        degraded = [mn for mn in tier if mn.degraded]
        active   = [mn for mn in tier if not mn.degraded]
        if degraded:
            self.get_logger().warn(f'Tier {tier_idx} DEGRADED: {[mn.name for mn in degraded]}')
        if active:
            self.get_logger().info(f'Tier {tier_idx}: {len(active)}/{len(tier)} nodes active')
        for mn in degraded:
            if mn.critical:
                self.get_logger().error(f'CRITICAL node failed: {mn.name}')
                self._publish_status()
                return False
        self._publish_status()
        return True

    def _reconcile_node(self, gen: int, mn: ManagedNode):
        """A late activation finished: bring the node to what the system wants now."""
        state = self._get_state(mn.name)
        want = self._desired_active(mn)
        if state == _STATE_ACTIVE and not want:
            self.get_logger().info(f'Reconcile: {mn.name} came up late but must be INACTIVE now')
            self._deactivate_if_active(mn.name)
        if mn.degraded_reason != _REASON_TIMEOUT:
            return
        if state == _STATE_ACTIVE:
            mn.degraded = False
            mn.degraded_reason = ''
            self.get_logger().info(f'Reconcile: {mn.name} finished late — OK ✓')
        else:
            mn.degraded_reason = _REASON_ACTIVATION
            self.get_logger().warn(f'Reconcile: {mn.name} finished late and is not ACTIVE')

    # ──────────────────────────────────────────────────────────────────────────
    # Watchdog (detection only — actions go through the queue)

    def _watchdog_start(self):
        """Waits startup_delay, then starts the watchdog loop."""
        self.get_logger().info(
            f'Watchdog starting in {self._watchdog_startup_delay:.0f}s '
            f'(interval {self._watchdog_interval:.0f}s, max_respawn={self._max_respawn_count})')
        self._watchdog_stop.wait(timeout=self._watchdog_startup_delay)
        if self._watchdog_stop.is_set():
            return
        self.get_logger().info('Watchdog active')
        while not self._watchdog_stop.is_set():
            if self._system_state not in ('deactivated', 'shutdown'):   # nodes are INACTIVE on purpose
                self._watchdog_check_all()
            self._watchdog_stop.wait(timeout=self._watchdog_interval)

    def _watchdog_check_all(self):
        """Polls all nodes; reacts to death and resurrection.

        We detect two scenarios:
          A. state is None        → process died, respawn hasn't happened yet
          B. state == UNCONFIGURED → process died + ros2 launch already respawned it,
                                     the node is alive but not activated (fast respawn_delay)
        """
        for tier_idx, tier in enumerate(self._tiers):
            for mn in tier:
                # Don't touch nodes degraded for other reasons
                if mn.degraded and mn.degraded_reason != _REASON_WATCHDOG:
                    continue

                state = self._get_state(mn.name, timeout_sec=2.0)

                if not mn.degraded:
                    # The node should be ACTIVE (or INACTIVE on purpose) — died otherwise
                    if state is None or state == _STATE_UNCONFIGURED:
                        cause = 'not responding' if state is None else 'in UNCONFIGURED (crash+respawn)'
                        self.get_logger().error(f'Watchdog: {mn.name} {cause} — degraded!')
                        mn.degraded = True
                        mn.degraded_reason = _REASON_WATCHDOG
                        self._publish_status()
                        if mn.critical:
                            self._enqueue('CASCADE', tier_idx)
                        # Scenario B: process already alive → recovery right after the cascade
                        if state == _STATE_UNCONFIGURED:
                            self._start_recovery(mn, tier_idx)

                elif mn.degraded_reason == _REASON_WATCHDOG and not mn.recovering:
                    # Scenario A: we were waiting for respawn — check whether the node finally appeared
                    if state is not None:
                        self._start_recovery(mn, tier_idx)

    def _start_recovery(self, mn: ManagedNode, tier_idx: int):
        """Queues recovery if the limit hasn't been exhausted."""
        if mn.recovering:
            return
        if mn.respawn_count >= self._max_respawn_count:
            self.get_logger().error(
                f'Watchdog: {mn.name} reached max_respawn_count={self._max_respawn_count}')
            return
        mn.respawn_count += 1
        mn.recovering = True
        self.get_logger().info(
            f'Watchdog: {mn.name} queueing recovery '
            f'(attempt {mn.respawn_count}/{self._max_respawn_count})')
        self._enqueue('RECOVER', mn, tier_idx)

    def _cascade_deactivate(self, gen: int, failed_tier_idx: int):
        """Deactivates all tiers above failed_tier_idx (they depended on the failed node)."""
        self.get_logger().warn(
            f'Cascade deactivation: tiers {failed_tier_idx+1}…{len(self._tiers)-1}')
        for tier_idx in range(len(self._tiers) - 1, failed_tier_idx, -1):
            active = [mn for mn in self._tiers[tier_idx] if not mn.degraded]
            if not active:
                continue
            self._parallel(self._deactivate_if_active, [mn.name for mn in active],
                           self._trans_timeout)
            for mn in active:
                mn.degraded = True
                mn.degraded_reason = f'{_REASON_CASCADE}_{failed_tier_idx}'
        self._system_state = 'degraded'

    def _recover_node(self, gen: int, mn: ManagedNode, tier_idx: int):
        """Re-activates a resurrected node; on success — brings the cascaded tiers back up.

        Runs after any CASCADE queued before it, so the cascade marks are final.
        """
        mn.degraded = True  # keep degraded for the duration of the attempt
        try:
            ok = self._activate_node_with_retry(mn, gen)
            # Asleep (or DEACTIVATE'd meanwhile): the node must not stay ACTIVE —
            # bring it back up (configure) but leave it INACTIVE like its siblings.
            if ok and not self._desired_active(mn):
                self._deactivate_if_active(mn.name)
        except Exception as e:
            self.get_logger().error(f'Watchdog: {mn.name} recovery error: {e!r}')
            ok = False
        finally:
            mn.recovering = False
        if not ok:
            mn.degraded = True
            mn.degraded_reason = _REASON_WATCHDOG
            self.get_logger().error(f'Watchdog: {mn.name} failed to recover')
            return
        mn.degraded = False
        mn.degraded_reason = ''
        self.get_logger().info(f'Watchdog: {mn.name} successfully recovered ✓')
        if mn.critical:
            self._cascade_recover(gen, tier_idx)
        elif self._system_state == 'degraded':
            self._system_state = self._settled_state()

    def _cascade_recover(self, gen: int, recovered_tier_idx: int):
        """After a critical node recovers — brings cascade-degraded tiers back up."""
        cascade_prefix = f'{_REASON_CASCADE}_{recovered_tier_idx}'
        for tier_idx in range(recovered_tier_idx + 1, len(self._tiers)):
            to_recover = [mn for mn in self._tiers[tier_idx]
                          if mn.degraded_reason == cascade_prefix]
            if not to_recover:
                continue
            self.get_logger().info(
                f'Cascade recovery of tier {tier_idx} ({len(to_recover)} nodes)...')
            for mn in to_recover:
                mn.degraded = False
                mn.degraded_reason = ''
                mn.attempts = 0
            self._activate_tier(tier_idx, to_recover, gen)
            if self._sleep_mode:   # asleep — SLEEP-set nodes go back to INACTIVE
                for mn in to_recover:
                    if mn.name in self._sleep_nodes:
                        self._deactivate_if_active(mn.name)
            self._publish_status()

        self._system_state = self._settled_state()
        if self._system_state != 'degraded':
            self.get_logger().info('Watchdog: full system recovery ✓')

    # ──────────────────────────────────────────────────────────────────────────
    # SLEEP / WAKE

    def _sleep_cb(self, msg: Bool):
        if msg.data and not self._sleep_mode:
            self.get_logger().info('robot_sleep → SLEEP: deactivating vision')
            self._sleep_mode = True
            self._enqueue('SLEEP')
        elif not msg.data and self._sleep_mode:
            self.get_logger().info('robot_sleep → WAKE: reactivating vision')
            self._sleep_mode = False
            self._enqueue('WAKE')

    def _deactivate_sleep_nodes(self):
        names = [n for n in self._sleep_nodes
                 if n in self._all_nodes and not self._all_nodes[n].degraded]
        self._parallel(self._deactivate_if_active, names, self._trans_timeout)

    def _do_sleep(self, gen: int):
        # _sleep_mode is the desired state; the queue may hold SLEEP, WAKE, SLEEP…
        # so each op applies the CURRENT desire (a stale SLEEP after WAKE is a no-op).
        if not self._sleep_mode or self._system_state in ('deactivated', 'starting', 'fault', 'idle'):
            return   # ACTIVATE / the end of startup honours _sleep_mode
        self._deactivate_sleep_nodes()
        self._system_state = self._settled_state()

    def _do_wake(self, gen: int):
        if self._sleep_mode or self._system_state in ('deactivated', 'starting', 'fault', 'idle'):
            return
        self._system_state = 'waking'
        self._publish_status()
        names = [n for n in self._sleep_nodes
                 if n in self._all_nodes and not self._all_nodes[n].degraded]
        self._parallel(lambda n: self._configure_and_activate(self._all_nodes[n]),
                       names, 2 * self._trans_timeout)
        self._system_state = self._settled_state()

    # ──────────────────────────────────────────────────────────────────────────
    # Control commands

    def _command_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        self.get_logger().info(f'/lifecycle/command: {cmd}')

        if cmd == 'SHUTDOWN':
            self._watchdog_stop.set()
            self._next_gen()               # pre-empt the running operation
            self._enqueue('SHUTDOWN')
        elif cmd in ('ACTIVATE', 'DEACTIVATE'):
            self._enqueue(cmd)
        elif cmd == 'SLEEP':
            self._sleep_mode = True
            self._enqueue('SLEEP')
        elif cmd == 'WAKE':
            self._sleep_mode = False
            self._enqueue('WAKE')
        elif cmd.startswith('RESTART_TIER '):
            try:
                self._enqueue('RESTART_TIER', int(cmd.split()[1]))
            except (IndexError, ValueError):
                self.get_logger().error(f'Invalid format: {cmd}')
        else:
            self.get_logger().warn(f'Unknown command: {cmd}')

    def _do_deactivate(self, gen: int):
        """DEACTIVATE: deactivates tiers N..1 (top-down); Foundation (tier 0) stays active."""
        if self._system_state in ('deactivated', 'shutdown'):
            self.get_logger().warn(f'DEACTIVATE ignored in state {self._system_state}')
            return
        self._system_state = 'deactivated'
        self._publish_status()
        for tier in reversed(self._tiers[1:]):
            self._parallel(self._deactivate_if_active,
                           [mn.name for mn in tier if not mn.degraded], self._trans_timeout)
        self.get_logger().info('DEACTIVATE done: only Foundation (tier 0) is active')

    def _do_activate(self, gen: int):
        """ACTIVATE: brings the system (back) up.

        After DEACTIVATE — re-activates tiers 1..N (every node gets a fresh retry
        budget; SLEEP-set nodes stay inactive if the robot is asleep). After a fault
        or if the initial activation never ran — runs the full tier 0..N activation.
        """
        state = self._system_state
        if state == 'shutdown':
            self.get_logger().warn('ACTIVATE ignored after SHUTDOWN')
            return
        if state not in ('deactivated', 'fault') and self._startup_timer is None \
                and self._watchdog_started:
            self.get_logger().info(f'ACTIVATE: system already up ({state})')
            return
        if state != 'deactivated':
            if self._startup_timer is not None:
                self.destroy_timer(self._startup_timer)
                self._startup_timer = None
            self._activate_all(gen)
            return

        self._system_state = 'starting'
        self._publish_status()
        for tier_idx, tier in enumerate(self._tiers):
            if tier_idx == 0:
                continue
            nodes = [mn for mn in tier
                     if not (self._sleep_mode and mn.name in self._sleep_nodes)]
            for mn in nodes:
                mn.degraded = False
                mn.degraded_reason = ''
                mn.attempts = 0
            if not self._activate_tier(tier_idx, nodes, gen):
                self._system_state = 'fault'
                return
        self._system_state = self._settled_state()
        self.get_logger().info('ACTIVATE done')

    def _restart_tier(self, gen: int, tier_idx: int):
        if tier_idx >= len(self._tiers):
            self.get_logger().error(f'Tier {tier_idx} does not exist')
            return
        tier = [mn for mn in self._tiers[tier_idx] if self._desired_active(mn)]
        for mn in tier:
            mn.degraded = False
            mn.degraded_reason = ''
            mn.attempts = 0
        self.get_logger().info(f'Restarting tier {tier_idx}...')
        self._activate_tier(tier_idx, tier, gen)

    def _do_shutdown(self, gen: int):
        # Stop the watchdog before shutdown
        self._watchdog_stop.set()
        self._system_state = 'shutdown'
        self._publish_status()
        for tier in reversed(self._tiers):
            self._parallel(self._shutdown_node, [mn.name for mn in tier],
                           3 * self._trans_timeout)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers

    @staticmethod
    def _parallel(fn, items: list, timeout: float):
        """fn(item) for every item in parallel; waits up to `timeout` in total."""
        threads = [threading.Thread(target=fn, args=(i,), daemon=True) for i in items]
        for t in threads:
            t.start()
        deadline = time.monotonic() + timeout
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))

    # ──────────────────────────────────────────────────────────────────────────
    # Status

    def _publish_status(self):
        degraded_nodes   = [mn.name for tier in self._tiers for mn in tier if mn.degraded]
        recovering_nodes = [mn.name for tier in self._tiers for mn in tier if mn.recovering]
        status = {
            'system':          self._system_state,
            'sleep_mode':      self._sleep_mode,
            'degraded_nodes':  degraded_nodes,
            'recovering_nodes': recovering_nodes,
            'pending_ops':     self._ops.qsize(),
            'tiers': [
                {
                    'id': tier_idx,
                    'nodes': {
                        mn.name: {
                            'critical':        mn.critical,
                            'degraded':        mn.degraded,
                            'degraded_reason': mn.degraded_reason,
                            'respawn_count':   mn.respawn_count,
                            'recovering':      mn.recovering,
                            'attempts':        mn.attempts,
                        }
                        for mn in tier
                    },
                }
                for tier_idx, tier in enumerate(self._tiers)
            ],
        }
        try:
            self._status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))
        except Exception:
            pass   # context already shut down


def main(args=None):
    rclpy.init(args=args)
    node = InmoovLifecycleManager()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
