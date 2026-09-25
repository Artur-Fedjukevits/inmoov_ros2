"""
test_lifecycle_manager.py — the lifecycle manager against fake in-process lifecycle nodes.

Covers the serial-queue model: tier deadline + late-node reconcile, SLEEP/WAKE
bursts, a late node reconciled into sleep, critical crash → cascade → recovery
(ordered by the queue, no sleep), SHUTDOWN pre-empting a retry loop.

Run:
  cd ~/ros2_ws && source install/setup.bash
  python3 -m pytest src/inmoov_bringup/test/test_lifecycle_manager.py -v
"""

import itertools
import os
import sys
import threading
import time

import pytest
import rclpy
from lifecycle_msgs.msg import State
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_bringup.lifecycle_manager import InmoovLifecycleManager  # noqa: E402

ACTIVE, INACTIVE, UNCONFIGURED, FINALIZED = (
    State.PRIMARY_STATE_ACTIVE, State.PRIMARY_STATE_INACTIVE,
    State.PRIMARY_STATE_UNCONFIGURED, State.PRIMARY_STATE_FINALIZED)
_uid = itertools.count()


class FakeNode(LifecycleNode):
    def __init__(self, name, activate_delay=0.0, fail_activate=False):
        super().__init__(name)
        self.activate_delay = activate_delay
        self.fail_activate = fail_activate

    def on_configure(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        time.sleep(self.activate_delay)
        if self.fail_activate:
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def state(self):
        return self._state_machine.current_state[0]


@pytest.fixture(scope='module', autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class Rig:
    """Fake nodes + manager spinning in one MultiThreadedExecutor."""

    def __init__(self, tiers, critical=(), node_kwargs=None, **params):
        sfx = next(_uid)
        node_kwargs = node_kwargs or {}
        self.names = {}   # short → unique ROS name
        self.nodes = {}
        cfg = []
        for tier in tiers:
            full = []
            for short in tier:
                name = f'{short}_{sfx}'
                self.names[short] = name
                self.nodes[short] = FakeNode(name, **node_kwargs.get(short, {}))
                full.append(name)
            cfg.append({'nodes': full, 'critical': [self.names[c] for c in critical
                                                    if c in tier]})
        defaults = dict(retry_count=2, retry_interval_sec=0.5, transition_timeout_sec=5.0,
                        tier_advance_timeout_sec=10.0, autostart_delay_sec=0.0,
                        watchdog_interval_sec=0.0, watchdog_startup_delay_sec=0.0)
        defaults.update(params)
        self.mgr = InmoovLifecycleManager(parameter_overrides=[
            Parameter(k, value=v) for k, v in defaults.items()])
        self.mgr.configure_tiers(cfg)
        self.ex = MultiThreadedExecutor(num_threads=16)
        for n in [self.mgr, *self.nodes.values()]:
            self.ex.add_node(n)
        self._spin = threading.Thread(target=self.ex.spin, daemon=True)
        self._spin.start()

    def sleep_nodes(self, *shorts):
        self.mgr._sleep_nodes = {self.names[s] for s in shorts}

    def state(self, short):
        return self.nodes[short].state()

    def mn(self, short):
        return self.mgr._all_nodes[self.names[short]]

    def close(self):
        self.mgr._watchdog_stop.set()
        self.mgr._enqueue('_STOP')
        self.ex.shutdown(timeout_sec=1.0)
        for n in [self.mgr, *self.nodes.values()]:
            n.destroy_node()


@pytest.fixture
def rigs():
    made = []
    yield lambda *a, **kw: made.append(Rig(*a, **kw)) or made[-1]
    for r in made:
        r.close()


def wait_for(pred, timeout=15.0, msg=''):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f'timed out waiting for: {msg or pred}')


# ─────────────────────────────────────────────────────────────────────────────

def test_full_activation(rigs):
    r = rigs([['a'], ['b', 'c'], ['d']], critical=['a'])
    r.mgr.start_activation()
    wait_for(lambda: r.mgr._system_state == 'active', msg='system active')
    assert all(r.state(s) == ACTIVE for s in 'abcd')


def test_tier_deadline_then_late_node_reconciled(rigs):
    r = rigs([['a', 'slow'], ['b']], node_kwargs={'slow': {'activate_delay': 3.0}},
             tier_advance_timeout_sec=1.0)
    t0 = time.monotonic()
    r.mgr.start_activation()
    wait_for(lambda: r.state('b') == ACTIVE, msg='tier 1 activated')
    assert time.monotonic() - t0 < 2.5, 'tier 0 waited for the slow node past its deadline'
    assert r.mn('slow').degraded_reason in ('timeout', '')
    # The late node finishes and RECONCILE clears the timeout mark
    wait_for(lambda: r.state('slow') == ACTIVE and not r.mn('slow').degraded,
             msg='slow node reconciled')
    wait_for(lambda: r.mgr._system_state == 'active')


def test_sleep_wake_burst_settles_on_last_request(rigs):
    r = rigs([['a'], ['cam', 'mic']])
    r.sleep_nodes('cam')
    r.mgr.start_activation()
    wait_for(lambda: r.mgr._system_state == 'active')

    for v in (True, False, True, False, True):
        r.mgr._sleep_cb(Bool(data=v))
    wait_for(lambda: r.mgr._ops.empty() and r.mgr._system_state == 'sleep', msg='asleep')
    time.sleep(0.3)
    assert r.state('cam') == INACTIVE and r.state('mic') == ACTIVE

    for v in (False, True, False):
        r.mgr._sleep_cb(Bool(data=v))
    wait_for(lambda: r.mgr._ops.empty() and r.mgr._system_state == 'active', msg='awake')
    wait_for(lambda: r.state('cam') == ACTIVE)


def test_late_node_reconciled_into_sleep(rigs):
    r = rigs([['a', 'cam']], node_kwargs={'cam': {'activate_delay': 2.0}},
             tier_advance_timeout_sec=0.5)
    r.sleep_nodes('cam')
    r.mgr.start_activation()
    wait_for(lambda: r.mn('cam').degraded_reason == 'timeout', msg='cam timed out')
    r.mgr._sleep_cb(Bool(data=True))      # robot falls asleep while cam is still activating
    wait_for(lambda: r.state('cam') == INACTIVE and not r.mn('cam').degraded, timeout=10,
             msg='cam reconciled to INACTIVE')


def test_critical_crash_cascade_then_recover(rigs):
    r = rigs([['base'], ['mid'], ['top']], critical=['base'],
             watchdog_interval_sec=0.3, watchdog_startup_delay_sec=0.0)
    r.mgr.start_activation()
    wait_for(lambda: r.mgr._system_state == 'active')
    wait_for(lambda: r.mgr._watchdog_started)

    # "crash + respawn": the node is alive but UNCONFIGURED
    r.nodes['base'].trigger_deactivate()
    r.nodes['base'].trigger_cleanup()
    assert r.state('base') == UNCONFIGURED

    wait_for(lambda: r.mgr._system_state == 'degraded', msg='cascade')
    wait_for(lambda: r.mgr._system_state == 'active' and r.mgr._ops.empty(), timeout=20,
             msg='full recovery')
    assert all(r.state(s) == ACTIVE for s in ('base', 'mid', 'top'))
    assert not any(r.mn(s).degraded for s in ('base', 'mid', 'top'))


def test_shutdown_preempts_retry_loop(rigs):
    r = rigs([['a', 'broken']], node_kwargs={'broken': {'fail_activate': True}},
             retry_count=5, retry_interval_sec=30.0)
    r.mgr.start_activation()
    wait_for(lambda: r.mn('broken').attempts >= 1, msg='first attempt')
    time.sleep(0.3)                       # now inside the 30 s back-off
    t0 = time.monotonic()
    r.mgr._command_cb(String(data='SHUTDOWN'))
    wait_for(lambda: r.mgr._system_state == 'shutdown', timeout=5, msg='shutdown started')
    assert time.monotonic() - t0 < 3.0
    wait_for(lambda: r.state('a') == FINALIZED and r.state('broken') == FINALIZED,
             msg='nodes shut down')


# ─────────────────────────────────────────────────────────────────────────────
# Regressions from the 2026-09-26 review: the requested mode must win over
# late transitions, queued recoveries and queued activations.

def test_shutdown_during_slow_activation_leaves_nothing_active(rigs):
    # Tier deadline shorter than the activation → 'slow' becomes a late node
    # (8 s > the 5 s get_state timeout: shutdown can't even read its state meanwhile)
    r = rigs([['a', 'slow']], node_kwargs={'slow': {'activate_delay': 8.0}},
             tier_advance_timeout_sec=0.5)
    r.mgr.start_activation()
    wait_for(lambda: r.mn('slow').degraded_reason == 'timeout', msg='slow node late')
    r.mgr._command_cb(String(data='SHUTDOWN'))
    wait_for(lambda: r.state('slow') in (FINALIZED, UNCONFIGURED, INACTIVE) and
             r.state('a') == FINALIZED, timeout=40,
             msg='both nodes shut down, the late one too')
    time.sleep(1.0)
    assert r.state('slow') != ACTIVE


def test_queued_recovery_respects_manual_deactivate(rigs):
    r = rigs([['base'], ['top']], critical=['base'])
    r.mgr.start_activation()
    wait_for(lambda: r.mgr._system_state == 'active')

    # DEACTIVATE is queued first; the crash's CASCADE + RECOVER land behind it
    r.mgr._enqueue('DEACTIVATE')
    r.nodes['base'].trigger_deactivate()
    r.nodes['base'].trigger_cleanup()
    base = r.mn('base')
    base.degraded, base.degraded_reason = True, 'watchdog'
    r.mgr._enqueue('CASCADE', 0)
    r.mgr._start_recovery(base, 0)

    wait_for(lambda: r.mgr._ops.empty() and not base.recovering, timeout=20, msg='queue drained')
    time.sleep(0.5)
    assert r.state('base') == ACTIVE              # Foundation recovered (tier 0 stays up)
    assert r.state('top') == INACTIVE             # …but DEACTIVATE is honoured
    assert r.mgr._system_state == 'deactivated'


def test_shutdown_drops_queued_activation(rigs):
    r = rigs([['a'], ['b']])
    r.mgr.start_activation()
    wait_for(lambda: r.mgr._system_state == 'active')
    r.mgr._enqueue('DEACTIVATE')
    wait_for(lambda: r.state('b') == INACTIVE)

    ran = []
    orig_sleep = r.mgr._do_sleep
    r.mgr._do_sleep = lambda gen: (time.sleep(1.0), orig_sleep(gen))   # keep the worker busy
    orig_activate = r.mgr._do_activate
    r.mgr._do_activate = lambda gen: (ran.append('ACTIVATE'), orig_activate(gen))
    r.mgr._enqueue('SLEEP')
    r.mgr._enqueue('ACTIVATE')                    # queued before the shutdown request…
    r.mgr._command_cb(String(data='SHUTDOWN'))    # …must not run after it
    wait_for(lambda: r.state('a') == FINALIZED and r.state('b') == FINALIZED, timeout=20)
    assert ran == []
