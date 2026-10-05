"""
joint_arbiter.py — per-joint ownership ("leases") for servo commands. ROS-free.

A command from `source` with `priority` for a joint is accepted if:
  - the joint has no owner, or the owner's lease has expired, or
  - the owner is the same source, or
  - the command's priority is strictly higher than the owner's.
An accepted command with lease_sec > 0 makes `source` the owner until
now + lease_sec; with lease_sec == 0 it is a plain write that takes no
ownership (and doesn't disturb an existing owner of the same source).
release() drops ownership only if `source` is the owner.

Joints in the same group share one lease (the mirrored eyes: eye_lr_L/eye_lr_R).

See inmoov_msgs/msg/JointCommand.msg for the priorities in use.

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import time
from dataclasses import dataclass


@dataclass
class Lease:
    source: str
    priority: int
    expires: float


class JointArbiter:

    def __init__(self, groups: dict[str, str] | None = None, clock=time.monotonic):
        """groups: joint name → group key; joints not listed are their own group."""
        self._groups = groups or {}
        self._clock  = clock
        self._leases: dict[str, Lease] = {}
        self.rejected = 0

    def _key(self, joint: str) -> str:
        return self._groups.get(joint, joint)

    def _owner(self, key: str, now: float) -> Lease | None:
        lease = self._leases.get(key)
        if lease is not None and lease.expires <= now:
            del self._leases[key]
            return None
        return lease

    def claim(self, joint: str, source: str, priority: int, lease_sec: float) -> tuple[bool, Lease | None]:
        """Returns (accepted, blocking_owner). blocking_owner is set only when rejected."""
        now   = self._clock()
        key   = self._key(joint)
        owner = self._owner(key, now)
        if owner is not None and owner.source != source and priority <= owner.priority:
            self.rejected += 1
            return False, owner
        if lease_sec > 0:
            self._leases[key] = Lease(source, priority, now + lease_sec)
        return True, None

    def release(self, joint: str, source: str) -> bool:
        key = self._key(joint)
        lease = self._leases.get(key)
        if lease is not None and lease.source == source:
            del self._leases[key]
            return True
        return False

    def owners(self) -> dict[str, dict]:
        """Live leases: group → {source, priority, remaining_sec}."""
        now = self._clock()
        return {
            key: {'source': lease.source, 'priority': lease.priority,
                  'remaining_sec': round(lease.expires - now, 2)}
            for key in list(self._leases)
            if (lease := self._owner(key, now)) is not None
        }
