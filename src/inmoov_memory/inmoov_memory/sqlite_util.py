"""
sqlite_util.py — one place for SQLite connection settings.

Every memory DB (social, episodic, semantic, reminders) is touched from several
threads and processes (memory_node, telegram_bridge, maintenance scripts):
  - WAL: readers don't block the writer and vice versa;
  - busy_timeout: a writer waits for the lock instead of failing at once with
    "database is locked".
foreign_keys is intentionally left off: the existing DBs predate it and may hold
orphan rows.

Backups of a WAL database must include the -wal file (or run `PRAGMA
wal_checkpoint` first).

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import sqlite3
from contextlib import contextmanager

BUSY_TIMEOUT_MS = 5000


def connect(path: str, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a connection with WAL + busy_timeout."""
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000,
                           check_same_thread=check_same_thread)
    conn.execute(f'PRAGMA busy_timeout={BUSY_TIMEOUT_MS}')
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


@contextmanager
def session(path: str, row_factory=None):
    """Short-lived connection: commit on success, rollback on error, always close."""
    conn = connect(path)
    if row_factory is not None:
        conn.row_factory = row_factory
    try:
        with conn:
            yield conn
    finally:
        conn.close()
