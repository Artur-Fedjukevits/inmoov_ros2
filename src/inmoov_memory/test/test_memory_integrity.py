"""
test_memory_integrity.py — merge_persons, embedding validation, Chroma reconcile, SQLite
settings. Everything runs on temp files; no ROS graph, LLM or ChromaDB needed.

Run:
  cd /home/artur/ros2_ws && source install/setup.bash
  python3 -m pytest src/inmoov_memory/test/test_memory_integrity.py -v
"""

import json
import os
import sys

import numpy as np
import pytest
import rclpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_memory.episodic_memory import EpisodicMemory    # noqa: E402
from inmoov_memory.memory_node import MemoryNode, as_embedding  # noqa: E402
from inmoov_memory.semantic_memory import SemanticMemory, _chroma_id  # noqa: E402
from inmoov_memory.sqlite_util import connect, session       # noqa: E402

FACE_DIM, VOICE_DIM = 512, 192


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module', autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def _vec(seed, dim):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_person(node, pid, name, n_photos, n_voice, seed):
    db = node._db
    db.execute('INSERT INTO persons (id, name, embedding, first_seen, last_seen, meet_count) '
               'VALUES (?,?,?,?,?,?)', (pid, name, _vec(seed, FACE_DIM).tobytes(), 'x', 'x', 2))
    pdir = os.path.join(node._gallery_dir, 'persons', f'{pid}_{name}')
    os.makedirs(pdir)
    for i in range(n_photos):
        path = os.path.join(pdir, f'face_{i:03d}.jpg')
        with open(path, 'wb') as f:
            f.write(f'{name}-{i}'.encode())
        db.execute('INSERT INTO person_gallery (person_id, photo_path, embedding, quality, created_at) '
                   'VALUES (?,?,?,?,?)',
                   (pid, path, _vec(seed * 100 + i, FACE_DIM).tobytes(), 1.0 - i * 0.01, 'x'))
    for i in range(n_voice):
        db.execute('INSERT INTO voice_gallery (person_id, embedding, recorded_at) VALUES (?,?,?)',
                   (pid, _vec(seed * 1000 + i, VOICE_DIM).tobytes(), float(i)))
    db.execute('INSERT INTO person_notes (person_id, key, value) VALUES (?,?,?)',
               (pid, f'note_{name}', 'v'))
    db.commit()
    return pdir


@pytest.fixture
def node(tmp_path):
    n = MemoryNode()
    n._db = connect(str(tmp_path / 'social.db'), check_same_thread=False)
    n._gallery_dir = str(tmp_path / 'faces')
    n._episodic_db_path = str(tmp_path / 'episodic.db')
    n._init_social_db()
    EpisodicMemory(n._episodic_db_path).save('встреча', participants=['Никол', 'Артур'])
    yield n
    n._db.close()
    n.destroy_node()


def _load_caches(node):
    node._load_gallery_cache()
    node._load_voice_gallery_cache()


# ─────────────────────────────────────────────────────────────────────────────
# merge_persons
# ─────────────────────────────────────────────────────────────────────────────

def test_merge_keeps_transferred_photos(node):
    _add_person(node, 1, 'Артур', n_photos=28, n_voice=3, seed=1)
    from_dir = _add_person(node, 2, 'Никол', n_photos=4, n_voice=2, seed=2)
    _load_caches(node)

    res = node._merge_persons({'from_id': 2, 'to_id': 1})
    assert res.get('merged'), res

    rows = node._db.execute('SELECT person_id, photo_path FROM person_gallery').fetchall()
    assert len(rows) == 30                       # 28 + 2 transferred, 2 discarded (limit 30)
    assert {pid for pid, _ in rows} == {1}
    for _, path in rows:
        assert os.path.exists(path), f'gallery row points to a missing file: {path}'
        assert '/persons/1_' in path
    assert not os.path.exists(from_dir)
    assert node._db.execute('SELECT COUNT(*) FROM persons WHERE id=2').fetchone()[0] == 0

    # Caches match the DB exactly (discarded photos are not in the cache)
    assert len(node._gallery_cache[1]) == 30 and 2 not in node._gallery_cache
    assert len(node._voice_gallery_cache[1]) == 5 and 2 not in node._voice_gallery_cache

    notes = {k for (k,) in node._db.execute('SELECT key FROM person_notes WHERE person_id=1')}
    assert notes == {'note_Артур', 'note_Никол'}

    with session(node._episodic_db_path) as c:
        parts = json.loads(c.execute('SELECT participants FROM episodes').fetchone()[0])
    assert parts == ['Артур', 'Артур']


def test_merge_rolls_back_on_db_error(node):
    to_dir = _add_person(node, 1, 'Артур', n_photos=2, n_voice=1, seed=1)
    from_dir = _add_person(node, 2, 'Никол', n_photos=2, n_voice=1, seed=2)
    before = node._db.execute('SELECT id, person_id, photo_path FROM person_gallery').fetchall()
    # Fails mid-transaction, after the gallery UPDATEs already ran
    node._db.execute('DROP TABLE voice_gallery')
    node._db.commit()

    res = node._merge_persons({'from_id': 2, 'to_id': 1})
    assert 'error' in res

    after = node._db.execute('SELECT id, person_id, photo_path FROM person_gallery').fetchall()
    assert after == before                                   # rolled back
    assert node._db.execute('SELECT COUNT(*) FROM persons').fetchone()[0] == 2
    assert os.path.isdir(from_dir) and len(os.listdir(from_dir)) == 2
    assert len(os.listdir(to_dir)) == 2                      # temp copies removed


def test_merge_same_id_rejected(node):
    _add_person(node, 1, 'Артур', n_photos=1, n_voice=0, seed=1)
    assert 'error' in node._merge_persons({'from_id': 1, 'to_id': 1})
    assert node._db.execute('SELECT COUNT(*) FROM persons').fetchone()[0] == 1


def test_gallery_add_for_merged_person_rejected(node):
    _add_person(node, 1, 'Артур', n_photos=0, n_voice=0, seed=1)
    res = node._gallery_add({'person_id': 99, 'photo_path': '/tmp/x.jpg',
                             'embedding': _vec(5, FACE_DIM).tolist()})
    assert res == {'added': False, 'reason': 'person_not_found'}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding validation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('raw', [
    [0.1] * 511,                       # wrong length
    [[0.1] * 512],                     # wrong shape
    [float('nan')] + [0.1] * 511,      # NaN
    [float('inf')] + [0.1] * 511,      # Inf
    [0.0] * 512,                       # zero norm
])
def test_as_embedding_rejects(raw):
    with pytest.raises(ValueError):
        as_embedding(raw, FACE_DIM)


def test_as_embedding_normalizes():
    emb = as_embedding([3.0] * FACE_DIM, FACE_DIM)
    assert emb.dtype == np.float32 and abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_bad_voice_embedding_not_stored(node):
    _add_person(node, 1, 'Артур', n_photos=0, n_voice=0, seed=1)
    res = node._add_voice_to_gallery({'person_id': 1, 'embedding': [0.1] * 100})
    assert res['added'] is False and res['reason'].startswith('invalid_embedding')
    assert node._db.execute('SELECT COUNT(*) FROM voice_gallery').fetchone()[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Chroma reconcile (fake collection — no model download)
# ─────────────────────────────────────────────────────────────────────────────

class FakeCollection:
    def __init__(self):
        self.items = {}   # id → metadata
        self.upserted = []

    def get(self, include=None):
        return {'ids': list(self.items), 'metadatas': list(self.items.values())}

    def upsert(self, ids, documents, metadatas):
        self.upserted += ids
        self.items.update(zip(ids, metadatas))

    def delete(self, ids):
        for i in ids:
            self.items.pop(i, None)


def test_reconcile_chroma(tmp_path):
    sm = SemanticMemory.__new__(SemanticMemory)
    sm.db_path, sm._chroma = str(tmp_path / 'sem.db'), None
    sm._init_db()
    sm.save_fact('Артур', 'любит', 'чай')          # Chroma was down: SQLite only
    sm.save_fact('Артур', 'живёт', 'Таллин')
    sm.save_fact('Никол', 'любит', 'кофе')

    fake = FakeCollection()
    sm._chroma = fake
    res = sm.reconcile_chroma()
    assert res == {'upserted': 3, 'deleted': 0}

    # In sync → nothing re-embedded
    fake.upserted.clear()
    assert sm.reconcile_chroma() == {'upserted': 0, 'deleted': 0}

    # Stale value (failed update) + leftover id (failed delete)
    fake.items[_chroma_id('Артур', 'любит')]['value'] = 'кофе'
    fake.items['deadbeef'] = {'subject': 'X'}
    assert sm.reconcile_chroma() == {'upserted': 1, 'deleted': 1}
    assert fake.items[_chroma_id('Артур', 'любит')]['value'] == 'чай'
    assert 'deadbeef' not in fake.items


# ─────────────────────────────────────────────────────────────────────────────
# SQLite settings
# ─────────────────────────────────────────────────────────────────────────────

def test_connect_sets_wal_and_busy_timeout(tmp_path):
    conn = connect(str(tmp_path / 'x.db'))
    assert conn.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    assert conn.execute('PRAGMA busy_timeout').fetchone()[0] == 5000
    conn.close()


def test_session_commits_and_closes(tmp_path):
    path = str(tmp_path / 'x.db')
    with session(path) as c:
        c.execute('CREATE TABLE t (v)')
        c.execute('INSERT INTO t VALUES (1)')
    with pytest.raises(Exception):
        c.execute('SELECT 1')                      # closed
    with session(path) as c:
        assert c.execute('SELECT v FROM t').fetchone()[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Telegram reminders: delivered only on the bridge's ACK

class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(json.loads(m.data))


def _reminder_node(node, tmp_path):
    from inmoov_memory.reminder_db import ReminderDB
    node._reminder_db = ReminderDB(str(tmp_path / 'rem.db'))
    node._tg_push_pub = _Pub()
    node._tg_reminder_person_id = 5
    rid = node._reminder_db.add_reminder(5, 'Артур', 'полить цветы', trigger_date='2000-01-01')
    return rid


def _ack(node, push_id, ok, error=''):
    from std_msgs.msg import String
    node._tg_push_ack_cb(String(data=json.dumps({'id': push_id, 'ok': ok, 'error': error})))


def _delivered(node, rid):
    return any(r['id'] == rid and r['delivered'] for r in node._reminder_db.list_reminders(5))


def test_reminder_marked_delivered_only_after_ack(node, tmp_path):
    rid = _reminder_node(node, tmp_path)
    node._send_due_reminders_to_telegram()
    assert node._tg_push_pub.msgs[0]['id'] == f'reminder:{rid}'
    assert not _delivered(node, rid)

    node._send_due_reminders_to_telegram()            # ACK pending → no duplicate
    assert len(node._tg_push_pub.msgs) == 1

    _ack(node, f'reminder:{rid}', True)
    assert _delivered(node, rid)
    node._send_due_reminders_to_telegram()
    assert len(node._tg_push_pub.msgs) == 1           # delivered → not due any more


def test_reminder_retried_after_failed_or_missing_ack(node, tmp_path):
    rid = _reminder_node(node, tmp_path)
    node._send_due_reminders_to_telegram()
    _ack(node, f'reminder:{rid}', False, 'network down')
    assert not _delivered(node, rid)
    node._send_due_reminders_to_telegram()            # failed → sent again right away
    assert len(node._tg_push_pub.msgs) == 2

    node._tg_inflight[f'reminder:{rid}'] -= node._TG_ACK_TIMEOUT_SEC + 1   # no ACK for too long
    node._send_due_reminders_to_telegram()
    assert len(node._tg_push_pub.msgs) == 3


def test_foreign_ack_ignored(node, tmp_path):
    rid = _reminder_node(node, tmp_path)
    _ack(node, f'reminder:{rid}', True)               # never sent by us
    _ack(node, 'openhab:42', True)
    assert not _delivered(node, rid)
