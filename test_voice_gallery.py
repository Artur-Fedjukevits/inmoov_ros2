#!/usr/bin/env python3
"""
test_voice_gallery.py
======================
Standalone diagnostic for the ECAPA-TDNN voice gallery.

Records a voice sample -> ECAPA embedding -> compares against the gallery in
the DB. Does not need ROS2, only speechbrain + sounddevice.

Usage:
    python3 test_voice_gallery.py [record_seconds]
    python3 test_voice_gallery.py --gallery-only   # only cross-session similarities already in the DB
    python3 test_voice_gallery.py --rebuild --person-id <id> [--samples N] [--max-sec S]

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""
import sys
import time
import sqlite3
import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier

RATE      = 16000
DB_PATH   = '/home/artur/inmoov_memory.db'
EMB_DIM   = 192
SAVEDIR   = '/home/artur/.cache/speechbrain/spkrec-ecapa-voxceleb'
MIN_SAMP  = 8000  # 0.5s @ 16kHz

# ── Gallery from the DB ───────────────────────────────────────────────────────

def load_gallery():
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        'SELECT vg.person_id, p.name, vg.embedding, vg.recorded_at '
        'FROM voice_gallery vg JOIN persons p ON p.id=vg.person_id '
        'ORDER BY vg.person_id, vg.recorded_at'
    ).fetchall()
    gallery: dict[int, dict] = {}
    for pid, name, blob, ts in rows:
        emb = np.frombuffer(blob, dtype=np.float32).copy()
        if emb.shape[0] != EMB_DIM:
            print(f'  [SKIP] pid={pid} ({name}): wrong dim={emb.shape[0]}')
            continue
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            print(f'  [SKIP] pid={pid} ({name}): zero norm')
            continue
        emb /= norm
        gallery.setdefault(pid, {'name': name, 'embs': [], 'ts': []})
        gallery[pid]['embs'].append(emb)
        gallery[pid]['ts'].append(ts)
    db.close()
    return gallery

def print_gallery_stats(gallery: dict):
    print('\n=== Gallery in the DB ===')
    for pid, info in gallery.items():
        n = len(info['embs'])
        ts_strs = [f'{t:.0f}' for t in info['ts']]
        print(f'  {info["name"]} (pid={pid}): {n} entries, ts={ts_strs}')
        if n >= 2:
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    s = float(np.dot(info['embs'][i], info['embs'][j]))
                    sims.append((i, j, s))
            sim_vals = [s for _, _, s in sims]
            print(f'    inter-entry similarity: avg={np.mean(sim_vals):.3f}  '
                  f'min={min(sim_vals):.3f}  max={max(sim_vals):.3f}')
            for i, j, s in sims:
                print(f'    [{i}]<->[{j}] sim={s:.3f}')
        mean_emb = np.mean(info['embs'], axis=0)
        mean_emb /= np.linalg.norm(mean_emb) + 1e-8
        info['mean_emb'] = mean_emb

# ── Recording ──────────────────────────────────────────────────────────────────

def load_vad():
    """Loads Silero VAD (the same model used in voice_detector_node)."""
    print('  Loading Silero VAD...')
    vad, _ = torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    vad.eval()
    return vad


def record_vad(vad_model, max_sec: float = 10.0, silence_sec: float = 1.5,
               vad_threshold: float = 0.5, countdown: int = 3) -> np.ndarray:
    """Records with VAD: waits for voice, records until a pause or max_sec.

    Behavior mirrors voice_detector_node:
    - Countdown before it starts listening
    - Recording only begins once voice is detected
    - Ends on silence > silence_sec or on the max_sec timer
    """
    try:
        import sounddevice as sd
    except ImportError:
        print('[ERROR] pip install sounddevice')
        sys.exit(1)

    CHUNK = 512  # same as voice_detector

    for i in range(countdown, 0, -1):
        print(f'  {i}...', end=' ', flush=True)
        time.sleep(1.0)
    print('Speak now!')

    chunks_collected: list[np.ndarray] = []
    silence_chunks   = 0
    voice_started    = False
    silence_thresh   = int(silence_sec * RATE / CHUNK)
    max_chunks       = int(max_sec * RATE / CHUNK)
    total_chunks     = 0

    with sd.InputStream(samplerate=RATE, channels=1, dtype='float32',
                        blocksize=CHUNK) as stream:
        while True:
            chunk, _ = stream.read(CHUNK)
            chunk = chunk.flatten()
            total_chunks += 1

            conf = float(vad_model(torch.from_numpy(chunk), RATE).item())

            if conf > vad_threshold:
                voice_started  = True
                silence_chunks = 0
                chunks_collected.append(chunk)
            elif voice_started:
                chunks_collected.append(chunk)
                silence_chunks += 1
                if silence_chunks >= silence_thresh:
                    print('  [silence — recording finished]')
                    break
            elif total_chunks >= max_chunks:
                print('  [timed out waiting for voice]')
                break

            if voice_started and len(chunks_collected) >= max_chunks:
                print('  [reached length limit]')
                break

    if not chunks_collected or not voice_started:
        print('  [WARN] VAD did not detect any voice')
        return np.zeros(CHUNK, dtype=np.float32)

    audio = np.concatenate(chunks_collected)
    dur   = len(audio) / RATE
    rms   = float(np.sqrt(np.mean(audio ** 2)))
    peak  = float(np.max(np.abs(audio)))
    print(f'  Recorded: {dur:.2f}s, RMS={rms:.4f}, peak={peak:.4f}')
    if rms < 0.002:
        print('  [WARN] Very quiet! Check the microphone.')
    return audio

# ── ECAPA-TDNN ─────────────────────────────────────────────────────────────────

def load_model():
    print('\nLoading ECAPA-TDNN...')
    model = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir=SAVEDIR,
        run_opts={'device': 'cpu'},
    )
    print('Model loaded.')
    return model

def embed(model, audio: np.ndarray, verbose: bool = True) -> np.ndarray | None:
    if len(audio) < MIN_SAMP:
        print(f'[ERROR] Audio too short: {len(audio)/RATE:.2f}s < 0.5s')
        return None
    # Same RMS normalization as voice_detector_node._sv_embed
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > 1e-6:
        audio = np.clip(audio * (0.05 / rms), -1.0, 1.0)
    if verbose:
        print(f'  RMS after normalization: {float(np.sqrt(np.mean(audio**2))):.4f}')
    wav = torch.tensor(audio).unsqueeze(0)           # [1, N]
    wav_lens = torch.tensor([1.0])
    with torch.no_grad():
        raw = model.encode_batch(wav, wav_lens)       # [1, 1, 192] or [1, 192]
    if verbose:
        print(f'  encode_batch output shape: {raw.shape}')
    emb = raw.squeeze().numpy().astype(np.float32)
    if verbose:
        print(f'  After squeeze shape: {emb.shape}, norm before normalization: {np.linalg.norm(emb):.4f}')
    if emb.ndim != 1 or emb.shape[0] != EMB_DIM:
        print(f'[ERROR] Unexpected embedding shape: {emb.shape}')
        return None
    emb /= np.linalg.norm(emb) + 1e-8
    return emb

# ── Comparison ─────────────────────────────────────────────────────────────────

def lookup(query: np.ndarray, gallery: dict, high: float = 0.62, uncertain: float = 0.50):
    """Uses the centroid — same as _lookup_by_voice in memory_node."""
    print('\n=== Comparing against the gallery (centroid, as in memory_node) ===')
    best_pid, best_sim, best_name = None, -1.0, '?'
    for pid, info in gallery.items():
        sims = [float(np.dot(e, query)) for e in info['embs']]
        centroid_sim = float(np.dot(info['mean_emb'], query))
        print(f'  {info["name"]} (pid={pid}):')
        print(f'    individual: {[f"{s:.3f}" for s in sims]}')
        print(f'    centroid sim:   {centroid_sim:.3f}')
        if centroid_sim > best_sim:
            best_sim, best_pid, best_name = centroid_sim, pid, info['name']

    print(f'\n  Best match: {best_name} (pid={best_pid}), sim={best_sim:.3f}')
    if best_sim >= high:
        print(f'  => HIGH confidence (>= {high})')
    elif best_sim >= uncertain:
        print(f'  => UNCERTAIN (>= {uncertain})')
    else:
        print(f'  => UNKNOWN (< {uncertain})')

# ── Gallery sanity check (verify the same model gives the same embeddings) ────

def sanity_cross_session(model, vad_model, gallery: dict):
    """Records two samples with VAD and compares them to each other and to the gallery."""
    embs = []
    for i in range(2):
        if i == 1:
            print('\nPausing 2s...')
            time.sleep(2.0)
        audio = record_vad(vad_model, max_sec=10.0, countdown=3)
        emb = embed(model, audio, verbose=False)
        if emb is not None:
            embs.append(emb)

    if len(embs) == 2:
        s = float(np.dot(embs[0], embs[1]))
        print(f'\n  Similarity between the two fresh recordings: {s:.3f}')
        print('  (should be > 0.70 for the same person)')

    for emb in embs:
        lookup(emb, gallery)

# ── Gallery rebuild ─────────────────────────────────────────────────────────────

def rebuild_voice_gallery(model, vad_model, person_id: int, n_samples: int = 5,
                          max_sec: float = 10.0):
    """Records n_samples samples with VAD and saves them to the DB, replacing the old gallery.

    Recording each sample:
    - 3s countdown
    - Waits for voice (VAD)
    - Ends automatically after 1.5s of silence
    - Outlier samples are filtered out by centroid similarity at the end
    """

    db = sqlite3.connect(DB_PATH)
    name_row = db.execute('SELECT name FROM persons WHERE id=?', (person_id,)).fetchone()
    if not name_row:
        print(f'[ERROR] person_id={person_id} not found in the DB')
        db.close()
        return

    name = name_row[0]
    print(f'\nRebuilding the voice gallery for {name} (pid={person_id})')
    print(f'Will record {n_samples} samples (max {max_sec:.0f}s each).')
    print('Speak a few words — recording stops on its own after a pause.\n')

    embs = []
    i = 0
    attempts = 0
    max_attempts = n_samples * 3

    while len(embs) < n_samples and attempts < max_attempts:
        attempts += 1
        print(f'Sample {len(embs)+1}/{n_samples}:')
        audio = record_vad(vad_model, max_sec=max_sec, countdown=3)
        emb = embed(model, audio, verbose=False)
        if emb is None:
            print('  [SKIP] audio too short (< 0.5s)')
            continue
        print(f'  Accepted (dim={emb.shape[0]}, norm=1.0)')
        embs.append(emb)
        if len(embs) < n_samples:
            print('  Pausing 2s...')
            time.sleep(2.0)

    if not embs:
        print('[ERROR] No sample was accepted')
        db.close()
        return

    # Mutual similarity of the new samples — diagnostic only
    if len(embs) >= 2:
        sims_all = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims_all.append(float(np.dot(embs[i], embs[j])))
        print(f'\nMutual similarity across all {len(embs)} samples: '
              f'avg={np.mean(sims_all):.3f}  min={min(sims_all):.3f}  max={max(sims_all):.3f}')

        # Outlier filtering: drop samples with low average similarity to the rest.
        # Threshold 0.30: a sample with avg_sim < 0.30 to all others is noise or someone else's voice.
        OUTLIER_THRESH = 0.30
        good_embs = []
        for k, emb in enumerate(embs):
            others = [embs[j] for j in range(len(embs)) if j != k]
            avg_sim = float(np.mean([np.dot(emb, o) for o in others]))
            if avg_sim >= OUTLIER_THRESH:
                good_embs.append(emb)
            else:
                print(f'  [OUTLIER] Sample {k+1} dropped (avg_sim={avg_sim:.3f} < {OUTLIER_THRESH})')

        if len(good_embs) < len(embs):
            print(f'  After filtering: {len(good_embs)}/{len(embs)} samples')
        embs = good_embs

    if not embs:
        print('[ERROR] All samples were dropped as outliers — re-record in a quiet place')
        db.close()
        return

    # Centroid, to check the expected similarity at lookup time
    centroid = np.mean(np.stack(embs), axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-8
    centroid_sims = [float(np.dot(centroid, e)) for e in embs]
    print(f'  Sample-to-centroid similarity: avg={np.mean(centroid_sims):.3f}  '
          f'min={min(centroid_sims):.3f}  max={max(centroid_sims):.3f}')
    print(f'  (lookup will compare against the centroid — expect sim ~ {np.mean(centroid_sims):.2f})')

    # Drop the old gallery and insert the new one
    db.execute('DELETE FROM voice_gallery WHERE person_id=?', (person_id,))
    now = time.time()
    for emb in embs:
        db.execute(
            'INSERT INTO voice_gallery (person_id, embedding, recorded_at) VALUES (?,?,?)',
            (person_id, emb.tobytes(), now))
    db.commit()
    db.close()
    print(f'\nSaved {len(embs)} new entries for {name} to the DB.')
    print('Restart memory_node so the gallery reloads.')


# ── main ────────────────────────────────────────────────────────────────────────

def _get_flag(flag: str, default: str) -> str:
    """Returns the value following the flag, or default."""
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return default


def main():
    gallery_only = '--gallery-only' in sys.argv
    rebuild      = '--rebuild'      in sys.argv

    n_arg   = _get_flag('--samples',   '6')
    pid_arg = _get_flag('--person-id', '5')
    max_arg = _get_flag('--max-sec',   '10')

    gallery = load_gallery()
    print_gallery_stats(gallery)

    if gallery_only:
        return

    model     = load_model()
    print()
    vad_model = load_vad()

    if rebuild:
        rebuild_voice_gallery(model, vad_model, int(pid_arg), int(n_arg), float(max_arg))
        return

    print('\nMode: single recording (VAD)')
    audio = record_vad(vad_model, max_sec=float(max_arg), countdown=3)
    query = embed(model, audio, verbose=False)
    if query is None:
        return
    lookup(query, gallery)

    print('\n--- Bonus: two samples back-to-back ---')
    sanity_cross_session(model, vad_model, gallery)


if __name__ == '__main__':
    main()
