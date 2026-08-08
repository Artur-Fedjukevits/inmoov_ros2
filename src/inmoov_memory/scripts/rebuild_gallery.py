#!/usr/bin/env python3
"""
rebuild_gallery.py
==================
Пересчитывает embeddings в БД из фотогалереи.
Запускать после ручной обработки фотографий:
  - удалил плохие/размытые снимки → запусти rebuild
  - переместил фото из одного человека в другой → запусти rebuild
  - добавил фото вручную → запусти rebuild

Использование:
  python3 rebuild_gallery.py [--db /path/to/inmoov_memory.db] [--gallery /path/to/inmoov_faces]

После rebuild нужно уведомить запущенный memory_node:
  ros2 service call /memory/query inmoov_msgs/srv/MemoryQuery \
    "request_json: '{op: reload_gallery}'"
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
    from insightface.app import FaceAnalysis
except ImportError as e:
    print(f'Ошибка импорта: {e}')
    print('Установи: pip install insightface opencv-python')
    sys.exit(1)


def load_insightface():
    print('Загружаю InsightFace buffalo_l...')
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print('InsightFace готов.')
    return app


def get_embedding(app, image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        # Попробуем с уменьшенным порогом det_thresh
        return None
    # Берём лицо с наибольшей площадью bbox
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb = best.normed_embedding.astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


def rebuild(db_path: str, gallery_dir: str):
    gallery_root = Path(gallery_dir) / 'persons'
    if not gallery_root.exists():
        print(f'Директория не найдена: {gallery_root}')
        sys.exit(1)

    db = sqlite3.connect(db_path)
    app = load_insightface()

    # Очищаем текущую галерею
    db.execute('DELETE FROM person_gallery')
    db.commit()
    print('Галерея очищена. Начинаю пересчёт...\n')

    total_added = 0
    total_skipped = 0

    for person_dir in sorted(gallery_root.iterdir()):
        if not person_dir.is_dir():
            continue

        # Имя директории: {id}_{name}
        dir_name = person_dir.name
        parts = dir_name.split('_', 1)
        if len(parts) < 2 or not parts[0].isdigit():
            print(f'  Пропускаю (неверный формат имени): {dir_name}')
            continue

        person_id   = int(parts[0])
        person_name = parts[1].replace('_', ' ')

        # Проверяем существование в БД
        row = db.execute('SELECT name FROM persons WHERE id=?', (person_id,)).fetchone()
        if not row:
            print(f'  Человек id={person_id} не найден в БД — пропускаю {dir_name}')
            continue

        photos = sorted(list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')))
        print(f'  {dir_name}: {len(photos)} фото', end='', flush=True)

        added = 0
        skipped = 0
        now = datetime.now().isoformat()

        for photo_path in photos:
            emb = get_embedding(app, str(photo_path))
            if emb is None:
                skipped += 1
                print('.', end='', flush=True)
                continue

            # Определяем источник по имени файла
            source = 'enroll' if 'enroll' in photo_path.name else (
                     'manual' if 'manual' in photo_path.name else 'auto')

            db.execute(
                'INSERT INTO person_gallery '
                '(person_id, photo_path, embedding, quality, source, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (person_id, str(photo_path), emb.tobytes(), 1.0, source, now),
            )
            added += 1
            print('+', end='', flush=True)

        db.commit()
        total_added   += added
        total_skipped += skipped
        print(f'  → {added} добавлено, {skipped} пропущено')

    db.close()
    print(f'\nГотово: {total_added} embedding добавлено, {total_skipped} фото пропущено (нет лица).')
    print('\nУведомите memory_node о перезагрузке:')



def main():
    parser = argparse.ArgumentParser(description='Rebuild face gallery embeddings from photos')
    parser.add_argument('--db',      default='/home/artur/inmoov_memory.db',
                        help='Path to SQLite database')
    parser.add_argument('--gallery', default='/home/artur/inmoov_faces',
                        help='Path to gallery root directory')
    args = parser.parse_args()

    print(f'БД:      {args.db}')
    print(f'Галерея: {args.gallery}')
    print()
    rebuild(args.db, args.gallery)


if __name__ == '__main__':
    main()
