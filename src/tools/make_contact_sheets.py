#!/usr/bin/env python3
"""Контактные листы для визуальной проверки авторазметки.

Собирает сетки кадров с нарисованными боксами: bag — зелёный, person — синий.
Нужно, чтобы глазами найти систематические ошибки учителя (памятники,
манекены, отражения) до того, как они попадут в обучение.

Запуск (из папки src/):
    python tools/make_contact_sheets.py
    python tools/make_contact_sheets.py --split holdout --n 40 --per-sheet 20
    python tools/make_contact_sheets.py --split train --only-person

Выход: runs/12k_qc/sheet_<split>_<i>.jpg
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

COLOR_BAG = (0, 200, 0)
COLOR_PERSON = (255, 140, 0)


def parse_args():
    p = argparse.ArgumentParser(description="Контактные листы с боксами для QC разметки.")
    p.add_argument("--root", default=str(cfg.BASE_DIR / "dataset" / "12k_merged"))
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=100, help="сколько картинок всего")
    p.add_argument("--per-sheet", type=int, default=20, help="картинок на лист (4x5)")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--tile", type=int, default=480, help="ширина плитки, px")
    p.add_argument("--out", default=str(cfg.BASE_DIR / "runs" / "12k_qc"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only-person", action="store_true", help="только кадры с person-боксами")
    p.add_argument("--only-empty", action="store_true", help="только кадры без боксов")
    return p.parse_args()


def read_boxes(path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(errors="ignore").splitlines():
        s = line.split()
        if len(s) != 5:
            continue
        try:
            boxes.append((int(s[0]), *(float(v) for v in s[1:])))
        except ValueError:
            continue
    return boxes


def draw(img, boxes, tile_w):
    h, w = img.shape[:2]
    for cls, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = COLOR_BAG if cls == 0 else COLOR_PERSON
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, w // 600))
    scale = tile_w / w
    return cv2.resize(img, (tile_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    root = Path(args.root)
    split = args.split
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    if not img_dir.is_dir():
        print(f"[ERROR] no images: {img_dir}")
        return 1

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    records = []
    for p in images:
        boxes = read_boxes(lbl_dir / f"{p.stem}.txt")
        if args.only_person and not any(b[0] == 1 for b in boxes):
            continue
        if args.only_empty and boxes:
            continue
        records.append((p, boxes))

    if not records:
        print("[ERROR] no images matched the filters")
        return 1

    random.Random(args.seed).shuffle(records)
    records = records[:args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = args.cols
    rows = (args.per_sheet + cols - 1) // cols
    tile_w = args.tile
    tile_h = int(tile_w * 9 / 16)
    n_sheets = 0
    for start in range(0, len(records), args.per_sheet):
        chunk = records[start:start + args.per_sheet]
        sheet = np.full((rows * tile_h, cols * tile_w, 3), 30, np.uint8)
        for i, (p, boxes) in enumerate(chunk):
            img = cv2.imread(str(p))
            if img is None:
                continue
            t = draw(img, boxes, tile_w)
            t = cv2.resize(t, (tile_w, tile_h))
            r, c = divmod(i, cols)
            sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = t
        n_sheets += 1
        path = out / f"sheet_{split}_{n_sheets:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"[{split}] sheet {n_sheets}: {len(chunk)} кадров -> {path}")

    print(f"[DONE] {len(records)} картинок, {n_sheets} листов в {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
