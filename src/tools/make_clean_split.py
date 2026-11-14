#!/usr/bin/env python3
"""Собрать честный train: dataset/all без кадров из cctv/valid+test.

Зачем: dataset/all собран из dataset/{train,valid,test}, а исходный экспорт
dataset/cctv содержит те же кадры. Его valid+test — единственный набор, где
сплиты не пересекаются по исходным кадрам (0 общих префиксов). Но 155 из 168
этих кадров лежат в dataset/all, поэтому любая модель, обученная на all,
holdout уже видела.

Скрипт копирует all, выбрасывая все копии кадров, чьи префиксы есть в
cctv/valid или cctv/test, и пишет data_all_clean.yaml с val/test на cctv.

Запуск (из папки src/):
    python tools/make_clean_split.py
    python tools/make_clean_split.py --force   # перезаписать all_clean
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg


def prefix(name):
    """crossing_mp4-0009_jpg.rf.90e7...jpg -> crossing_mp4-0009."""
    m = re.match(r"(.+?)_jpg\.rf\.", name)
    return m.group(1) if m else re.sub(r"\.(jpg|png)$", "", name)


def images(root):
    root = Path(root)
    return sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))


def parse_args():
    p = argparse.ArgumentParser(description="Train без кадров честного holdout.")
    p.add_argument("--all", default=str(cfg.BASE_DIR / "dataset" / "all"),
                   help="исходный датасет (images/ + labels/)")
    p.add_argument("--cctv", default=str(cfg.BASE_DIR / "dataset" / "cctv"),
                   help="исходный экспорт с чистыми сплитами")
    p.add_argument("--out", default=str(cfg.BASE_DIR / "dataset" / "all_clean"),
                   help="куда собрать очищенный train")
    p.add_argument("--yaml", default=str(cfg.BASE_DIR / "dataset" / "data_all_clean.yaml"),
                   help="куда записать data.yaml")
    p.add_argument("--force", action="store_true", help="перезаписать --out")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.all)
    cctv = Path(args.cctv)
    out = Path(args.out)

    holdout = {prefix(f.name)
               for split in ("valid", "test")
               for f in images(cctv / split / "images")}
    if not holdout:
        print(f"[ERR] в {cctv}/valid|test нет картинок")
        return 1

    if out.exists():
        if not args.force:
            print(f"[ERR] {out} уже существует, добавь --force")
            return 1
        shutil.rmtree(out)

    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)

    kept = skipped = missing_labels = 0
    kept_frames, skipped_frames = set(), set()
    for img in images(src / "images"):
        p = prefix(img.name)
        if p in holdout:
            skipped += 1
            skipped_frames.add(p)
            continue
        label = src / "labels" / (img.stem + ".txt")
        if not label.exists():
            missing_labels += 1
            continue
        shutil.copy2(img, out / "images" / img.name)
        shutil.copy2(label, out / "labels" / label.name)
        kept += 1
        kept_frames.add(p)

    yaml = Path(args.yaml)
    yaml.write_text(
        "train: all_clean/images\n"
        "val: cctv_holdout/valid/images\n"
        "test: cctv_holdout/test/images\n"
        "\n"
        "nc: 2\n"
        "names: ['bag', 'person']\n",
        encoding="utf-8",
    )

    print(f"holdout frames (cctv valid+test): {len(holdout)}")
    print(f"kept   images: {kept:5}  frames: {len(kept_frames):4}")
    print(f"skipped images: {skipped:5}  frames: {len(skipped_frames):4}")
    print(f"missing labels: {missing_labels}")
    print(f"overlap kept & holdout frames: {len(kept_frames & holdout)}")
    print(f"[OK] {out}")
    print(f"[OK] {yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
