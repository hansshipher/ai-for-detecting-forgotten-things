#!/usr/bin/env python3
"""Сборка dataset/12k_merged: bag из ручной разметки, person от COCO-учителя."""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def parse_args():
    p = argparse.ArgumentParser(
        description="Сборка 12k_merged: bag + person(учитель), сплиты по dHash-кластерам.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", default=str(cfg.BASE_DIR / "dataset" / "12k-dataset"))
    p.add_argument("--clusters", default=str(cfg.BASE_DIR / "runs" / "12k_audit" / "clusters.csv"))
    p.add_argument("--person", default=str(cfg.BASE_DIR / "runs" / "12k_person_labels"),
                   help="person-лейблы учителя (<split>/<stem>.txt)")
    p.add_argument("--out", default=str(cfg.BASE_DIR / "dataset" / "12k_merged"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit", type=int, default=0, help="максимум картинок на сплит (для проверки)")
    p.add_argument("--dry-run", action="store_true", help="только посчитать и напечатать план")
    return p.parse_args()


def read_labels(path):
    """Строгий разбор YOLO-лейбла: [(cls, cx, cy, w, h)], отброшенные строки.

    Поддерживает и боксы ("cls cx cy w h"), и полигоны ("cls x1 y1 x2 y2 ..."):
    полигон сводится к охватывающему боксу, чтобы не терять разметку.
    """
    boxes, bad = [], 0
    if not path.exists():
        return boxes, bad
    for line in path.read_text(errors="ignore").splitlines():
        s = line.split()
        if not s:
            continue
        try:
            cls = int(s[0])
            vals = [float(v) for v in s[1:]]
        except ValueError:
            bad += 1
            continue

        if len(vals) == 4:                      # bbox
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:   # полигон
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            bad += 1
            continue

        if cls not in (0, 1) or w <= 0 or h <= 0:
            bad += 1
            continue
        # мелкий выход за границу кадра — зажимаем, крупный — брак
        if not (-0.02 <= cx <= 1.02 and -0.02 <= cy <= 1.02):
            bad += 1
            continue
        boxes.append((cls, min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
                      min(w, 1.0), min(h, 1.0)))
    return boxes, bad


def fmt(boxes):
    return "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes)


def link_or_copy(src, dst):
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    person_dir = Path(args.person)

    # --- кластеры ---
    rows = list(csv.DictReader(open(args.clusters, encoding="utf-8")))
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["cluster_id"]].append(r["path"].replace("/", os.sep))

    # --- ручная разметка людей: какие кластеры уходят в holdout ---
    manual = set()
    for split in SPLITS:
        lbl_dir = root / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for f in lbl_dir.glob("*.txt"):
            boxes, _ = read_labels(f)
            if any(c == 1 for c, *_ in boxes):
                manual.add(f"{split}/images/{f.stem}.jpg")

    holdout_clusters = set()
    for cid, paths in clusters.items():
        if any(p.replace(os.sep, "/") in manual for p in paths):
            holdout_clusters.add(cid)

    holdout_paths = [p for cid in holdout_clusters for p in clusters[cid]]
    rest = {cid: paths for cid, paths in clusters.items() if cid not in holdout_clusters}
    n_rest = sum(len(v) for v in rest.values())

    # --- назначение кластеров по сплитам ---
    order = sorted(rest.items())
    random.Random(args.seed).shuffle(order)
    targets = {"train": 1.0 - args.val_frac - args.test_frac,
               "valid": args.val_frac, "test": args.test_frac}
    target_n = {k: v * n_rest for k, v in targets.items()}
    assigned = {k: [] for k in targets}
    have = {k: 0 for k in targets}
    for cid, paths in order:
        k = max(targets, key=lambda s: target_n[s] - have[s])
        assigned[k].append((cid, paths))
        have[k] += len(paths)

    plan = {k: sum(len(p) for _, p in v) for k, v in assigned.items()}
    print(f"[PLAN] root: {root}")
    print(f"[PLAN] clusters: {len(clusters)} | holdout clusters: {len(holdout_clusters)}")
    print(f"[PLAN] holdout: {len(holdout_paths)} картинок (из них с ручной разметкой: {len(manual)})")
    print(f"[PLAN] train={plan['train']} valid={plan['valid']} test={plan['test']} "
          f"(всего {n_rest})")
    if args.dry_run:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    stats = {"root": str(root), "out": str(out), "seed": args.seed,
             "clusters": len(clusters), "holdout_clusters": len(holdout_clusters),
             "splits": {}}
    totals = defaultdict(int)

    # holdout: ручные person-лейблы как GT, bag — как есть
    jobs = [("holdout", [(cid, p) for cid in holdout_clusters for p in clusters[cid]])]
    jobs += [(k, [(cid, p) for cid, paths in assigned[k] for p in paths]) for k in ("train", "valid", "test")]

    for split, items in jobs:
        if args.limit:
            items = items[:args.limit]
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_img = n_bag = n_person = n_bad = n_empty = n_missing_person = 0
        for _, rel in items:
            src_img = root / rel
            if not src_img.exists():
                continue
            stem = src_img.stem
            src_split = Path(rel).parts[0]
            link_or_copy(src_img, img_dir / src_img.name)

            src_boxes, bad = read_labels(root / src_split / "labels" / f"{stem}.txt")
            n_bad += bad
            bag = [b for b in src_boxes if b[0] == 0]

            if split == "holdout":
                person = [b for b in src_boxes if b[0] == 1]
            else:
                pfile = person_dir / src_split / f"{stem}.txt"
                if not pfile.exists():
                    n_missing_person += 1
                    person = []
                else:
                    person, bad_p = read_labels(pfile)
                    n_bad += bad_p
                    person = [(1, cx, cy, w, h) for c, cx, cy, w, h in person if c == 1]

            boxes = bag + person
            (lbl_dir / f"{stem}.txt").write_text(fmt(boxes), encoding="utf-8")
            n_img += 1
            n_bag += len(bag)
            n_person += len(person)
            if not boxes:
                n_empty += 1

        stats["splits"][split] = {"images": n_img, "bag": n_bag, "person": n_person,
                                  "empty": n_empty, "bad_lines": n_bad,
                                  "missing_teacher_label": n_missing_person}
        totals["images"] += n_img
        totals["bag"] += n_bag
        totals["person"] += n_person
        totals["bad_lines"] += n_bad
        print(f"[{split}] images={n_img} bag={n_bag} person={n_person} "
              f"empty={n_empty} bad_lines={n_bad} no_teacher={n_missing_person}", flush=True)

    (out / "data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "holdout: holdout/images\n"
        "\n"
        "nc: 2\n"
        "names: ['bag', 'person']\n",
        encoding="utf-8")

    stats["totals"] = dict(totals)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    print(f"\n[DONE] images={totals['images']} bag={totals['bag']} person={totals['person']} "
          f"bad_lines={totals['bad_lines']}")
    print(f"[INFO] dataset: {out}")
    print(f"[INFO] stats:   {out / 'stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
