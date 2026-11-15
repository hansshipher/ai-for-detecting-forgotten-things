#!/usr/bin/env python3
"""Честный holdout из dataset/cctv: полигоны переводятся в bbox."""

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg


def prefix(name):
    m = re.match(r"(.+?)_jpg\.rf\.", name)
    return m.group(1) if m else re.sub(r"\.(jpg|png)$", "", name)


def images(root):
    root = Path(root)
    return sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))


def read_rows(path):
    return [line.split() for line in path.read_text(errors="ignore").splitlines() if line.split()]


def to_boxes(rows):
    """Строки YOLO -> только detection-строки (polygon -> bbox)."""
    out, converted, bad = [], 0, 0
    for r in rows:
        if len(r) == 5:
            out.append(r)
            continue
        coords = r[1:]
        if len(coords) < 4 or len(coords) % 2:
            bad += 1
            continue
        xs = [float(v) for v in coords[0::2]]
        ys = [float(v) for v in coords[1::2]]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        if x2 <= x1 or y2 <= y1:
            bad += 1
            continue
        out.append([r[0], f"{(x1 + x2) / 2:.6f}", f"{(y1 + y2) / 2:.6f}",
                    f"{x2 - x1:.6f}", f"{y2 - y1:.6f}"])
        converted += 1
    return out, converted, bad


def key(row):
    return (row[0], *(round(float(v), 3) for v in row[1:5]))


def rows_match(a, b, tol=0.01):
    """Совпадают ли наборы боксов с допуском (у экспортов разное округление)."""
    if len(a) != len(b):
        return False
    used = [False] * len(b)
    for ra in a:
        best, bi = None, -1
        for i, rb in enumerate(b):
            if used[i] or ra[0] != rb[0]:
                continue
            d = max(abs(float(ra[j]) - float(rb[j])) for j in range(1, 5))
            if best is None or d < best:
                best, bi = d, i
        if bi < 0 or best > tol:
            return False
        used[bi] = True
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Holdout из cctv без polygon-строк.")
    p.add_argument("--cctv", default=str(cfg.BASE_DIR / "dataset" / "cctv"))
    p.add_argument("--out", default=str(cfg.BASE_DIR / "dataset" / "cctv_holdout"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cctv, out = Path(args.cctv), Path(args.out)
    if out.exists():
        if not args.force:
            print(f"[ERR] {out} уже существует, добавь --force")
            return 1
        shutil.rmtree(out)

    total_bad = 0
    for split in ("valid", "test"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        converted = rows_total = files = bad_rows = 0
        mismatch = compared = 0
        # rf-хэш в имени у экспортов разный, поэтому сверяем по префиксу кадра
        ref_by_prefix = {}
        ref_dir = cfg.BASE_DIR / "dataset" / split / "labels"
        if ref_dir.exists():
            for p in ref_dir.glob("*.txt"):
                ref_by_prefix.setdefault(prefix(p.name), p)
        for img in images(cctv / split / "images"):
            label = cctv / split / "labels" / (img.stem + ".txt")
            if not label.exists():
                print(f"[WARN] нет метки: {label.name}")
                continue
            rows, n_conv, n_bad = to_boxes(read_rows(label))
            total_bad += n_bad
            bad_rows += n_bad
            converted += n_conv
            rows_total += len(rows)
            files += 1
            shutil.copy2(img, out / split / "images" / img.name)
            (out / split / "labels" / label.name).write_text(
                "\n".join(" ".join(r) for r in rows) + "\n", encoding="utf-8")

            ref = ref_by_prefix.get(prefix(img.name))
            if ref is not None:
                ref_rows = to_boxes(read_rows(ref))[0]
                compared += 1
                if not rows_match(rows, ref_rows):
                    mismatch += 1

        print(f"{split:5} files={files:4} rows={rows_total:5} polygon->box={converted:4} "
              f"bad_rows={bad_rows:2} | сверка с dataset/{split}: {compared} файлов, "
              f"расхождений {mismatch}")

    print(f"[OK] {out}")
    if total_bad:
        print(f"[WARN] отброшено битых polygon-строк: {total_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
