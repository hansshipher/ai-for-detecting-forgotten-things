#!/usr/bin/env python3
"""Аудит датасета: артефакты разметки и следы аугментации.

Зачем: после чистого ре-экспорта из Roboflow (без augmentation) нужно
убедиться, что не вернулись известные дефекты:

  - полигональные строки (ultralytics выбрасывает такие изображения целиком);
  - боксы «на весь кадр» (след crop-аугментации, `0 0.5 0.5 1 1`);
  - точные дубли кадров (след ×3 копий Roboflow);
  - источники, которые мы исключили (`dc_luggage`);
  - координаты вне [0, 1], вырожденные боксы (w/h <= 0);
  - изображения без лейблов и лейблы без изображений.

Скрипт только читает. Ничего не пишет и не удаляет.

Использование (из папки src/):
    python check_dataset.py                      # train valid test
    python check_dataset.py --splits all         # собранный dataset/all
    python check_dataset.py --dir ../dataset/all # произвольная пара images/labels
    python check_dataset.py --no-md5             # без проверки дублей (быстрее)
"""

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config as cfg

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
BAG_CLASS = 0

# префикс имени файла -> источник (порядок важен: длинные раньше коротких)
SOURCE_PREFIXES = (
    ("dc_luggage", "dc_luggage"),
    ("abandonedbag_2", "abandonedbag_2"),
    ("abandonedbag_3", "abandonedbag_3"),
    ("abandonedbag", "abandonedbag"),
    ("supermarket", "supermarket"),
    ("crossing", "crossing"),
    ("youtube", "youtube"),
    ("tokio", "tokio"),
    ("Metro", "metro"),
)

AREA_BUCKETS = (0.10, 0.25, 0.50, 0.90)


@dataclass
class Stats:
    images: int = 0
    labels: int = 0
    bbox_lines: int = 0
    polygon_lines: int = 0
    malformed_lines: int = 0
    out_of_range_lines: int = 0
    degenerate_lines: int = 0
    empty_labels: int = 0
    img_without_label: list = field(default_factory=list)
    label_without_img: list = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)
    bag_areas: list = field(default_factory=list)
    per_source: dict = field(default_factory=lambda: defaultdict(list))
    source_counts: Counter = field(default_factory=Counter)
    duplicates: list = field(default_factory=list)


def parse_args():
    p = argparse.ArgumentParser(
        description="Аудит датасета YOLO: артефакты разметки и аугментации.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=str(cfg.BASE_DIR / "dataset"),
                   help="корень датасета (по умолчанию dataset/ проекта)")
    p.add_argument("--splits", nargs="+", default=list(SPLITS),
                   help="какие сплиты проверять: train valid test all (по умолчанию train valid test)")
    p.add_argument("--dir", default=None,
                   help="проверить одну папку с подпапками images/ и labels/")
    p.add_argument("--no-md5", action="store_true",
                   help="не искать точные дубли (быстрее на больших датасетах)")
    return p.parse_args()


def source_of(name: str) -> str:
    for prefix, label in SOURCE_PREFIXES:
        if name.startswith(prefix):
            return label
    return "other"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels(label_path: Path, stats: Stats) -> None:
    try:
        text = label_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        print(f"  [!] не читается {label_path.name}: {exc}")
        return

    rows = [line for line in text.splitlines() if line.strip()]
    if not rows:
        stats.empty_labels += 1
        return

    for line in rows:
        parts = line.split()
        if len(parts) < 5:
            stats.malformed_lines += 1
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            stats.malformed_lines += 1
            continue

        if len(parts) == 5:
            stats.bbox_lines += 1
            cx, cy, bw, bh = coords
            if bw <= 0 or bh <= 0:
                stats.degenerate_lines += 1
            if any(v < 0.0 or v > 1.0 for v in coords):
                stats.out_of_range_lines += 1
            if cls == BAG_CLASS:
                area = max(bw, 0.0) * max(bh, 0.0)
                stats.bag_areas.append(area)
                stats.per_source[source_of(label_path.stem)].append(area)
        else:
            # полигон: class x1 y1 x2 y2 ...
            stats.polygon_lines += 1
            if any(v < 0.0 or v > 1.0 for v in coords):
                stats.out_of_range_lines += 1
            if cls == BAG_CLASS:
                xs = coords[0::2]
                ys = coords[1::2]
                area = max(max(xs) - min(xs), 0.0) * max(max(ys) - min(ys), 0.0)
                stats.bag_areas.append(area)
                stats.per_source[source_of(label_path.stem)].append(area)

        stats.classes[cls] += 1


def check_pair(images: Path, labels: Path, stats: Stats, with_md5: bool) -> None:
    if not images.is_dir():
        print(f"  [!] нет папки {images}")
        return
    if not labels.is_dir():
        print(f"  [!] нет папки {labels}")

    img_files = [p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    stats.images = len(img_files)

    for img in img_files:
        stats.source_counts[source_of(img.name)] += 1
        if not (labels / f"{img.stem}.txt").exists():
            stats.img_without_label.append(img.name)

    if labels.is_dir():
        for lbl in labels.glob("*.txt"):
            if lbl.name.endswith(".cache"):
                continue
            stats.labels += 1
            if not any((images / f"{lbl.stem}{ext}").exists() for ext in IMG_EXTS):
                stats.label_without_img.append(lbl.name)
            read_labels(lbl, stats)

    if with_md5:
        by_hash = defaultdict(list)
        for img in img_files:
            try:
                by_hash[md5_of(img)].append(img.name)
            except OSError:
                continue
        stats.duplicates = [names for names in by_hash.values() if len(names) > 1]


def report(name: str, stats: Stats) -> None:
    print(f"\n=== {name} ===")
    print(f"  изображений: {stats.images}   лейблов: {stats.labels}   "
          f"пустых лейблов: {stats.empty_labels}")
    print(f"  строк bbox: {stats.bbox_lines}   полигонов: {stats.polygon_lines}   "
          f"битых: {stats.malformed_lines}")
    print(f"  вне [0,1]: {stats.out_of_range_lines}   вырожденных (w/h<=0): "
          f"{stats.degenerate_lines}")

    if stats.classes:
        cls_str = ", ".join(f"{c}: {n}" for c, n in sorted(stats.classes.items()))
        print(f"  классы: {cls_str}")

    if stats.img_without_label:
        print(f"  [!] изображений без лейблов: {len(stats.img_without_label)}")
        for n in stats.img_without_label[:5]:
            print(f"      {n}")
        if len(stats.img_without_label) > 5:
            print(f"      ... ещё {len(stats.img_without_label) - 5}")
    if stats.label_without_img:
        print(f"  [!] лейблов без изображений: {len(stats.label_without_img)}")
        for n in stats.label_without_img[:5]:
            print(f"      {n}")

    areas = np.array(stats.bag_areas) if stats.bag_areas else np.zeros(0)
    if areas.size:
        print(f"\n  площадь bag: медиана {np.median(areas) * 100:.2f}%   "
              f"максимум {areas.max() * 100:.1f}%   всего {areas.size}")
        for th in AREA_BUCKETS:
            cnt = int((areas > th).sum())
            flag = "  <-- ПОДОЗРИТЕЛЬНО (crop-аугментация?)" if th >= 0.25 and cnt else ""
            print(f"    > {th * 100:>3.0f}% кадра: {cnt:>5}  "
                  f"({cnt / areas.size * 100:5.2f}%){flag}")
        print("\n  по источникам:")
        print(f"    {'источник':<15}{'кадров':>8}{'bag':>7}{'медиана':>10}{'max':>8}{'>25%':>7}")
        for src, vals in sorted(stats.per_source.items(), key=lambda kv: -len(kv[1])):
            v = np.array(vals)
            print(f"    {src:<15}{stats.source_counts.get(src, 0):>8}{len(v):>7}"
                  f"{np.median(v) * 100:>9.2f}%{v.max() * 100:>7.1f}%"
                  f"{int((v > 0.25).sum()):>7}")
        if stats.source_counts.get("dc_luggage"):
            print("  [!] dc_luggage всё ещё в датасете — его нужно исключить")
    else:
        print("  [i] боксов класса bag (0) не найдено")

    if stats.duplicates:
        total = sum(len(g) for g in stats.duplicates)
        print(f"\n  [!] точных дублей: {len(stats.duplicates)} групп, {total} файлов")
        for group in stats.duplicates[:3]:
            print(f"      {group[0]}  x{len(group)}")
    elif stats.images:
        print("\n  дублей не найдено")


def resolve(path_str: str) -> Path:
    """Относительный путь: сначала от текущей папки, потом от корня проекта."""
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    alt = cfg.BASE_DIR / p
    return alt if alt.exists() else p


def main() -> int:
    args = parse_args()
    with_md5 = not args.no_md5

    targets = []
    if args.dir:
        root = resolve(args.dir)
        targets.append((root.name, root / "images", root / "labels"))
    else:
        root = resolve(args.dataset)
        for split in args.splits:
            targets.append((split, root / split / "images", root / split / "labels"))

    print(f"Датасет: {root}")
    print(f"Проверка дублей (md5): {'да' if with_md5 else 'нет'}")

    problems = 0
    for name, images, labels in targets:
        stats = Stats()
        check_pair(images, labels, stats, with_md5)
        report(name, stats)
        problems += (stats.malformed_lines + stats.polygon_lines
                     + stats.out_of_range_lines + stats.degenerate_lines
                     + len(stats.img_without_label) + len(stats.duplicates)
                     + (0 if stats.images else 1)
                     + (1 if stats.source_counts.get("dc_luggage") else 0))

    print("\n" + ("Итог: дефектов не найдено." if not problems
                  else f"Итог: найдено проблем — {problems} (см. [!] выше)."))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
