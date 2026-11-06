#!/usr/bin/env python3
"""Сборка единого train из train/valid/test с дедупликацией.

Зачем: сплиты сделаны по кадрам, а не по видео, поэтому val и test
пересекаются с train (одни и те же сцены, местами точные дубли).
Для маленького датасета выгоднее учить на всём, а честно мерить
на отдельном полевом наборе (videos/).

Что делает:
  - собирает изображения из dataset/{train,valid,test}/images;
  - выкидывает источники из --exclude (по умолчанию dc_luggage: комнатные
    фото рюкзаков, где 713 боксов занимают >50% кадра и 55 — весь кадр;
    модель из-за них считает "сумкой" любую крупную текстуру);
  - выкидывает точные дубли (md5 содержимого);
  - копирует в dataset/all/{images,labels} (train);
  - случайные --val-frac уводит в dataset/all_val/ — ТОЛЬКО для контроля
    кривой обучения (соседние кадры тех же видео остаются в train,
    поэтому mAP с них нельзя считать честным);
  - пишет dataset/data_all.yaml.

Использование (из папки src/):
    python merge_splits.py                      # отчёт, ничего не пишет
    python merge_splits.py --apply              # скопировать и создать yaml
    python merge_splits.py --apply --overwrite  # перезаписать готовый all/
"""

import argparse
import hashlib
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import config as cfg

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SRC_SPLITS = ("train", "valid", "test")


@dataclass
class Item:
    split: str
    image: Path
    label: Path | None


@dataclass
class Report:
    found: int = 0
    excluded: int = 0
    duplicates: list = field(default_factory=list)
    img_without_label: list = field(default_factory=list)
    label_without_img: list = field(default_factory=list)
    empty_labels: int = 0
    class_instances: Counter = field(default_factory=Counter)
    per_split: Counter = field(default_factory=Counter)


def parse_args():
    p = argparse.ArgumentParser(
        description="Собрать train/valid/test в один датасет без дублей.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=str(cfg.BASE_DIR / "dataset"),
                   help="корень датасета (по умолчанию dataset/ проекта)")
    p.add_argument("--apply", action="store_true",
                   help="реально копировать файлы (без флага — только отчёт)")
    p.add_argument("--overwrite", action="store_true",
                   help="разрешить перезапись уже существующего all/ и all_val/")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="доля кадров в мониторинговый val (по умолчанию 0.05)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed случайного отбора в val")
    p.add_argument("--exclude", nargs="*", default=["dc_luggage"],
                   help="префиксы имён файлов, которые не берём в обучение "
                        "(по умолчанию dc_luggage; передайте --exclude без "
                        "значений, чтобы взять всё)")
    return p.parse_args()


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect(root, report, exclude=()):
    """Собирает изображения всех сплитов, помечая дубли и сирот."""
    items, seen_md5 = [], {}
    for split in SRC_SPLITS:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            print(f"[{split}] нет папки {img_dir} — пропуск")
            continue

        images = sorted(p for p in img_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS)
        all_stems = {p.stem for p in images}
        for img in images:
            report.found += 1
            report.per_split[split] += 1
            if any(img.name.startswith(prefix) for prefix in exclude):
                report.excluded += 1
                continue
            label = lbl_dir / f"{img.stem}.txt"
            if not label.exists():
                label = None
                report.img_without_label.append(f"{split}/{img.name}")

            digest = md5(img)
            if digest in seen_md5:
                report.duplicates.append(f"{split}/{img.name} == {seen_md5[digest]}")
                continue
            seen_md5[digest] = f"{split}/{img.name}"
            items.append(Item(split, img, label))

        if lbl_dir.is_dir():
            for lbl in lbl_dir.glob("*.txt"):
                if lbl.stem not in all_stems and lbl.stat().st_size > 0:
                    report.label_without_img.append(f"{split}/{lbl.name}")
    return items


def scan_labels(items, report):
    """Считает инстансы по классам и пустые лейблы."""
    for it in items:
        if it.label is None:
            report.empty_labels += 1
            continue
        text = it.label.read_text(encoding="utf-8-sig").strip()
        if not text:
            report.empty_labels += 1
            continue
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) == 5:
                report.class_instances[int(float(tokens[0]))] += 1


def plan_split(items, val_frac, seed):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac))) if val_frac > 0 else 0
    return shuffled[n_val:], shuffled[:n_val]


def write_dataset(root, train_items, val_items, names, apply, report):
    out_train = root / "all"
    out_val = root / "all_val"

    for out_dir, items in ((out_train, train_items), (out_val, val_items)):
        img_dir = out_dir / "images"
        lbl_dir = out_dir / "labels"
        if apply:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
        for it in items:
            if not apply:
                continue
            dst_img = img_dir / it.image.name
            shutil.copy2(it.image, dst_img)
            dst_lbl = lbl_dir / f"{it.image.stem}.txt"
            if it.label is not None:
                shutil.copy2(it.label, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")   # фон, явно

    if not apply:
        return

    data = {
        "train": "../all/images",
        "val": "../all_val/images",
        "nc": len(names),
        "names": list(names),
    }
    (root / "data_all.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    root = Path(args.dataset).resolve()

    if args.apply:
        for name in ("all", "all_val"):
            target = root / name
            if target.exists() and any(target.iterdir()):
                if not args.overwrite:
                    print(f"ОШИБКА: {target} уже существует и не пуст.")
                    print("Добавьте --overwrite, чтобы перезаписать, или удалите вручную.")
                    return 2
                # перезапись = начать с чистого листа, иначе останутся
                # файлы от прошлой сборки (например, исключённые источники)
                for sub in ("images", "labels"):
                    if (target / sub).is_dir():
                        shutil.rmtree(target / sub)
                cache = target / "labels.cache"
                if cache.exists():
                    cache.unlink()

    src_yaml = root / "data.yaml"
    meta = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) if src_yaml.exists() else {}
    names = meta.get("names") or ["bag", "person"]

    report = Report()
    items = collect(root, report, exclude=args.exclude)
    scan_labels(items, report)

    if not items:
        print("Ничего не найдено — проверьте путь к датасету.")
        return 1

    train_items, val_items = plan_split(items, args.val_frac, args.seed)

    print(f"Датасет: {root}")
    print(f"Режим:   {'ПРИМЕНЕНИЕ' if args.apply else 'ТОЛЬКО ОТЧЁТ'}")
    print()
    print("Найдено изображений по сплитам:")
    for split in SRC_SPLITS:
        if report.per_split[split]:
            print(f"  {split:<6} {report.per_split[split]}")
    print(f"  всего  {report.found}")
    if args.exclude:
        print(f"  исключено по --exclude ({', '.join(args.exclude)}): {report.excluded}")
    print()
    print(f"Точных дублей (по md5):      {len(report.duplicates)}")
    print(f"Изображений без лейбла:      {len(report.img_without_label)}")
    print(f"Лейблов без изображения:     {len(report.label_without_img)}")
    print(f"Пустых лейблов (фон):        {report.empty_labels}")
    print()
    print("Инстансы по классам (до деления на train/val):")
    for cls_id, count in sorted(report.class_instances.items()):
        name = names[cls_id] if cls_id < len(names) else f"class{cls_id}"
        print(f"  {name:<8} {count}")
    print()
    print(f"Итог: train={len(train_items)}  val={len(val_items)} "
          f"(val_frac={args.val_frac}, seed={args.seed})")

    for label, lst in (("дубли", report.duplicates),
                       ("без лейбла", report.img_without_label),
                       ("лейблы-сироты", report.label_without_img)):
        if lst:
            print(f"\n{label} ({len(lst)}), первые 5:")
            for entry in lst[:5]:
                print(f"  {entry}")

    write_dataset(root, train_items, val_items, names, args.apply, report)

    if not args.apply:
        print("\nФайлы не тронуты. Для сборки запустите с --apply.")
        return 0

    print(f"\nГотово: {root / 'all'} , {root / 'all_val'} , {root / 'data_all.yaml'}")
    print("Проверьте, что ultralytics видит 0 corrupt, перед обучением.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
