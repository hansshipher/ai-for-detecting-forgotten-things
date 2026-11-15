#!/usr/bin/env python3
"""Авторазметка людей COCO-детектором для обучения своей модели."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Авторазметка person COCO-моделью (учителем) для 12k-dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", default=str(cfg.BASE_DIR / "dataset" / "12k-dataset"),
                   help="корень датасета (train/valid/test)")
    p.add_argument("--out", default=str(cfg.BASE_DIR / "runs" / "12k_person_labels"),
                   help="куда писать person-лейблы")
    p.add_argument("--weights", default=str(cfg.BASE_DIR / "weights" / "yolo26m.pt"),
                   help="COCO-веса учителя")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.40,
                   help="порог учителя (0.40: P=0.70 R=0.87 на ручной разметке)")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--batch", type=int, default=8, help="картинок за раз (память GPU)")
    p.add_argument("--device", default="0")
    p.add_argument("--splits", nargs="*", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--limit", type=int, default=0, help="обработать только N картинок (отладка)")
    p.add_argument("--overwrite", action="store_true", help="перезаписать готовые лейблы")
    return p.parse_args()


def list_images(root, split):
    d = Path(root) / split / "images"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def main():
    args = parse_args()
    from ultralytics import YOLO

    root = Path(args.root)
    out_root = Path(args.out)
    model = YOLO(args.weights)

    t0 = time.perf_counter()
    total_done = total_boxes = total_imgs = 0
    stats = {"conf": args.conf, "imgsz": args.imgsz, "weights": args.weights, "splits": {}}

    for split in args.splits:
        images = list_images(root, split)
        if not images:
            print(f"[WARN] no images in {root / split / 'images'}")
            continue

        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        todo = images if args.overwrite else [p for p in images if not (out_dir / f"{p.stem}.txt").exists()]
        skipped = len(images) - len(todo)
        if args.limit:
            todo = todo[:args.limit]
        print(f"[{split}] images={len(images)} | todo={len(todo)} | already done={skipped}", flush=True)
        if not todo:
            stats["splits"][split] = {"images": len(images), "annotated": 0, "boxes": 0, "skipped": skipped}
            continue

        n_boxes = 0
        n_with = 0
        done = 0
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            try:
                results = model.predict(
                    [str(p) for p in chunk],
                    imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                    max_det=args.max_det, classes=[0],
                    verbose=False, device=args.device,
                )
            except RuntimeError as e:
                # OOM на большом батче: падаем на поштучно
                print(f"  [WARN] batch failed ({type(e).__name__}), retry one by one", flush=True)
                import torch
                torch.cuda.empty_cache()
                results = []
                for p in chunk:
                    results.extend(model.predict(
                        str(p), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                        max_det=args.max_det, classes=[0],
                        verbose=False, device=args.device,
                    ))

            for img_path, res in zip(chunk, results):
                lines = []
                if res.boxes is not None and len(res.boxes):
                    xywhn = res.boxes.xywhn.cpu().numpy()
                    for x in xywhn:
                        cx, cy, w, h = (float(v) for v in x)
                        if w <= 0 or h <= 0:
                            continue
                        lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                (out_dir / f"{img_path.stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                n_boxes += len(lines)
                n_with += 1 if lines else 0
                done += 1

            total_done += len(chunk)
            if done % 400 < args.batch:
                el = time.perf_counter() - t0
                rate = total_done / max(el, 1e-9)
                print(f"  [{split}] {done}/{len(todo)} | boxes={n_boxes} | "
                      f"{rate:.1f} img/s | elapsed={el/60:.1f} min", flush=True)

        total_boxes += n_boxes
        total_imgs += done
        stats["splits"][split] = {"images": len(images), "annotated": done,
                                  "with_person": n_with, "boxes": n_boxes, "skipped": skipped}
        print(f"[{split}] done: {done} images, {n_boxes} person boxes, "
              f"{n_with} images with at least one person", flush=True)

    stats["elapsed_min"] = round((time.perf_counter() - t0) / 60, 2)
    stats["total_images"] = total_imgs
    stats["total_boxes"] = total_boxes
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"\n[DONE] images={total_imgs} boxes={total_boxes} "
          f"elapsed={stats['elapsed_min']} min")
    print(f"[INFO] labels: {out_root}")
    print(f"[INFO] stats:  {out_root / 'stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
