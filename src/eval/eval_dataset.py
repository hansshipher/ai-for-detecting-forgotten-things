#!/usr/bin/env python3
"""Оценка модели на размеченном сплите (mAP + P/R/F1 по классам).

Зачем отдельный скрипт: ultralytics.val() печатает красиво, но нам нужно
сравнивать модели между собой в одном формате и сохранять JSON в runs/.
Плюс holdout-сплит (ручная разметка людей) — единственное честное место,
где person-метрика не измеряет согласие с учителем.

Запуск (из папки src/):
    python eval/eval_dataset.py --weights ../weights/best26n_nodc_640.pt --split holdout
    python eval/eval_dataset.py --weights ../runs/train/x/weights/best.pt --split valid --conf 0.40

Выход: печать таблицы + runs/eval/<weights_stem>_<split>.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="Оценка модели на сплите датасета.")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(cfg.BASE_DIR / "dataset" / "12k_merged" / "data.yaml"))
    p.add_argument("--split", default="holdout",
                   help="train/val/test/holdout (для 'valid' ключ автоматически станет 'val')")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001,
                   help="порог для mAP (по умолчанию низкий, mAP считает кривую)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default=str(cfg.BASE_DIR / "runs" / "eval"))
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    split = "val" if args.split == "valid" else args.split
    metrics = model.val(
        data=args.data, split=split, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, batch=args.batch,
        device=args.device, verbose=False, plots=False,
    )

    names = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))
    result = {
        "weights": str(args.weights),
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "classes": {},
    }
    print(f"\n{Path(args.weights).name} @ {args.split} (imgsz={args.imgsz})")
    print(f"  mAP50={result['map50']:.4f}  mAP50-95={result['map50_95']:.4f}")
    print(f"  {'class':<10} {'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9} {'n_gt':>6}")
    for i, cname in names.items():
        p = float(metrics.box.p[i]) if i < len(metrics.box.p) else float("nan")
        r = float(metrics.box.r[i]) if i < len(metrics.box.r) else float("nan")
        ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else float("nan")
        ap = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else float("nan")
        result["classes"][cname] = {"precision": p, "recall": r,
                                    "map50": ap50, "map50_95": ap}
        print(f"  {cname:<10} {p:7.4f} {r:7.4f} {ap50:7.4f} {ap:9.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.weights).stem}_{args.split}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
