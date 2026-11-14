#!/usr/bin/env python3
"""Обучение детектора (обёртка над ultralytics с нашими дефолтами).

Демо-модель weights/best26n_nodc_640.pt этот скрипт не трогает: каждый
прогон пишется в runs/train/<name>, веса оттуда копируются в weights/
только вручную и осознанно.

Запуск (из папки src/):
    python train.py --data ../dataset/12k_merged/data.yaml --name cctv_yolo26n_12k_640
    python train.py --data ../dataset/data_all.yaml --name finetune_all_640 \
        --weights ../runs/train/cctv_yolo26n_12k_640/weights/best.pt --epochs 40

Дефолты повторяют прошлые прогоны (65 эпох, batch 32, 640, seed 0).
"""

import argparse
import sys
from pathlib import Path

import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="Обучение YOLO с дефолтами проекта.")
    p.add_argument("--data", required=True, help="путь к data.yaml")
    p.add_argument("--name", required=True, help="имя прогона (runs/train/<name>)")
    p.add_argument("--weights", default=str(cfg.BASE_DIR / "weights" / "yolo26n.pt"),
                   help="стартовые веса (по умолчанию COCO yolo26n)")
    p.add_argument("--epochs", type=int, default=65)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--project", default=str(cfg.BASE_DIR / "runs" / "train"))
    p.add_argument("--cache", action="store_true", help="кэшировать картинки в RAM")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        lr0=args.lr0,
        close_mosaic=args.close_mosaic,
        project=args.project,
        name=args.name,
        cache=args.cache,
        pretrained=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"[DONE] run: {save_dir}")
    print(f"[INFO] best: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
