#!/usr/bin/env python3
"""Полевой замер: сколько сумок модель находит на целевых видео.

Метрика на dataset/val завышена (соседние кадры тех же видео в train),
поэтому честный ответ даёт только прогон по videos/video1..4 — то есть
ровно по тем роликам, на которых будет демо.

Считает по каждому видео:
  - кадров обработано (каждый --step-й);
  - детекций bag и person;
  - долю кадров, где найдена хотя бы одна сумка;
  - медианный размер бокса сумки в пикселях (для сравнения масштаба).

Использование (из папки src/):
    python eval_videos.py                                  # best26n
    python eval_videos.py --models best26n best26s
    python eval_videos.py --model ../runs/train/cctv_yolo26n_all_640/weights/best.pt
    python eval_videos.py --save-dir ../runs/field_dump   # сохранить кадры с боксами
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config as cfg


def parse_args():
    p = argparse.ArgumentParser(
        description="Прогон моделей по videos/ с подсчётом детекций.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=["best26n"],
                   help="имена из weights/ или пути к .pt")
    p.add_argument("--videos", nargs="*", default=None,
                   help="файлы видео (по умолчанию videos/video1..4.*)")
    p.add_argument("--step", type=int, default=10,
                   help="брать каждый N-й кадр (по умолчанию 10)")
    p.add_argument("--imgsz", type=int, default=cfg.IMGSZ,
                   help=f"размер входа (по умолчанию {cfg.IMGSZ})")
    p.add_argument("--conf", type=float, default=cfg.CONF_THRESHOLD,
                   help=f"порог уверенности (по умолчанию {cfg.CONF_THRESHOLD})")
    p.add_argument("--save-dir", default=None,
                   help="куда сохранять кадры с боксами (по умолчанию не сохранять)")
    return p.parse_args()


def resolve_model(name):
    path = Path(name)
    if path.suffix == ".pt":
        return path if path.is_absolute() else (cfg.BASE_DIR / path)
    return cfg.BASE_DIR / "weights" / f"{name}.pt"


def find_videos(explicit):
    if explicit:
        return [Path(v) for v in explicit]
    vids = sorted((cfg.BASE_DIR / "videos").glob("video*.*"))
    return [v for v in vids if v.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}]


def evaluate(model, video, step, imgsz, conf, save_dir):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = 0
    bags = 0
    persons = 0
    frames_with_bag = 0
    bag_sizes = []
    idx = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            res = model.predict(frame, imgsz=imgsz, conf=conf,
                                classes=cfg.TARGET_CLASSES, verbose=False)[0]
            frames += 1
            boxes = res.boxes
            if boxes is not None and len(boxes):
                cls = boxes.cls.cpu().numpy().astype(int)
                xywh = boxes.xywh.cpu().numpy()
                n_bag = int((cls == cfg.BAG_CLASS).sum())
                bags += n_bag
                persons += int((cls == cfg.PERSON_CLASS).sum())
                if n_bag:
                    frames_with_bag += 1
                    for b in xywh[cls == cfg.BAG_CLASS]:
                        bag_sizes.append(float(max(b[2], b[3])))
            if save_dir is not None and frames % 5 == 0:
                out = save_dir / f"{video.stem}_{idx:06d}.jpg"
                cv2.imwrite(str(out), res.plot())
        idx += 1

    cap.release()
    elapsed = time.perf_counter() - t0
    return {
        "video": video.name,
        "size": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        "fps": fps,
        "total_frames": total,
        "frames": frames,
        "bags": bags,
        "persons": persons,
        "bag_frame_share": (frames_with_bag / frames) if frames else 0.0,
        "bag_median_px": float(np.median(bag_sizes)) if bag_sizes else 0.0,
        "ms_per_frame": (elapsed / frames * 1000.0) if frames else 0.0,
    }


def main():
    args = parse_args()
    from ultralytics import YOLO

    videos = find_videos(args.videos)
    if not videos:
        print("Не найдено видео в videos/")
        return 1

    save_root = None
    if args.save_dir:
        save_root = Path(args.save_dir)
        if not save_root.is_absolute():
            save_root = (cfg.BASE_DIR / save_root)
        save_root.mkdir(parents=True, exist_ok=True)

    print(f"Видео: {', '.join(v.name for v in videos)}")
    print(f"Каждый {args.step}-й кадр, imgsz={args.imgsz}, conf={args.conf}")
    print()

    for name in args.models:
        path = resolve_model(name)
        if not path.exists():
            print(f"НЕТ ФАЙЛА: {path}")
            continue
        model = YOLO(str(path))
        names = model.names
        print(f"=== {name} ===")
        if names.get(cfg.BAG_CLASS) != "bag":
            print(f"  ВНИМАНИЕ: класс {cfg.BAG_CLASS} называется '{names.get(cfg.BAG_CLASS)}', "
                  f"а не 'bag' — метрики ниже не про сумки")

        dump = None
        if save_root is not None:
            dump = save_root / Path(name).stem
            dump.mkdir(parents=True, exist_ok=True)

        header = (f"  {'видео':<14}{'кадров':>8}{'bag':>6}{'person':>8}"
                  f"{'кадров с bag':>15}{'медиана bag, px':>18}{'мс/кадр':>10}")
        print(header)
        for video in videos:
            r = evaluate(model, video, args.step, args.imgsz, args.conf, dump)
            if r is None:
                print(f"  {video.name:<14} не открылось")
                continue
            print(f"  {r['video']:<14}{r['frames']:>8}{r['bags']:>6}{r['persons']:>8}"
                  f"{r['bag_frame_share'] * 100:>14.1f}%{r['bag_median_px']:>18.0f}"
                  f"{r['ms_per_frame']:>10.1f}")
        print()

    if save_root is not None:
        print(f"Кадры с боксами: {save_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
