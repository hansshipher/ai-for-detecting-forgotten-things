#!/usr/bin/env python3
"""Офлайн-прогон логики по видео: проверяем, что система не врёт.

Что проверяется:

  1. Сумку НЕСУТ — тревоги быть не может (правило стационарности).
  2. Сумку ОСТАВИЛИ — ALARM появляется примерно через ALARM_DELAY_SEC.
  3. Пропуск кадров (--check) не сдвигает время тревоги больше чем на 1 с.
     Это прямая проверка, что логика считает секунды, а не кадры.

Запуск (из папки src/):
    python eval_logic.py                          # все видео, каждый кадр
    python eval_logic.py --stride 3               # имитация слабого ноутбука
    python eval_logic.py --check                  # сравнить stride 1 и 3
    python eval_logic.py --save-dir ../runs/logic_check
    python eval_logic.py --csv ../runs/logic_eval.csv
"""

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

import config as cfg
from logic import AbObjDetector
from tracker import ObjTracker
from visualizer import Visualizer

VIDEOS = ("video1", "video2", "video3", "video4")
INTERESTING = {"alarm", "owner_left", "owner_unknown", "bystander_near",
               "bag_taken", "owner_returned"}


@dataclass
class BagStats:
    first: float = 0.0
    last: float = 0.0
    alarms: list = field(default_factory=list)
    max_elapsed: float = 0.0
    statuses: set = field(default_factory=set)
    reasons: set = field(default_factory=set)


@dataclass
class VideoResult:
    video: str
    frames: int = 0
    processed: int = 0
    bags: dict = field(default_factory=dict)
    events: Counter = field(default_factory=Counter)
    alarm_times: list = field(default_factory=list)
    infer_ms: float = 0.0
    logic_ms: float = 0.0
    rows: list = field(default_factory=list)


def parse_args():
    p = argparse.ArgumentParser(
        description="Офлайн-проверка логики забытых вещей на videos/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--videos", nargs="+", default=list(VIDEOS))
    p.add_argument("--model", default=None,
                   help="путь к весам (по умолчанию из config.py)")
    p.add_argument("--stride", type=int, default=1,
                   help="обрабатывать каждый N-й кадр (по умолчанию 1)")
    p.add_argument("--check", action="store_true",
                   help="сравнить stride 1 и stride 3 (тест на независимость от FPS)")
    p.add_argument("--save-dir", default=None,
                   help="куда сохранять кадры на событиях (от корня проекта)")
    p.add_argument("--csv", default=None, help="CSV с покадровыми состояниями")
    p.add_argument("--max-shots", type=int, default=40,
                   help="сколько кадров максимум сохранять на видео")
    return p.parse_args()


def resolve(path_str: str) -> Path:
    """Относительный путь: сначала от текущей папки, потом от корня проекта."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return cwd_path.resolve()
    base_path = cfg.BASE_DIR / p
    if base_path.exists():
        return base_path.resolve()
    return cwd_path.resolve()


def run_video(video: str, model_path, stride: int, save_dir: Path | None,
              max_shots: int) -> VideoResult:
    path = cfg.BASE_DIR / "videos" / f"{video}.avi"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = ObjTracker(model_path=model_path)
    detector = AbObjDetector()
    visualizer = Visualizer()
    result = VideoResult(video=video)

    shots = 0
    idx = 0
    infer_total = 0.0
    logic_total = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result.frames += 1
        if idx % stride != 0:
            idx += 1
            continue

        vtime = idx / fps

        t0 = time.perf_counter()
        persons, bags = tracker.process_frame(frame)
        infer_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        tracked = detector.update(persons, bags, vtime, frame.shape)
        logic_total += time.perf_counter() - t0

        for ev in detector.drain_events():
            result.events[ev.type] += 1
            if ev.type == "alarm":
                result.alarm_times.append(round(vtime, 2))
                bs = result.bags.setdefault(ev.bag_id, BagStats())
                bs.alarms.append(round(vtime, 2))

            if save_dir is not None and ev.type in INTERESTING and shots < max_shots:
                save_dir.mkdir(parents=True, exist_ok=True)
                out = visualizer.draw_status(frame.copy(), persons, tracked)
                name = f"{video}_{ev.type}_{idx:06d}.jpg"
                cv2.imwrite(str(save_dir / name), out)
                shots += 1

        for bid, st in tracked.items():
            bs = result.bags.setdefault(bid, BagStats())
            if bs.first == 0.0:
                bs.first = round(vtime, 2)
            bs.last = round(vtime, 2)
            bs.max_elapsed = max(bs.max_elapsed, st.get("elapsed", 0.0))
            bs.statuses.add(st["status"])
            bs.reasons.add(st.get("reason", ""))
            if st.get("missing_sec", 0.0) > cfg.GHOST_GRACE_SEC:
                continue
            result.rows.append({
                "video": video,
                "time": round(vtime, 2),
                "bag_id": bid,
                "status": st["status"],
                "reason": st.get("reason", ""),
                "elapsed": round(st.get("elapsed", 0.0), 2),
                "owner_lid": st.get("owner_lid"),
                "distance": None if st.get("owner_distance") is None
                            else round(st["owner_distance"], 1),
                "stationary": int(bool(st.get("stationary"))),
                "x1": st["bbox"][0], "y1": st["bbox"][1],
                "x2": st["bbox"][2], "y2": st["bbox"][3],
            })

        result.processed += 1
        idx += 1

    cap.release()
    if result.processed:
        result.infer_ms = infer_total / result.processed * 1000
        result.logic_ms = logic_total / result.processed * 1000
    return result


def print_video(res: VideoResult) -> None:
    print(f"\n=== {res.video}.avi ===")
    print(f"  кадров: {res.frames}, обработано: {res.processed} "
          f"({res.infer_ms:.1f} мс инференс + {res.logic_ms:.1f} мс логика)")
    print(f"  событий: " + (", ".join(f"{k}={v}" for k, v in sorted(res.events.items()))
                             or "нет"))
    if not res.bags:
        print("  сумок не найдено")
        return

    print(f"  {'bag':>5} {'первый':>8} {'последний':>10} {'max t':>7} "
          f"{'ALARM':>7}  статусы")
    for bid, bs in sorted(res.bags.items()):
        alarms = f"{bs.alarms[0]:.1f}s" if bs.alarms else "-"
        print(f"  {bid:>5} {bs.first:>8.1f} {bs.last:>10.1f} {bs.max_elapsed:>7.1f} "
              f"{alarms:>7}  {','.join(sorted(bs.statuses))}")

    if res.alarm_times:
        print(f"  ALARM в моменты: {res.alarm_times}")
    else:
        print("  ALARM не было")


def compare_strides(results: dict[str, list[VideoResult]]) -> bool:
    """Сравнивает время тревог при stride 1 и stride 3."""
    print("\n" + "=" * 60)
    print("ТЕСТ НЕЗАВИСИМОСТИ ОТ FPS (stride 1 vs stride 3)")
    print("=" * 60)
    ok = True
    for video in results:
        a = results[video][0].alarm_times
        b = results[video][1].alarm_times
        if len(a) != len(b):
            print(f"  {video}: РАСХОЖДЕНИЕ — тревог {len(a)} против {len(b)}")
            ok = False
            continue
        diffs = [abs(x - y) for x, y in zip(a, b)]
        worst = max(diffs) if diffs else 0.0
        verdict = "OK" if worst <= 1.0 else "ПЛОХО"
        if worst > 1.0:
            ok = False
        print(f"  {video}: тревог {len(a)}, макс. сдвиг {worst:.2f} с — {verdict}")
    return ok


def main() -> int:
    args = parse_args()
    save_dir = resolve(args.save_dir) if args.save_dir else None
    csv_path = resolve(args.csv) if args.csv else None

    strides = [1, 3] if args.check else [args.stride]
    results: dict[str, list[VideoResult]] = {}

    all_rows = []
    for stride in strides:
        if args.check:
            print(f"\n########## stride = {stride} ##########")
        for video in args.videos:
            res = run_video(video, args.model, stride, save_dir, args.max_shots)
            results.setdefault(video, []).append(res)
            print_video(res)
            all_rows.extend(res.rows)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV: {csv_path} ({len(all_rows)} строк)")

    if save_dir is not None:
        print(f"кадры: {save_dir}")

    if args.check:
        return 0 if compare_strides(results) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
