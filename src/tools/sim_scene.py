#!/usr/bin/env python3
"""Синтетическая сцена для проверки логики, журнала и GUI без модели и без видео.

Сценарий (25 fps, 45 с):
   0–4 с   человек P1 входит слева и несёт сумку        -> SAFE (carried)
   4–6 с   ставит сумку и стоит рядом                   -> SAFE (owner_near)
   6–10 с  уходит вправо и пропадает из кадра           -> WARNING (owner_left)
  14–17 с  посторонний P2 проходит по дальнему плану    -> тревогу не отменяет
  ~22.8 с  прошло ALARM_DELAY_SEC без владельца         -> ALARM
  32–36 с  P1 возвращается. Без re-ID система не знает, что это владелец
           (он пропадал > PERSON_MISSING_SEC), поэтому для неё это «посторонний»
           и тревога держится                           -> ALARM (bystander_near)
  36–42 с  человек уносит сумку                         -> alarm_cleared (carried)

Детекции здесь «идеальные» — проверяется именно логика и обвязка, а не YOLO.

Запуск (из src/):
    python tools/sim_scene.py                 # прогон без окна, печать событий
    python gui.py --sim                       # то же в GUI
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H, FPS, DURATION = 960, 540, 25.0, 45.0
BAG_DOWN = (430, 400)


def _lerp(a, b, k):
    k = min(max(k, 0.0), 1.0)
    return a + (b - a) * k


def scene_at(t: float):
    """Возвращает (persons, bags) в формате tracker.ObjTracker.process_frame."""
    persons, bags = {}, {}

    def person(tid, x, feet_y=430, h=230):
        x1, x2 = int(x - 35), int(x + 35)
        y1, y2 = int(feet_y - h), int(feet_y)
        persons[tid] = {"center": ((x1 + x2) // 2, (y1 + y2) // 2), "feet": ((x1 + x2) // 2, y2),
                        "bbox": (x1, y1, x2, y2), "height": h, "conf": 0.9}

    def bag(tid, cx, cy, s=44):
        x1, y1, x2, y2 = int(cx - s // 2), int(cy - s // 2), int(cx + s // 2), int(cy + s // 2)
        bags[tid] = {"center": ((x1 + x2) // 2, (y1 + y2) // 2), "feet": ((x1 + x2) // 2, y2),
                     "bbox": (x1, y1, x2, y2), "height": s, "conf": 0.8}

    # P1
    if t < 4:
        x = _lerp(60, 390, t / 4)
        person(1, x)
        bag(2, x + 40, 360)                       # в руке
    elif t < 6:
        person(1, 390)
        bag(2, *BAG_DOWN)
    elif t < 10:
        x = _lerp(390, 1000, (t - 6) / 4)
        if x < W - 20:
            person(1, x)
        bag(2, *BAG_DOWN)
    elif t < 32:
        bag(2, *BAG_DOWN)
    elif t < 36:
        x = _lerp(1000, 470, (t - 32) / 4)
        if x < W - 20:
            person(1, x)
        bag(2, *BAG_DOWN)
    elif t < 42:
        x = _lerp(470, -100, (t - 36) / 6)
        if x > 20:
            person(1, x)
            bag(2, x - 40, 360)

    # P2 проходит по дальнему плану
    if 14 <= t < 17:
        person(3, _lerp(950, 0, (t - 14) / 3), feet_y=330, h=170)
    return persons, bags


def render(t: float, idx: int):
    img = np.full((H, W, 3), (58, 52, 48), np.uint8)
    cv2.rectangle(img, (0, 300), (W, H), (85, 85, 80), -1)        # пол
    persons, bags = scene_at(t)
    for p in persons.values():
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(img, (x1, y1 + 40), (x2, y2), (150, 110, 70), -1)
        cv2.circle(img, ((x1 + x2) // 2, y1 + 22), 22, (170, 190, 220), -1)
    for b in bags.values():
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 60, 140), -1)
    cv2.putText(img, "SIMULATION", (W - 190, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    # номер кадра зашит в первые пиксели: «трекер» узнаёт по нему время,
    # даже если очередь drop-old пропустила кадры
    img[0, 0:4] = np.frombuffer(np.uint32(idx).tobytes(), np.uint8)[:, None]
    return img


class SimCapture:
    """Подмена cv2.VideoCapture."""

    def __init__(self, duration: float = DURATION, fps: float = FPS):
        self.fps, self.n = fps, int(duration * fps)
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, render(self.i / self.fps, self.i)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.n
        return 0

    def set(self, *a):
        return True

    def release(self):
        pass


class SimTracker:
    """Подмена tracker.ObjTracker: детекции берутся из сценария."""

    def __init__(self, fps: float = FPS):
        self.fps = fps

    def process_frame(self, frame):
        idx = int(np.frombuffer(frame[0, 0:4, 0].tobytes(), np.uint32)[0])
        return scene_at(idx / self.fps)


def run_headless(realtime=False, log=True):
    from event_logger import EventLogger
    from pipeline import Pipeline

    logger = EventLogger(source="simulation", model="SimTracker") if log else None
    p = Pipeline("simulation", tracker=SimTracker(), capture=SimCapture(),
                 logger=logger, realtime=realtime, hud=True)
    p.start()
    while not p.finished:
        import time
        time.sleep(0.05)
    report = p.stop()
    rows = []
    while not p.events.empty():
        rows.append(p.events.get())
    return rows, report, logger


if __name__ == "__main__":
    rows, report, logger = run_headless()
    for r in rows:
        if r.get("level") != "debug":
            print(f"{float(r['video_time']):6.2f}s  {r['type']:<15} bag#{r['bag_id']} "
                  f"{r['status']:<8} {r.get('frame_path', '')}")
    print("report:", report)
