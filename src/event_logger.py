"""Журнал сессии: CSV, JSONL, снимки, session.json и HTML-отчёт."""

from __future__ import annotations

import csv
import json
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config as cfg

# Человекочитаемые названия событий: используются в CSV, GUI и отчёте.
EVENT_TITLES_RU = {
    "alarm": "ТРЕВОГА: вещь оставлена",
    "alarm_cleared": "Тревога снята",
    "owner_left": "Владелец отошёл",
    "owner_unknown": "Сумка без владельца",
    "owner_returned": "Владелец вернулся",
    "owner_assigned": "Назначен владелец",
    "owner_forgotten": "Владелец пропал из кадра",
    "bystander_near": "Рядом посторонний",
    "bystander_left": "Посторонний ушёл",
    "bag_taken": "Сумку забрали",
    "bag_appeared": "Появилась сумка",
    "bag_stationary": "Сумка остановилась",
    "bag_lost": "Сумка пропала из кадра",
    "bag_relinked": "Трек сумки восстановлен",
    "bag_dedup": "Дубль сумки удалён",
    "manual_snapshot": "Снимок оператора",
}

# Уровень важности: определяет цвет строки в GUI/отчёте.
EVENT_LEVEL = {
    "alarm": "alarm",
    "owner_left": "warning",
    "owner_unknown": "warning",
    "bystander_near": "warning",
    "alarm_cleared": "info",
    "owner_returned": "info",
    "bag_taken": "info",
    "manual_snapshot": "info",
}

# События, которые оператор видит в журнале (остальные — служебные, только в jsonl).
VISIBLE_EVENTS = set(EVENT_LEVEL)

# Для этих событий сохраняем картинки.
SNAPSHOT_EVENTS = {"alarm", "owner_left", "owner_unknown", "bystander_near",
                   "bag_taken", "owner_returned", "alarm_cleared", "manual_snapshot"}

REASONS_RU = {
    "owner_left": "владелец ушёл",
    "owner_unknown": "владелец неизвестен",
    "owner_near": "владелец рядом",
    "bystander": "рядом посторонний",
    "carried": "сумку несут",
    "new": "новая",
    "reset": "сброс",
}

CSV_FIELDS = ["n", "wall_time", "video_time", "type", "title", "level", "bag_id",
              "status", "reason", "elapsed_sec", "owner_lid", "person_lid",
              "distance_px", "bbox", "frame_path", "crop_path"]


def _native(o):
    """numpy → python: json (Flask, jsonl) не умеет np.int64/np.float32."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    return o


def imwrite_any(path: Path, img) -> bool:
    """cv2.imwrite не умеет пути с кириллицей на Windows — пишем через imencode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def crop_with_margin(frame, bbox, margin: float = 1.2, min_side: int = 160):
    """Вырезка вокруг сумки с запасом — чтобы было видно, что рядом."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    side_w = max(int(bw * (1 + 2 * margin)), min_side)
    side_h = max(int(bh * (1 + 2 * margin)), min_side)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax1 = max(0, cx - side_w // 2)
    ay1 = max(0, cy - side_h // 2)
    ax2 = min(w, ax1 + side_w)
    ay2 = min(h, ay1 + side_h)
    crop = frame[ay1:ay2, ax1:ax2].copy()
    if crop.size == 0:
        return None
    # рамка сумки на вырезке
    cv2.rectangle(crop, (x1 - ax1, y1 - ay1), (x2 - ax1, y2 - ay1), (0, 0, 255), 2)
    return crop


class EventLogger:
    """Пишет события сессии на диск. Потокобезопасность обеспечивает вызывающий
    (pipeline вызывает логгер только из одного потока)."""

    def __init__(self, root: Path | str | None = None, source: str = "",
                 model: str = "", min_repeat_sec: float = 2.0,
                 save_snapshots: bool = True, meta: dict | None = None):
        root = Path(root) if root else cfg.BASE_DIR / "runs" / "sessions"
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = root / stamp
        n = 1
        while self.dir.exists():          # два запуска в одну секунду
            n += 1
            self.dir = root / f"{stamp}_{n}"
        self.snap_dir = self.dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)

        self.save_snapshots = save_snapshots
        self.min_repeat_sec = min_repeat_sec
        self.started = datetime.now()
        self.meta = {
            "source": str(source),
            "model": str(model),
            "started": self.started.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": self._settings_snapshot(),
        }
        if meta:
            self.meta.update(meta)

        self._csv_f = open(self.dir / "events.csv", "w", newline="", encoding="utf-8-sig")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=CSV_FIELDS, delimiter=";")
        self._csv.writeheader()
        self._jsonl = open(self.dir / "events.jsonl", "w", encoding="utf-8")

        self.counts: Counter = Counter()
        self.rows: list[dict] = []          # только видимые события (для отчёта)
        self._last_shot: dict[tuple, float] = {}
        self._n = 0
        self.max_video_time = 0.0
        self.closed = False

    @staticmethod
    def _settings_snapshot() -> dict:
        keys = ["ALARM_DELAY_SEC", "CONF_THRESHOLD", "IMGSZ", "BYSTANDER_ESCALATE",
                "OWNER_RADIUS_K", "REQUIRE_MOVED", "MAX_BAG_AREA_FRAC",
                "BAG_MISSING_SEC", "PERSON_MISSING_SEC", "MODEL_NAME"]
        return {k: getattr(cfg, k, None) for k in keys}

    # ------------------------------------------------------------------ запись

    def log(self, ev, frame_annotated=None, frame_raw=None) -> dict:
        """Записывает событие. Возвращает строку журнала (dict) — её показывает GUI."""
        self._n += 1
        d = ev.as_dict() if hasattr(ev, "as_dict") else dict(ev)
        etype = d.get("type", "")
        self.counts[etype] += 1
        self.max_video_time = max(self.max_video_time, float(d.get("time") or 0.0))

        row = {
            "n": self._n,
            "wall_time": datetime.now().strftime("%H:%M:%S"),
            "video_time": _native(d.get("time")),
            "type": etype,
            "title": EVENT_TITLES_RU.get(etype, etype),
            "level": EVENT_LEVEL.get(etype, "debug"),
            "bag_id": _native(d.get("bag_id")),
            "status": d.get("status", ""),
            "reason": REASONS_RU.get(d.get("reason", ""), d.get("reason", "")),
            "elapsed_sec": _native(d.get("elapsed")),
            "owner_lid": _native(d.get("owner_lid")),
            "person_lid": _native(d.get("person_lid")),
            "distance_px": _native(d.get("distance_px")),
            "bbox": _native(d.get("bbox")),
            "frame_path": "",
            "crop_path": "",
        }

        if self.save_snapshots and etype in SNAPSHOT_EVENTS and frame_annotated is not None:
            key = (etype, d.get("bag_id"))
            now = time.monotonic()
            if now - self._last_shot.get(key, -1e9) >= self.min_repeat_sec:
                self._last_shot[key] = now
                base = f"{self._n:04d}_{etype}_bag{d.get('bag_id')}"
                fp = self.snap_dir / f"{base}_frame.jpg"
                if imwrite_any(fp, frame_annotated):
                    row["frame_path"] = fp.relative_to(self.dir).as_posix()
                bbox = d.get("bbox")
                src = frame_raw if frame_raw is not None else frame_annotated
                if bbox is not None:
                    crop = crop_with_margin(src, bbox)
                    if crop is not None:
                        cp = self.snap_dir / f"{base}_crop.jpg"
                        if imwrite_any(cp, crop):
                            row["crop_path"] = cp.relative_to(self.dir).as_posix()

        csv_row = dict(row)
        csv_row["bbox"] = "" if row["bbox"] is None else " ".join(str(int(v)) for v in row["bbox"])
        self._csv.writerow(csv_row)
        self._csv_f.flush()
        self._jsonl.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        self._jsonl.flush()

        if etype in VISIBLE_EVENTS:
            self.rows.append(row)
        return row

    def manual_snapshot(self, frame_annotated, video_time: float = 0.0) -> dict:
        """Кнопка «Снимок» в GUI."""
        ev = {"type": "manual_snapshot", "time": video_time, "bag_id": -1,
              "status": "", "reason": "", "elapsed": 0.0, "bbox": None}
        return self.log(ev, frame_annotated, None)

    def abs_path(self, rel: str) -> Path | None:
        return (self.dir / rel) if rel else None

    # ---------------------------------------------------------------- закрытие

    def summary(self) -> dict:
        return {
            "alarms": self.counts.get("alarm", 0),
            "warnings": sum(self.counts.get(k, 0) for k in
                            ("owner_left", "owner_unknown", "bystander_near")),
            "bags_seen": self.counts.get("bag_appeared", 0),
            "events_total": self._n,
            "video_time_sec": round(self.max_video_time, 1),
            "duration_sec": round((datetime.now() - self.started).total_seconds(), 1),
            "by_type": dict(self.counts),
        }

    def close(self, extra: dict | None = None, build_report: bool = True) -> Path | None:
        if self.closed:
            return self.dir / "report.html"
        self.closed = True
        self._csv_f.close()
        self._jsonl.close()
        self.meta["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.meta["summary"] = self.summary()
        if extra:
            self.meta.update(extra)
        (self.dir / "session.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8")
        if build_report:
            from report import build_report
            return build_report(self.dir)
        return None

    def abort(self):
        """Аварийный старт (модель/источник не открылись): закрываем файлы и
        убираем папку сессии, если в неё не попало ни одного события."""
        if not self.closed:
            self.closed = True
            for f in (self._csv_f, self._jsonl):
                try:
                    f.close()
                except Exception:
                    pass
        if self._n == 0:
            shutil.rmtree(self.dir, ignore_errors=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (tuple, set)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)
