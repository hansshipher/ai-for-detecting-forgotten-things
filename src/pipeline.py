"""Конвейер: источник → YOLO+ByteTrack → логика → отрисовка → журнал."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path

import cv2

import config as cfg
from logic import AbObjDetector
from visualizer import Visualizer

PRINT_EVENTS = {"alarm", "owner_left", "owner_unknown", "bystander_near",
                "bag_taken", "owner_returned", "alarm_cleared"}


def parse_source(src: str):
    """'0' -> 0 (веб-камера), остальное — путь/URL как есть."""
    s = str(src).strip()
    return int(s) if s.isdigit() else s


def visible_bags(tracked_bags: dict) -> dict:
    """Сумки, которые реально показываются оператору (те же правила, что в Visualizer)."""
    out = {}
    for bid, b in tracked_bags.items():
        if b.get("missing_sec", 0.0) > cfg.GHOST_GRACE_SEC:
            continue
        if not b.get("was_moving", False) and b.get("age", 0.0) > cfg.STATIC_HIDE_SEC:
            continue
        out[bid] = b
    return out


def overall_status(bags: dict) -> str:
    st = {b.get("status") for b in bags.values()}
    if "ALARM" in st:
        return "ALARM"
    if "WARNING" in st:
        return "WARNING"
    return "SAFE"


class Pipeline:
    def __init__(self, source, camera: bool = False, tracker=None,
                 model_path: str | None = None, logger=None,
                 record_path: str | Path | None = None, realtime: bool = True,
                 hud: bool = True, capture=None, verbose: bool = True):
        self.source = source
        self.camera = camera
        self.logger = logger
        self.record_path = Path(record_path) if record_path else None
        self.realtime = realtime
        self.hud = hud
        self.verbose = verbose

        if tracker is None:
            from tracker import ObjTracker   # тянет ultralytics — импортируем только здесь
            tracker = ObjTracker(model_path)
        self.tracker = tracker
        self.detector = AbObjDetector()
        self.visualizer = Visualizer()

        self.cap = capture if capture is not None else cv2.VideoCapture(parse_source(source))
        if not self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
            raise RuntimeError(
                f"Не удалось открыть источник: {source}\n"
                "Проверьте путь/адрес камеры. На Windows OpenCV не открывает файлы, "
                "в пути к которым есть кириллица — переложите видео, например, в C:\\videos")
        if camera:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 0 < fps < 240 else 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not camera else 0

        self.events: queue.Queue = queue.Queue()       # строки журнала для GUI
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._reader_done = threading.Event()
        self._writer = None
        self._snapshot_request = False

        self._view = {
            "frame": None, "raw": None, "vtime": 0.0, "frame_idx": 0,
            "persons": 0, "bags": 0, "warnings": 0, "alarms_now": 0,
            "status": "SAFE", "infer_fps": 0.0, "infer_ms": 0.0,
            "finished": False, "error": None, "progress": 0.0, "bag_list": [],
        }
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ API

    def start(self):
        self._threads = [
            threading.Thread(target=self._reader, name="reader", daemon=True),
            threading.Thread(target=self._worker, name="worker", daemon=True),
        ]
        for t in self._threads:
            t.start()
        return self

    def pause(self, on: bool | None = None):
        on = (not self._paused.is_set()) if on is None else on
        if on:
            self._paused.set()
        else:
            self._paused.clear()
        return on

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._view["finished"]

    def request_snapshot(self):
        """Ручной снимок оператора: сохраняется ближайший обработанный кадр."""
        self._snapshot_request = True

    def get_view(self) -> dict:
        with self._lock:
            return dict(self._view)

    def stop(self, build_report: bool = True):
        """Останавливает потоки, закрывает файлы. Возвращает путь к report.html (если есть)."""
        self._stop.set()
        self._paused.clear()
        for t in self._threads:
            t.join(timeout=3.0)
        if any(t.is_alive() for t in self._threads):
            # read() на RTSP/сетевой камере может блокироваться надолго:
            # освобождаем источник, чтобы поток чтения проснулся и вышел.
            try:
                self.cap.release()
            except Exception:
                pass
            for t in self._threads:
                t.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        report = None
        if self.logger is not None:
            report = self.logger.close(
                extra={"fps_source": self.fps, "realtime": self.realtime,
                       "record": str(self.record_path) if self.record_path else None},
                build_report=build_report)
        return report

    # -------------------------------------------------------------- потоки

    def _reader(self):
        frame_idx = 0
        frame_delay = 1.0 / self.fps
        start_wall = time.time()
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.05)
                    # после паузы темп отсчитываем заново
                    start_wall = time.time() - frame_idx * frame_delay
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    if self.camera:
                        time.sleep(0.01)    # камера моргнула — ждём
                        continue
                    break
                frame_idx += 1
                vtime = time.time() if self.camera else frame_idx / self.fps
                item = (frame, vtime, frame_idx)

                if self.realtime:
                    # drop-old: затираем кадр, который worker не успел взять
                    try:
                        self._frames.put_nowait(item)
                    except queue.Full:
                        try:
                            self._frames.get_nowait()
                        except queue.Empty:
                            pass
                        self._frames.put_nowait(item)
                    if not self.camera:
                        sleep_s = start_wall + frame_idx * frame_delay - time.time()
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                else:
                    while not self._stop.is_set():
                        try:
                            self._frames.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        finally:
            self._reader_done.set()

    def _worker(self):
        n_fps, t_fps, fps_val = 0, time.time(), 0.0
        try:
            while not self._stop.is_set():
                try:
                    frame, vtime, frame_idx = self._frames.get(timeout=0.1)
                except queue.Empty:
                    if self._reader_done.is_set() and self._frames.empty():
                        break
                    continue

                t0 = time.perf_counter()
                persons, bags = self.tracker.process_frame(frame)
                tracked = self.detector.update(persons, bags, vtime, frame.shape)
                events = self.detector.drain_events()
                infer_ms = (time.perf_counter() - t0) * 1000

                out = self.visualizer.draw_status(frame.copy(), persons, tracked)
                vis = visible_bags(tracked)
                status = overall_status(vis)

                n_fps += 1
                now = time.time()
                if now - t_fps >= 1.0:
                    fps_val = n_fps / (now - t_fps)
                    n_fps, t_fps = 0, now

                if self.hud:
                    self._draw_hud(out, status, vtime, fps_val)

                for ev in events:
                    self._handle_event(ev, out, frame, vtime)

                if self._snapshot_request and self.logger is not None:
                    self._snapshot_request = False
                    self.events.put(self.logger.manual_snapshot(out, vtime))

                if self.record_path is not None:
                    self._record(out)

                with self._lock:
                    self._view.update(
                        frame=out, raw=frame, vtime=vtime, frame_idx=frame_idx,
                        persons=len(persons), bags=len(vis),
                        warnings=sum(1 for b in vis.values() if b["status"] == "WARNING"),
                        alarms_now=sum(1 for b in vis.values() if b["status"] == "ALARM"),
                        status=status, infer_fps=fps_val, infer_ms=infer_ms,
                        progress=(frame_idx / self.total_frames) if self.total_frames else 0.0,
                        bag_list=[{"id": int(bid), "status": b["status"],
                                   "reason": b.get("reason", ""),
                                   "elapsed": round(float(b.get("elapsed", 0.0)), 1)}
                                  for bid, b in sorted(vis.items())],
                    )
        except Exception as e:
            traceback.print_exc()
            with self._lock:
                self._view["error"] = f"{type(e).__name__}: {e}"
        finally:
            with self._lock:
                self._view["finished"] = True

    # ------------------------------------------------------------ helpers

    def _handle_event(self, ev, annotated, raw, vtime):
        if self.verbose and ev.type in PRINT_EVENTS:
            print(f"[{vtime:7.2f}s] {ev.type:<14} bag#{ev.bag_id} status={ev.status} "
                  f"reason={ev.reason} elapsed={ev.elapsed:.1f}s", flush=True)
        if self.logger is not None:
            row = self.logger.log(ev, annotated, raw)
        else:
            row = ev.as_dict()
            row["title"] = ev.type
            row["level"] = "debug"
        self.events.put(row)

    def _record(self, frame):
        if self._writer is None:
            h, w = frame.shape[:2]
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.record_path), fourcc, self.fps, (w, h))
        self._writer.write(frame)

    def _draw_hud(self, img, status, vtime, fps_val):
        color = {"SAFE": cfg.COLOR_SAFE, "WARNING": cfg.COLOR_WARNING,
                 "ALARM": cfg.COLOR_ALARM}[status]
        mode = "CAM" if self.camera else "FILE"
        t = "" if self.camera else f" | t={vtime:.1f}s"
        cv2.putText(img, f"{mode} | FPS: {fps_val:.1f}{t}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if status == "ALARM":
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 6)
