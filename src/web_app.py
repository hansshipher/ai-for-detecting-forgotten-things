#!/usr/bin/env python3
"""Веб-панель оператора: открывается в браузере, работает на этом же компьютере.

    cd src
    python web_app.py                 # http://127.0.0.1:8000 откроется сам
    python web_app.py --port 8080
    python web_app.py --host 0.0.0.0  # открыть панель с телефона/другого ПК в той же сети

Внутри тот же pipeline.Pipeline, что у test.py и gui.py: модель, логика, журнал
и отчёт одинаковые. Сервер только отдаёт картинку (MJPEG) и состояние (JSON).
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import webbrowser
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, request, send_from_directory

import config as cfg
from event_logger import EventLogger
from pipeline import Pipeline

WEB_DIR = Path(__file__).resolve().parent / "web"
SESSIONS_DIR = cfg.BASE_DIR / "runs" / "sessions"
WEIGHTS_DIR = cfg.BASE_DIR / "weights"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}
SIM_SOURCE = "__sim__"

app = Flask(__name__, static_folder=None)


class State:
    """Одна активная сессия на сервер (одна камера — один оператор)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pipeline: Pipeline | None = None
        self.logger: EventLogger | None = None
        self.session: Path | None = None
        self.source_label = ""
        self.events: list[dict] = []
        self.alarm_total = 0
        self.report: Path | None = None
        self.error: str | None = None
        self.starting = False
        self.camera = False

    def drain(self, p=None):
        p = p or self.pipeline
        if p is None:
            return
        while not p.events.empty():
            row = dict(p.events.get_nowait())
            row["id"] = len(self.events) + 1
            if row.get("type") == "alarm":
                self.alarm_total += 1
            self.events.append(row)

    def stop(self):
        p, self.pipeline = self.pipeline, None
        if p is None:
            return
        try:
            self.report = p.stop()
        except Exception as e:      # отчёт не должен ронять сервер
            self.error = f"Не удалось сохранить отчёт: {e}"
        self.drain(p)               # события последних кадров


S = State()


# ------------------------------------------------------------------ страницы

@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/web/<path:name>")
def web_static(name):
    return send_from_directory(WEB_DIR, name)


@app.get("/sessions/<path:name>")
def session_file(name):
    return send_from_directory(SESSIONS_DIR, name)


# ----------------------------------------------------------------------- API

def _settings():
    return {"alarm_delay": cfg.ALARM_DELAY_SEC, "conf": cfg.CONF_THRESHOLD,
            "escalate": cfg.BYSTANDER_ESCALATE}


def _resolve_model(raw) -> Path:
    """Модель — только .pt из папки weights/.

    .pt загружается через pickle (torch.load), поэтому путь из браузера — это
    потенциальное выполнение чужого кода. Панель принимает только свои веса.
    """
    p = Path(raw).expanduser() if raw else Path(cfg.MODEL_PATH)
    if not p.is_absolute():
        p = WEIGHTS_DIR / p
    try:
        p = p.resolve()
    except OSError as e:
        raise ValueError(f"Некорректный путь к модели: {raw}") from e
    if p.suffix.lower() != ".pt" or WEIGHTS_DIR.resolve() not in p.parents:
        raise ValueError(f"Модель должна быть .pt-файлом из папки {WEIGHTS_DIR}.")
    if not p.is_file():
        raise ValueError(f"Нет файла модели: {p.name}. Скачайте best12k.pt по ссылке "
                         "из readme.md и положите в папку weights/.")
    return p


def _number(d: dict, key: str, lo: float, hi: float) -> float:
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        raise ValueError(f"Параметр «{key}» должен быть числом.")
    if not math.isfinite(v):
        raise ValueError(f"Параметр «{key}» должен быть конечным числом.")
    return min(max(v, lo), hi)


@app.get("/api/options")
def options():
    videos = []
    vdir = cfg.BASE_DIR / "videos"
    if vdir.is_dir():
        videos = [{"label": p.name, "value": str(p)} for p in sorted(vdir.iterdir())
                  if p.suffix.lower() in VIDEO_EXT]
    weights = []
    wdir = cfg.BASE_DIR / "weights"
    if wdir.is_dir():
        weights = [{"label": p.name, "value": str(p)} for p in sorted(wdir.glob("*.pt"))]
    return jsonify(videos=videos, weights=weights, default_model=str(cfg.MODEL_PATH),
                   default_video=str(cfg.VIDEO_PATH), settings=_settings(), sim=SIM_SOURCE)


@app.post("/api/start")
def start():
    d = request.get_json(silent=True) or {}
    with S.lock:
        if S.pipeline is not None and S.pipeline.finished:
            S.stop()                # видео закончилось, но страницу никто не опрашивал
        if S.pipeline is not None or S.starting:
            return jsonify(ok=False, error="Наблюдение уже идёт. Сначала остановите его."), 409
        S.starting = True
    try:
        source = str(d.get("source", "")).strip()
        sim = source == SIM_SOURCE
        camera = bool(d.get("camera")) or source.isdigit() or \
            source.lower().startswith(("rtsp://", "http://", "https://"))
        if not source:
            raise ValueError("Выберите видео или камеру.")
        if not sim and not camera and not Path(source).exists():
            raise ValueError(f"Файл не найден: {source}")
        model = None if sim else _resolve_model(d.get("model"))

        tracker = capture = None
        source_label = Path(source).name if not camera else source
        if sim:
            from tools.sim_scene import SimCapture, SimTracker
            tracker, capture = SimTracker(), SimCapture()
            model_label, source_label = "Симуляция (без модели)", "Тестовая сцена"
        else:
            model_label = model.name

        logger = p = None
        try:
            logger = EventLogger(source=source_label, model=model_label,
                                 save_snapshots=bool(d.get("snapshots", True)))
            record = (logger.dir / "annotated.mp4") if d.get("record") else None
            p = Pipeline(source, camera=camera, tracker=tracker,
                         model_path=str(model) if model else None,
                         logger=logger, record_path=record,
                         realtime=bool(d.get("realtime", True)) or camera,
                         hud=False, capture=capture)
            p.start()
        except Exception:
            # модель/источник не открылись: закрываем файлы и убираем пустую сессию
            if p is not None:
                try:
                    p.stop(build_report=False)
                except Exception:
                    pass
            if logger is not None:
                logger.abort()
            raise

        with S.lock:
            S.pipeline, S.logger, S.session = p, logger, logger.dir
            S.events, S.alarm_total, S.report, S.error = [], 0, None, None
            S.source_label, S.camera = source_label, camera
        return jsonify(ok=True, session=logger.dir.name)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400
    finally:
        S.starting = False


@app.post("/api/pause")
def pause():
    with S.lock:
        if S.pipeline is None:
            return jsonify(ok=False), 409
        return jsonify(ok=True, paused=S.pipeline.pause())


@app.post("/api/stop")
def stop():
    with S.lock:
        S.stop()
    return jsonify(ok=True)


@app.post("/api/snapshot")
def snapshot():
    with S.lock:
        if S.pipeline is None:
            return jsonify(ok=False), 409
        S.pipeline.request_snapshot()
    return jsonify(ok=True)


@app.post("/api/settings")
def settings():
    d = request.get_json(silent=True) or {}
    try:
        if "alarm_delay" in d:
            cfg.ALARM_DELAY_SEC = round(_number(d, "alarm_delay", 3, 120))
        if "conf" in d:
            cfg.CONF_THRESHOLD = round(_number(d, "conf", 0.05, 0.95), 2)
        if "escalate" in d:
            cfg.BYSTANDER_ESCALATE = bool(d["escalate"])
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, settings=_settings())


@app.get("/api/state")
def state():
    raw_since = request.args.get("since", "0")
    try:
        since = int(raw_since)
    except ValueError:
        return jsonify(ok=False, error="Параметр since должен быть целым числом."), 400
    with S.lock:
        p = S.pipeline
        S.drain()
        view = p.get_view() if p else None
        if p is not None and (view["finished"] or view["error"]):
            if view["error"]:
                S.error = view["error"]
            S.stop()
        running = S.pipeline is not None
        out = {
            "running": running,
            "starting": S.starting,
            "paused": bool(p and p.paused and running),
            "camera": S.camera,
            "source": S.source_label,
            "session": S.session.name if S.session else None,
            "report": (f"/sessions/{S.session.name}/report.html"
                       if S.report and S.session else None),
            "error": S.error,
            "alarm_total": S.alarm_total,
            "events": [e for e in S.events if e["id"] > since],
            "settings": _settings(),
        }
        if view:
            out.update({k: view[k] for k in ("vtime", "persons", "bags", "warnings", "alarms_now",
                                             "status", "infer_fps", "infer_ms", "progress",
                                             "bag_list")})
    return jsonify(out)


@app.get("/api/sessions")
def sessions():
    items = []
    if SESSIONS_DIR.is_dir():
        for d in sorted(SESSIONS_DIR.iterdir(), reverse=True)[:30]:
            meta_p = d / "session.json"
            if not meta_p.exists():
                continue
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append({
                "name": d.name, "source": meta.get("source"), "started": meta.get("started"),
                "summary": meta.get("summary", {}),
                "report": f"/sessions/{d.name}/report.html" if (d / "report.html").exists() else None,
            })
    return jsonify(items)


@app.get("/stream.mjpg")
def stream():
    def gen():
        last = None
        while True:
            p = S.pipeline
            frame = None
            if p is not None:
                v = p.get_view()
                frame = v["frame"]
            if frame is not None and frame is not last:
                last = frame
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/frame.jpg")
def frame_jpg():
    """Один кадр — запасной вариант, если браузер не держит MJPEG."""
    p = S.pipeline
    v = p.get_view() if p else None
    if not v or v["frame"] is None:
        abort(404)
    ok, buf = cv2.imencode(".jpg", v["frame"], [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


def main():
    ap = argparse.ArgumentParser(description="Веб-панель оператора")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"[INFO] панель оператора: {url}")
    if a.host not in ("127.0.0.1", "localhost"):
        print("[WARN] панель открыта в сеть без пароля: любой, кто видит порт, может "
              "запускать наблюдение и менять настройки. Только доверенная сеть.")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
