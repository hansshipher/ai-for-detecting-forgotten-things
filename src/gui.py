#!/usr/bin/env python3
"""Графический интерфейс оператора (Tkinter — входит в стандартный Python).

Возможности:
  * выбор источника: видеофайл, веб-камера, IP-камера (rtsp/http), выбор весов модели;
  * старт / пауза / стоп, ручной снимок кадра;
  * общий статус сцены (НОРМА / ВНИМАНИЕ / ТРЕВОГА), счётчики, FPS, прогресс файла;
  * журнал событий на русском с цветами; клик по строке — показать снимок;
  * звуковой сигнал и мигание при тревоге;
  * настройки «на лету»: задержка тревоги, порог уверенности, эскалация при постороннем;
  * по окончании — HTML-отчёт сессии (кнопка «Отчёт»), папка с журналом и снимками.

Запуск (из папки src/):
    python gui.py
    python gui.py --source ../videos/video1.avi
    python gui.py --sim            # синтетическая сцена, без модели (проверка логики)
"""

from __future__ import annotations

import argparse
import os
import platform
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import config as cfg

STATUS_VIEW = {
    "SAFE": ("НОРМА", "#1a7f37", "#ffffff"),
    "WARNING": ("ВНИМАНИЕ", "#e8a200", "#000000"),
    "ALARM": ("ТРЕВОГА", "#d93025", "#ffffff"),
    "IDLE": ("ОСТАНОВЛЕНО", "#5f6368", "#ffffff"),
    "LOADING": ("ЗАГРУЗКА…", "#1967d2", "#ffffff"),
}
LEVEL_TAG_COLORS = {"alarm": "#fde2e1", "warning": "#fff4d6", "info": "#e3f4e8"}
VIDEO_TYPES = [("Видео", "*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg *.wmv"), ("Все файлы", "*.*")]


def open_path(path: Path):
    """Открыть файл/папку системным приложением."""
    path = str(path)
    try:
        if platform.system() == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось открыть {path}:\n{e}")


def beep():
    try:
        if platform.system() == "Windows":
            import winsound
            threading.Thread(target=lambda: [winsound.Beep(1200, 180) for _ in range(3)],
                             daemon=True).start()
            return True
    except Exception:
        pass
    return False


class App:
    TICK_MS = 33

    def __init__(self, root: tk.Tk, source: str = "", sim: bool = False):
        self.root = root
        self.sim = sim
        self.pipeline = None
        self.logger = None
        self.report_path: Path | None = None
        self.session_dir: Path | None = None
        self._photo = None
        self._preview_photo = None
        self._rows: dict[str, dict] = {}
        self._alarm_total = 0
        self._flash = 0
        self._loading = False
        self._start_q: queue.Queue = queue.Queue()

        root.title("Детектор оставленных вещей")
        root.geometry("1360x860")
        root.minsize(1100, 700)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=24)
        style.configure("Big.TButton", padding=(10, 4))

        self.var_source = tk.StringVar(value=source or ("simulation" if sim else str(cfg.VIDEO_PATH)))
        self.var_model = tk.StringVar(value=str(cfg.MODEL_PATH))
        self.var_camera = tk.BooleanVar(value=False)
        self.var_delay = tk.DoubleVar(value=cfg.ALARM_DELAY_SEC)
        self.var_conf = tk.DoubleVar(value=cfg.CONF_THRESHOLD)
        self.var_escalate = tk.BooleanVar(value=cfg.BYSTANDER_ESCALATE)
        self.var_sound = tk.BooleanVar(value=True)
        self.var_realtime = tk.BooleanVar(value=True)
        self.var_record = tk.BooleanVar(value=False)
        self.var_snapshots = tk.BooleanVar(value=True)
        self.var_show_debug = tk.BooleanVar(value=False)
        self.var_statusbar = tk.StringVar(value="Выберите источник и нажмите «Старт».")

        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        self._set_status("IDLE")
        self._update_buttons()
        self.root.after(self.TICK_MS, self._tick)

    # ================================================================ UI

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar, text="Источник:").grid(row=0, column=0, sticky="w")
        self.ent_source = ttk.Entry(bar, textvariable=self.var_source, width=48)
        self.ent_source.grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(bar, text="Файл…", command=self.choose_file).grid(row=0, column=2, padx=2)
        ttk.Button(bar, text="Веб-камера", command=self.choose_webcam).grid(row=0, column=3, padx=2)
        ttk.Checkbutton(bar, text="Live-камера", variable=self.var_camera).grid(row=0, column=4, padx=6)

        ttk.Label(bar, text="Модель:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(bar, textvariable=self.var_model, width=48).grid(row=1, column=1, sticky="we",
                                                                    padx=4, pady=(4, 0))
        ttk.Button(bar, text="Веса…", command=self.choose_model).grid(row=1, column=2, padx=2, pady=(4, 0))

        btns = ttk.Frame(bar)
        btns.grid(row=0, column=5, rowspan=2, padx=(16, 0), sticky="e")
        self.btn_start = ttk.Button(btns, text="▶ Старт", style="Big.TButton", command=self.start)
        self.btn_pause = ttk.Button(btns, text="Пауза", style="Big.TButton", command=self.toggle_pause)
        self.btn_stop = ttk.Button(btns, text="■ Стоп", style="Big.TButton", command=self.stop)
        self.btn_snap = ttk.Button(btns, text="Снимок", style="Big.TButton", command=self.snapshot)
        self.btn_folder = ttk.Button(btns, text="Папка журнала", style="Big.TButton", command=self.open_folder)
        self.btn_report = ttk.Button(btns, text="Отчёт", style="Big.TButton", command=self.open_report)
        for i, b in enumerate((self.btn_start, self.btn_pause, self.btn_stop, self.btn_snap,
                               self.btn_folder, self.btn_report)):
            b.grid(row=0, column=i, padx=2)
        bar.columnconfigure(1, weight=1)

    def _build_body(self):
        body = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        top = ttk.Frame(body)
        body.add(top, weight=4)

        # --- видео
        # Canvas, а не Label: Label подстраивает размер под картинку, и окно
        # начинает «расти» на каждом кадре
        self.video = tk.Canvas(top, bg="#111111", highlightthickness=0, bd=0)
        self.video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._video_img_id = None
        self._video_text_id = self.video.create_text(
            10, 10, anchor="nw", text="Здесь будет видео", fill="#bbbbbb", font=("Segoe UI", 14))

        # --- правая панель
        side = ttk.Frame(top, width=330, padding=(8, 0, 0, 0))
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        self.banner = tk.Label(side, text="", font=("Segoe UI", 22, "bold"), height=2)
        self.banner.pack(fill=tk.X)

        stats = ttk.LabelFrame(side, text="Сцена", padding=8)
        stats.pack(fill=tk.X, pady=(8, 0))
        self.stat_vars = {}
        for i, (key, title) in enumerate([
                ("persons", "Людей в кадре"), ("bags", "Сумок в кадре"),
                ("warnings", "Под наблюдением (WARNING)"), ("alarms_now", "Тревог сейчас"),
                ("alarm_total", "Тревог за сессию"), ("fps", "Скорость обработки"),
                ("vtime", "Время видео")]):
            ttk.Label(stats, text=title).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value="—")
            ttk.Label(stats, textvariable=v, font=("Segoe UI", 10, "bold")).grid(row=i, column=1, sticky="e")
            self.stat_vars[key] = v
        stats.columnconfigure(1, weight=1)
        self.progress = ttk.Progressbar(stats, mode="determinate", maximum=1.0)
        self.progress.grid(row=10, column=0, columnspan=2, sticky="we", pady=(6, 0))

        sett = ttk.LabelFrame(side, text="Настройки", padding=8)
        sett.pack(fill=tk.X, pady=(8, 0))
        self.lbl_delay = ttk.Label(sett)
        self.lbl_delay.grid(row=0, column=0, sticky="w")
        ttk.Scale(sett, from_=3, to=60, variable=self.var_delay,
                  command=lambda _=None: self._apply_settings()).grid(row=1, column=0, sticky="we")
        self.lbl_conf = ttk.Label(sett)
        self.lbl_conf.grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(sett, from_=0.10, to=0.90, variable=self.var_conf,
                  command=lambda _=None: self._apply_settings()).grid(row=3, column=0, sticky="we")
        ttk.Checkbutton(sett, text="Тревога даже если рядом посторонний", variable=self.var_escalate,
                        command=self._apply_settings).grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(sett, text="Звук при тревоге", variable=self.var_sound).grid(row=5, column=0, sticky="w")
        ttk.Checkbutton(sett, text="Сохранять снимки событий", variable=self.var_snapshots).grid(
            row=6, column=0, sticky="w")
        ttk.Checkbutton(sett, text="Записывать видео с разметкой", variable=self.var_record).grid(
            row=7, column=0, sticky="w")
        ttk.Checkbutton(sett, text="Реальное время (пропуск кадров)", variable=self.var_realtime).grid(
            row=8, column=0, sticky="w")
        sett.columnconfigure(0, weight=1)
        self._apply_settings()

        prev = ttk.LabelFrame(side, text="Снимок события", padding=6)
        prev.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.preview = tk.Label(prev, text="Выберите событие в журнале", fg="#777")
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.preview.bind("<Double-Button-1>", lambda e: self._open_selected_frame())

        # --- журнал
        bottom = ttk.Frame(body)
        body.add(bottom, weight=1)
        head = ttk.Frame(bottom)
        head.pack(fill=tk.X)
        ttk.Label(head, text="Журнал событий", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Checkbutton(head, text="служебные события", variable=self.var_show_debug).pack(side=tk.RIGHT)

        cols = ("time", "vtime", "event", "bag", "status", "reason", "elapsed")
        titles = ("Время", "Видео", "Событие", "Сумка", "Статус", "Причина", "Без присмотра, с")
        widths = (80, 70, 260, 60, 90, 180, 120)
        self.tree = ttk.Treeview(bottom, columns=cols, show="headings", height=7)
        for c, t, w in zip(cols, titles, widths):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w", stretch=(c == "event"))
        for lvl, color in LEVEL_TAG_COLORS.items():
            self.tree.tag_configure(lvl, background=color)
        sb = ttk.Scrollbar(bottom, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._show_selected())
        self.tree.bind("<Double-1>", lambda e: self._open_selected_frame())

    def _build_statusbar(self):
        ttk.Label(self.root, textvariable=self.var_statusbar, anchor="w",
                  relief=tk.SUNKEN, padding=(8, 2)).pack(side=tk.BOTTOM, fill=tk.X)

    # =========================================================== действия

    def choose_file(self):
        p = filedialog.askopenfilename(title="Видеофайл", filetypes=VIDEO_TYPES,
                                       initialdir=str(cfg.BASE_DIR / "videos"))
        if p:
            self.var_source.set(p)
            self.var_camera.set(False)

    def choose_webcam(self):
        self.var_source.set("0")
        self.var_camera.set(True)

    def choose_model(self):
        p = filedialog.askopenfilename(title="Веса YOLO", filetypes=[("PyTorch", "*.pt"), ("Все", "*.*")],
                                       initialdir=str(cfg.BASE_DIR / "weights"))
        if p:
            self.var_model.set(p)

    def _apply_settings(self):
        cfg.ALARM_DELAY_SEC = round(float(self.var_delay.get()))
        cfg.CONF_THRESHOLD = round(float(self.var_conf.get()), 2)
        cfg.BYSTANDER_ESCALATE = bool(self.var_escalate.get())
        self.lbl_delay.configure(text=f"Задержка тревоги: {cfg.ALARM_DELAY_SEC:.0f} с")
        self.lbl_conf.configure(text=f"Порог уверенности детектора: {cfg.CONF_THRESHOLD:.2f}")

    def start(self):
        if self.pipeline is not None or self._loading:
            return
        source = self.var_source.get().strip()
        if not source:
            messagebox.showwarning("Источник", "Укажите видеофайл или камеру.")
            return
        is_cam = bool(self.var_camera.get()) or source.isdigit() or \
            source.lower().startswith(("rtsp://", "http://", "https://"))
        if not self.sim and not is_cam and not Path(source).exists():
            messagebox.showerror("Источник", f"Файл не найден:\n{source}")
            return
        model = self.var_model.get().strip()
        if not self.sim and not Path(model).exists():
            messagebox.showerror("Модель", f"Файл весов не найден:\n{model}\n\n"
                                 "Скачайте best12k.pt по ссылке из readme.md в папку weights/.")
            return

        self._clear_session_ui()
        self._loading = True
        self._set_status("LOADING")
        self.var_statusbar.set("Загрузка модели и открытие источника…")
        self._update_buttons()
        opts = dict(source=source, camera=is_cam, model=model,
                    realtime=bool(self.var_realtime.get()) or is_cam,
                    record=bool(self.var_record.get()), snapshots=bool(self.var_snapshots.get()))
        # модель грузится несколько секунд — не замораживаем окно
        threading.Thread(target=self._create_pipeline, args=(opts,), daemon=True).start()

    def _create_pipeline(self, o):
        logger = p = None
        try:
            from event_logger import EventLogger
            from pipeline import Pipeline
            if self.sim:
                from tools.sim_scene import SimCapture, SimTracker
                tracker, capture, model_name = SimTracker(), SimCapture(), "SimTracker (симуляция)"
            else:
                tracker, capture, model_name = None, None, Path(o["model"]).name
            logger = EventLogger(source=o["source"], model=model_name, save_snapshots=o["snapshots"])
            record = (logger.dir / "annotated.mp4") if o["record"] else None
            p = Pipeline(o["source"], camera=o["camera"], tracker=tracker, model_path=o["model"],
                         logger=logger, record_path=record, realtime=o["realtime"],
                         hud=False, capture=capture, verbose=True)
            p.start()
            self._start_q.put((p, logger, None))
        except Exception as e:
            # модель/источник не открылись: не оставляем пустую сессию и открытые файлы
            if p is not None:
                try:
                    p.stop(build_report=False)
                except Exception:
                    pass
            if logger is not None:
                logger.abort()
            self._start_q.put((None, None, f"{type(e).__name__}: {e}"))

    def toggle_pause(self):
        if self.pipeline is None:
            return
        on = self.pipeline.pause()
        self.btn_pause.configure(text="Продолжить" if on else "Пауза")
        self.var_statusbar.set("Пауза." if on else "Обработка…")

    def stop(self):
        if self.pipeline is None:
            return
        p, self.pipeline = self.pipeline, None
        self.var_statusbar.set("Остановка, формирование отчёта…")
        self.root.update_idletasks()
        try:
            self.report_path = p.stop()
        except Exception as e:
            messagebox.showerror("Ошибка при остановке", str(e))
        self._drain_events(p)
        self._set_status("IDLE")
        self.btn_pause.configure(text="Пауза")
        self._update_buttons()
        msg = f"Сессия сохранена: {self.session_dir}"
        if self.report_path:
            msg += "  ·  нажмите «Отчёт», чтобы открыть HTML-отчёт"
        self.var_statusbar.set(msg)

    def snapshot(self):
        if self.pipeline is not None:
            self.pipeline.request_snapshot()
            self.var_statusbar.set("Снимок сохранён в журнал.")

    def open_folder(self):
        target = self.session_dir or (cfg.BASE_DIR / "runs" / "sessions")
        Path(target).mkdir(parents=True, exist_ok=True)
        open_path(target)

    def open_report(self):
        if self.report_path and Path(self.report_path).exists():
            webbrowser.open(Path(self.report_path).resolve().as_uri())
        elif self.pipeline is not None:
            messagebox.showinfo("Отчёт", "Отчёт формируется после нажатия «Стоп» или окончания видео.")
        else:
            messagebox.showinfo("Отчёт", "Отчёта пока нет — запустите обработку.")

    def on_close(self):
        if self.pipeline is not None:
            self.stop()
        self.root.destroy()

    # ============================================================ цикл UI

    def _tick(self):
        try:
            self._poll_start()
            p = self.pipeline
            if p is not None:
                view = p.get_view()
                if view["frame"] is not None:
                    self._show_frame(view["frame"])
                self._update_stats(view)
                self._drain_events(p)
                if view["error"]:
                    self.stop()
                    messagebox.showerror("Ошибка обработки", view["error"])
                elif view["finished"]:
                    self.stop()
                    self.var_statusbar.set(self.var_statusbar.get().replace(
                        "Сессия сохранена", "Видео закончилось. Сессия сохранена"))
            self._animate_alarm()
        finally:
            self.root.after(self.TICK_MS, self._tick)

    def _poll_start(self):
        try:
            p, logger, err = self._start_q.get_nowait()
        except queue.Empty:
            return
        self._loading = False
        if err:
            self._set_status("IDLE")
            self._update_buttons()
            self.var_statusbar.set("Не удалось запустить.")
            messagebox.showerror("Не удалось запустить", err)
            return
        self.pipeline, self.logger = p, logger
        self.session_dir = logger.dir
        self._set_status("SAFE")
        self._update_buttons()
        self.var_statusbar.set(f"Обработка… Журнал пишется в {logger.dir}")

    def _show_frame(self, bgr):
        w = max(self.video.winfo_width(), 64)
        h = max(self.video.winfo_height(), 64)
        fh, fw = bgr.shape[:2]
        k = min(w / fw, h / fh)
        size = (max(1, int(fw * k)), max(1, int(fh * k)))
        img = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        if self._video_text_id is not None:
            self.video.delete(self._video_text_id)
            self._video_text_id = None
        if self._video_img_id is None:
            self._video_img_id = self.video.create_image(w // 2, h // 2, image=self._photo, anchor="center")
        else:
            self.video.coords(self._video_img_id, w // 2, h // 2)
            self.video.itemconfigure(self._video_img_id, image=self._photo)

    def _update_stats(self, v):
        sv = self.stat_vars
        sv["persons"].set(str(v["persons"]))
        sv["bags"].set(str(v["bags"]))
        sv["warnings"].set(str(v["warnings"]))
        sv["alarms_now"].set(str(v["alarms_now"]))
        sv["alarm_total"].set(str(self._alarm_total))
        sv["fps"].set(f"{v['infer_fps']:.1f} к/с · {v['infer_ms']:.0f} мс")
        if self.pipeline is not None and self.pipeline.camera:
            sv["vtime"].set("live")
        else:
            m, s = divmod(v["vtime"], 60)
            sv["vtime"].set(f"{int(m):02d}:{s:04.1f}")
        self.progress["value"] = v["progress"]
        if not (self.pipeline and self.pipeline.paused):
            self._set_status(v["status"])

    def _drain_events(self, p):
        while True:
            try:
                row = p.events.get_nowait()
            except queue.Empty:
                break
            level = row.get("level", "debug")
            if row.get("type") == "alarm":
                self._alarm_total += 1
                self._flash = 12
                if self.var_sound.get() and not beep():
                    self.root.bell()
            if level == "debug" and not self.var_show_debug.get():
                continue
            vt = float(row.get("video_time") or 0.0)
            vt_s = "live" if (self.pipeline and self.pipeline.camera) else f"{int(vt // 60):02d}:{vt % 60:04.1f}"
            bag = row.get("bag_id")
            iid = self.tree.insert("", 0, values=(
                row.get("wall_time", ""), vt_s, row.get("title", row.get("type")),
                bag if (bag is not None and bag >= 0) else "—",
                row.get("status", ""), row.get("reason", ""),
                f"{float(row.get('elapsed_sec') or row.get('elapsed') or 0):.1f}"),
                tags=(level,))
            self._rows[iid] = row
            if row.get("type") == "alarm":
                self.tree.selection_set(iid)

    def _animate_alarm(self):
        if self._flash > 0:
            self._flash -= 1
            on = self._flash % 2 == 0
            self.banner.configure(bg="#d93025" if on else "#ffffff",
                                  fg="#ffffff" if on else "#d93025")

    # ========================================================== helpers

    def _set_status(self, key):
        if self._flash > 0 and key != "ALARM":
            return
        text, bg, fg = STATUS_VIEW[key]
        self.banner.configure(text=text, bg=bg, fg=fg)

    def _update_buttons(self):
        running = self.pipeline is not None
        busy = running or self._loading
        self.btn_start.state(["disabled"] if busy else ["!disabled"])
        for b in (self.btn_pause, self.btn_stop, self.btn_snap):
            b.state(["!disabled"] if running else ["disabled"])
        self.ent_source.state(["disabled"] if busy else ["!disabled"])

    def _clear_session_ui(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._rows.clear()
        self._alarm_total = 0
        self.report_path = None
        self.progress["value"] = 0
        self.preview.configure(image="", text="Выберите событие в журнале")
        self._preview_photo = None

    def _selected_row(self):
        sel = self.tree.selection()
        return self._rows.get(sel[0]) if sel else None

    def _show_selected(self):
        row = self._selected_row()
        if not row or self.session_dir is None:
            return
        rel = row.get("crop_path") or row.get("frame_path")
        if not rel:
            self.preview.configure(image="", text="Для этого события снимка нет")
            self._preview_photo = None
            return
        path = Path(self.session_dir) / rel
        try:
            img = Image.open(path)
            img.thumbnail((310, 240))
            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self._preview_photo, text="")
        except Exception as e:
            self.preview.configure(image="", text=f"Не удалось открыть снимок: {e}")

    def _open_selected_frame(self):
        row = self._selected_row()
        if row and self.session_dir and row.get("frame_path"):
            open_path(Path(self.session_dir) / row["frame_path"])


def main():
    ap = argparse.ArgumentParser(description="GUI детектора оставленных вещей")
    ap.add_argument("--source", default="", help="видеофайл, номер веб-камеры или rtsp://")
    ap.add_argument("--sim", action="store_true", help="синтетическая сцена без модели")
    args = ap.parse_args()

    root = tk.Tk()
    App(root, source=args.source, sim=args.sim)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
