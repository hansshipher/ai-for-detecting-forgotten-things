"""Консольный запуск без GUI: окно OpenCV + журнал событий со снимками.

    python test.py                                  # видео из config.VIDEO_PATH
    python test.py --source ../videos/video2.avi
    python test.py --camera --source 0              # веб-камера
    python test.py --camera --source rtsp://...     # IP-камера
    python test.py --record                         # + видео с разметкой в папку сессии
    python test.py --no-log                         # без журнала (как раньше)

Окно закрывается клавишей q, пробел — пауза, s — ручной снимок.
Журнал и отчёт: runs/sessions/<дата_время>/ (report.html открывается в браузере).

Вся обработка — в pipeline.Pipeline (тот же код использует gui.py).
"""

import argparse
import time

import cv2

import config as cfg
from event_logger import EventLogger
from pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser()
    # по умолчанию — тестовое видео из конфига; для камеры: --source rtsp://...
    p.add_argument("--source", default=str(cfg.VIDEO_PATH),
                   help="путь к видеофайлу или URL камеры (rtsp://, http://, 0 для веб-камеры)")
    p.add_argument("--camera", action="store_true",
                   help="режим live-потока: таймер по реальному времени, буфер декодера = 1")
    p.add_argument("--model", default=None, help="путь к весам (по умолчанию из config.py)")
    p.add_argument("--no-log", action="store_true", help="не писать журнал и снимки")
    p.add_argument("--record", action="store_true", help="сохранить видео с разметкой")
    p.add_argument("--every-frame", action="store_true",
                   help="обработать каждый кадр файла (медленнее реального времени, без пропусков)")
    return p.parse_args()


def main():
    args = parse_args()
    model = args.model or str(cfg.MODEL_PATH)

    logger = None if args.no_log else EventLogger(source=args.source, model=model)
    record = (logger.dir / "annotated.mp4") if (logger and args.record) else None

    try:
        pipe = Pipeline(args.source, camera=args.camera, model_path=args.model,
                        logger=logger, record_path=record,
                        realtime=not args.every_frame or args.camera)
    except Exception as e:
        print(f"[ERROR] {e}")
        if logger:
            logger.abort()
        return

    print(f"[INFO] source={args.source} | camera={args.camera} | fps={pipe.fps:.1f}")
    if logger:
        print(f"[INFO] журнал: {logger.dir}")
    pipe.start()

    try:
        while True:
            view = pipe.get_view()
            if view["frame"] is not None:
                cv2.imshow("Forgotten things detector", view["frame"])
            key = cv2.waitKey(15) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                print("[INFO] пауза" if pipe.pause() else "[INFO] продолжение")
            if key == ord("s"):
                pipe.request_snapshot()
            if view["error"]:
                print(f"[ERROR] {view['error']}")
                break
            if view["finished"]:
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        report = pipe.stop()
        cv2.destroyAllWindows()

    if logger:
        s = logger.summary()
        print(f"[DONE] тревог: {s['alarms']} | предупреждений: {s['warnings']} | "
              f"событий: {s['events_total']}")
        print(f"[DONE] отчёт: {report}")


if __name__ == "__main__":
    main()
