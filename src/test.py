import argparse
import queue
import threading
import time

import cv2

import config as cfg
from tracker import ObjTracker
from logic import AbObjDetector
from visualizer import Visualizer


def parse_args():
    p = argparse.ArgumentParser()
    # по умолчанию — тестовое видео из конфига; для камеры: --source rtsp://...
    p.add_argument("--source", default=str(cfg.VIDEO_PATH),
                   help="путь к видеофайлу или URL камеры (rtsp://, http://, 0 для веб-камеры)")
    p.add_argument("--camera", action="store_true",
                   help="режим live-потока: таймер по реальному времени, буфер декодера = 1")
    return p.parse_args()


def main():
    args = parse_args()
    is_camera = args.camera

    tracker = ObjTracker()
    detector = AbObjDetector()
    visualizer = Visualizer()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] cannot open source: {args.source}")
        return

    # для live не копить кадры в декодере — иначе задержка прячется не в нашем цикле
    if is_camera:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:          # у RTSP fps часто приходит 0/мусор
        fps = 25.0
    frame_delay = 1.0 / fps

    print(f"[INFO] source={args.source} | camera={is_camera} | fps={fps:.1f}")

    # drop-old: worker всегда жуёт самый свежий кадр, старые затираются.
    # это и есть real-time паттерн — задержка ограничена одним прогоном модели,
    # а не растёт бесконечно, как при однопоточном отставании от камеры.
    frame_q = queue.Queue(maxsize=1)
    lock = threading.Lock()
    shared = {"persons": {}, "tracked_bags": {}}
    stop = threading.Event()

    def infer_worker():
        while not stop.is_set():
            try:
                frame, vtime = frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                persons, bags = tracker.process_frame(frame)
                tracked_bags = detector.update(persons, bags, vtime)
                with lock:
                    shared["persons"] = persons
                    shared["tracked_bags"] = tracked_bags
            except Exception as e:
                # исключения в потоке молча убивают worker — печатаем, иначе не поймёшь почему рамки пропали
                print(f"[WORKER ERROR] {type(e).__name__}: {e}")

    th = threading.Thread(target=infer_worker, daemon=True)
    th.start()

    fps_real = 0.0
    t_fps = time.time()
    n_fps = 0
    frame_idx = 0
    start_wall = time.time()

    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            if is_camera:
                time.sleep(0.01)     # камера на секунду отвалилась — не выходим, ждём
                continue
            break                    # файл закончился — выходим

        frame_idx += 1

        # ИСТОЧНИК ВРЕМЕНИ ДЛЯ ТАЙМЕРА. Файл и камера считаются по-разному, см. разбор ниже.
        if is_camera:
            vtime = time.time()              # wall-clock: «5 сек реального времени без хозяина»
        else:
            vtime = frame_idx / fps          # время внутри записи: детерминированно, не плывёт от FPS показа

        # кладём кадр в инференс, затирая предыдущий если worker не успел
        try:
            frame_q.put_nowait((frame, vtime))
        except queue.Full:
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
            frame_q.put_nowait((frame, vtime))

        with lock:
            persons = shared["persons"]
            tracked_bags = shared["tracked_bags"]

        out = visualizer.draw_status(frame, persons, tracked_bags)

        n_fps += 1
        now = time.time()
        if now - t_fps >= 1.0:
            fps_real = n_fps / (now - t_fps)
            n_fps = 0
            t_fps = now
        mode = "CAM" if is_camera else "FILE"
        cv2.putText(out, f"{mode} | FPS: {fps_real:.1f} | t={vtime if not is_camera else 0:.1f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Prototype", out)

        # throttle показа до fps источника — плавная картинка даже если инференс отстаёт
        target = start_wall + frame_idx * frame_delay
        sleep_s = target - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop.set()
    th.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()