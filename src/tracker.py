import cv2 as cv
from pathlib import Path
from ultralytics import YOLO

import config as cfg


class ObjTracker:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path is not None else Path(cfg.MODEL_PATH)
        if not path.is_absolute():
            path = cfg.BASE_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.model = YOLO(str(path))

    def process_frame(self, frame):
        tracker_arg = None
        if cfg.TRACKER_CONFIG is not None and Path(cfg.TRACKER_CONFIG).exists():
            tracker_arg = cfg.TRACKER_CONFIG

        kwargs = {
            "source": frame,
            "persist": True,
            "classes": cfg.TARGET_CLASSES,
            "verbose": False,
            "conf": cfg.CONF_THRESHOLD,
            "imgsz": cfg.IMGSZ,
        }
        if tracker_arg is not None:
            kwargs["tracker"] = tracker_arg

        results = self.model.track(**kwargs)
        persons = {}
        bags = {}

        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return persons, bags

        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = (r.boxes.conf.cpu().numpy() if r.boxes.conf is not None
                 else [0.0] * len(boxes))

        for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            # ступни: в перспективной съёмке расстояние «человек ↔ сумка на полу»
            # правильнее считать от низа бокса, а не от центра (центр — это торс)
            feet = (center[0], y2)
            info = {
                "center": center,
                "feet": feet,
                "bbox": (x1, y1, x2, y2),
                "height": max(1, y2 - y1),
                "conf": float(conf),
            }
            if cls_id == cfg.PERSON_CLASS:
                persons[track_id] = info
            elif cls_id == cfg.BAG_CLASS:
                bags[track_id] = info

        return persons, bags

if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(__file__))

    from tracker import ObjTracker

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tracker = ObjTracker()
    persons, bags = tracker.process_frame(frame)
    print("OK, persons =", len(persons), ", bags =", len(bags))