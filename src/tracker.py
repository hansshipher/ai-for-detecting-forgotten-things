import cv2 as cv
from pathlib import Path
from ultralytics import YOLO

import config as cfg


class ObjTracker:
    def __init__(self):
        if not Path(cfg.MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found: {cfg.MODEL_PATH}")
        self.model = YOLO(str(cfg.MODEL_PATH))

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

        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if cls_id == 0:
                persons[track_id] = center
            elif cls_id in cfg.BAG_CLASSES:
                bags[track_id] = {"center": center, "bbox": (x1, y1, x2, y2)}

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