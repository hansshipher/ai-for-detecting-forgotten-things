import config as cfg
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Tuple, Any

class ObjTracker:
    def __init__(self):
        if not Path(cfg.MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found: {cfg.MODEL_PATH}")

        self.model = YOLO(cfg.MODEL_PATH)

    def process_frame(self, frame: cv.Mat) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Dict[str, Any]]]:
        tracker_arg = cfg.TRACKER_CONFIG if Path(cfg.TRACKER_CONFIG).exists() else None

        results = self.model.track(
            source=frame,
            persist=True,
            classes=cfg.TARGET_CLASSES,
            tracker=tracker_arg,
            verbose=False,
            conf=0.5
        )

        persons: Dict[int, Tuple[int, int]] = {}
        bags: Dict[int, Dict[str, Any]]

        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return persons, bags

        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy.astype(int)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            if cls_id == 0:
                persons[track_id] = center
            elif cls_id in cfg.BAG_CLASSES:
                bags[track_id] = {
                    "center": center,
                    "bbox": (x1, y1, x2, y2)
                }
        
        return persons, bags