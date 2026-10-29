from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "yolo26m.pt"
MODEL_PATH = BASE_DIR / "weights" / MODEL_NAME

VIDEO_PATH = BASE_DIR / "videos" / "video1.avi"

TRACKER_CONFIG = str(BASE_DIR / "configs" / "custom_tracker.yaml")
CONF_THRESHOLD = 0.1
IMGSZ = 1280

TARGET_CLASSES = [0, 1]
BAG_CLASS = 0
PERSON_CLASS = 1

SAFE_RADIUS_PX = 150
ALARM_DELAY_SEC = 5.0

# BGR
COLOR_SAFE = (0, 255, 0)
COLOR_WARNING = (0, 255, 255)
COLOR_ALARM = (0, 0, 255)

COLOR_PERSON = (255, 180, 0)
COLOR_TEXT = (255, 255, 255)