"""Настройки проекта: модель, классы, пороги, таймеры, цвета."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Демо-модель (12k, yolo26n @640, 65 эпох). Ссылка и sha256 — в readme.md.
MODEL_NAME = "best12k.pt"
MODEL_PATH = BASE_DIR / "weights" / MODEL_NAME

VIDEO_PATH = BASE_DIR / "videos" / "video1.avi"

TRACKER_CONFIG = str(BASE_DIR / "configs" / "custom_tracker.yaml")
CONF_THRESHOLD = 0.25
IMGSZ = 640

TARGET_CLASSES = [0, 1]
BAG_CLASS = 0
PERSON_CLASS = 1

# --- логика забытых вещей (всё в секундах, не в кадрах) ---

# Сумка без владельца дольше этого — ALARM.
ALARM_DELAY_SEC = 15.0

# True — тревога даже если рядом посторонний (по умолчанию WARNING).
BYSTANDER_ESCALATE = False

# «Рядом» = OWNER_RADIUS_K диагоналей бокса, в пределах
# [OWNER_RADIUS_MIN_PX, OWNER_RADIUS_MAX_FRAC × высоты кадра].
OWNER_RADIUS_K = 2.5
OWNER_RADIUS_MIN_PX = 50
OWNER_RADIUS_MAX_FRAC = 0.25

# Гистерезис: из SAFE выходим на 1.3 радиуса, возвращаемся на 1.0.
OWNER_LEAVE_K = 1.3

# «Стоит»: центр сместился меньше STATIONARY_SHIFT_K диагоналей за
# STATIONARY_WINDOW_SEC (но не меньше STATIONARY_SHIFT_MIN_PX).
STATIONARY_WINDOW_SEC = 1.5
STATIONARY_SHIFT_K = 0.6
STATIONARY_SHIFT_MIN_PX = 8

# Пропуск детекций дольше этого обнуляет историю движения.
STATIONARY_GAP_SEC = 0.7

# Тревога только за сумку, которую принесли (она двигалась).
REQUIRE_MOVED = True

# Больше этой доли кадра — не сумка (дверной проём, стена).
# В датасете максимум 11.5%.
MAX_BAG_AREA_FRAC = 0.15

# Посторонний должен простоять рядом столько — проходящий мимо не считается.
BYSTANDER_CONFIRM_SEC = 1.0

# Пока окно движения не набралось, держим SAFE вместо WARNING.
MOTION_GRACE_SEC = 3.0

# Несдвинувшийся объект перестаём рисовать через столько секунд.
STATIC_HIDE_SEC = 4.0

# Сколько помним объект без детекции; сумку — дольше, детекции идут с перерывами.
PERSON_MISSING_SEC = 2.0
BAG_MISSING_SEC = 6.0

# Склейка людей при смене track_id: радиус = PERSON_LINK_K роста человека в боксе.
PERSON_LINK_K = 0.75

# Склейка состояний при смене track_id (ByteTrack теряет ID при пропуске кадров).
BAG_RELINK_SEC = 5.0
BAG_RELINK_IOU = 0.3
BAG_DEDUP_IOU = 0.6

# Состояние, пропавшее дольше этого, не рисуем (иначе «ghost-боксы»).
GHOST_GRACE_SEC = 0.3

# Ограничение шага времени: защита от скачка после зависания/перемотки.
MAX_DT_SEC = 1.0

# BGR
COLOR_SAFE = (0, 255, 0)
COLOR_WARNING = (0, 255, 255)
COLOR_ALARM = (0, 0, 255)

COLOR_PERSON = (255, 180, 0)
COLOR_TEXT = (255, 255, 255)