from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Рабочая модель: yolo26n @640, обучена без dc_luggage (65 эпох, 30.10.2026).
# Файл лежит в weights/ и не хранится в git — ссылка и sha256 в readme.md.
MODEL_NAME = "best26n_nodc_640.pt"
MODEL_PATH = BASE_DIR / "weights" / MODEL_NAME

VIDEO_PATH = BASE_DIR / "videos" / "video1.avi"

TRACKER_CONFIG = str(BASE_DIR / "configs" / "custom_tracker.yaml")
CONF_THRESHOLD = 0.25
IMGSZ = 640

TARGET_CLASSES = [0, 1]
BAG_CLASS = 0
PERSON_CLASS = 1

# --- логика забытых вещей (всё в секундах, не в кадрах) ---

# Сколько секунд сумка стоит без владельца, прежде чем дать ALARM.
# 15 с — компромисс для демо: человек успевает отойти, но тревога не «нервная».
ALARM_DELAY_SEC = 15.0

# Посторонний рядом блокирует тревогу: WARNING, решение за оператором.
# True — эскалировать в ALARM даже если кто-то стоит рядом.
BYSTANDER_ESCALATE = False

# «Рядом с сумкой» = OWNER_RADIUS_K диагоналей её бокса,
# но не меньше OWNER_RADIUS_MIN_PX и не больше OWNER_RADIUS_MAX_FRAC высоты кадра.
OWNER_RADIUS_K = 2.5
OWNER_RADIUS_MIN_PX = 50
OWNER_RADIUS_MAX_FRAC = 0.25

# Гистерезис: из SAFE выходим на 1.3 радиуса, входим на 1.0 — чтобы статус не мигал.
OWNER_LEAVE_K = 1.3

# Сумка «стоит», если за окно STATIONARY_WINDOW_SEC её центр сместился меньше чем на
# STATIONARY_SHIFT_K диагоналей (но не меньше STATIONARY_SHIFT_MIN_PX).
STATIONARY_WINDOW_SEC = 1.5
STATIONARY_SHIFT_K = 0.6
STATIONARY_SHIFT_MIN_PX = 8

# Если сумку не видели дольше этого, историю движения обнуляем: иначе «несут»
# и «стоит» склеиваются через пропуск детекций.
STATIONARY_GAP_SEC = 0.7

# Тревожиться можно только за сумку, которую ПРИНЕСЛИ (она двигалась).
# Статичное ложное срабатывание на двери/текстуре не должно давать ALARM.
REQUIRE_MOVED = True

# Сумка не может занимать больше этой доли кадра. Дверной проём, стена или
# кусок пола, которые модель иногда принимает за сумку, отсекаются здесь.
# В датасете максимум 11.5%, так что 15% — с запасом и без риска для реальных сумок.
MAX_BAG_AREA_FRAC = 0.15

# Посторонний должен простоять рядом BYSTANDER_CONFIRM_SEC — проходящий мимо не считается.
BYSTANDER_CONFIRM_SEC = 1.0

# В первые секунды после появления сумки мы ещё не знаем, несут её или она
# стоит: окно движения не набралось. Столько секунд держим SAFE, не мигая WARNING.
MOTION_GRACE_SEC = 3.0

# Объект, который так и не сдвинулся с места, не может быть забытой вещью
# (см. REQUIRE_MOVED). Через столько секунд перестаём рисовать его как сумку,
# чтобы статичное ложное срабатывание (дверной проём) не висело на экране.
STATIC_HIDE_SEC = 4.0

# Сколько помним объект без детекции (в секундах источника).
PERSON_MISSING_SEC = 2.0
# Сумку помним дольше: детектор видит лежащую сумку с перерывами, и таймер
# не должен обнуляться из-за пропуска в 3–4 секунды.
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