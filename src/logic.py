"""Логика «забытых вещей»: решение по времени и состоянию, а не по кадрам.

Правило тревоги (все три условия одновременно):

    ALARM = сумка СТОИТ  И  владельца рядом нет  И  рядом никто не стоит
            И  так продолжается дольше ALARM_DELAY_SEC секунд.

Почему так:

  - «стоит» проверяется по смещению центра за окно STATIONARY_WINDOW_SEC.
    Пока сумку несут — она движется, значит тревоги быть не может, даже если
    трекер потерял человека (раньше это давало ложный ALARM на идущих);
  - владелец — это человек, который был рядом в момент появления/переноски
    сумки. Радиус ограничен (раньше владельцем становился кто угодно,
    в том числе человек в 500 px);
  - посторонний рядом НЕ переводит в SAFE: он может не видеть сумку.
    Это WARNING с причиной 'bystander', тревога заморожена, решает оператор.

Всё время — секунды источника (`video_time`), шаг `dt` ограничен MAX_DT_SEC.
Кадры больше нигде не считаются.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import config as cfg


@dataclass
class Event:
    """Событие для логгера/GUI. Пишется только на переходах, не каждый кадр."""

    type: str
    time: float
    bag_id: int
    status: str = ""
    reason: str = ""
    owner_lid: int | None = None
    person_lid: int | None = None
    elapsed: float = 0.0
    distance_px: float | None = None
    bbox: tuple | None = None

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "time": round(self.time, 3),
            "bag_id": self.bag_id,
            "status": self.status,
            "reason": self.reason,
            "owner_lid": self.owner_lid,
            "person_lid": self.person_lid,
            "elapsed": round(self.elapsed, 3),
            "distance_px": None if self.distance_px is None else round(self.distance_px, 1),
            "bbox": self.bbox,
        }


def _box_gap(a, b) -> float:
    """Расстояние между прямоугольниками (0, если пересекаются).

    Меряем именно зазор между боксами, а не «ступни ↔ центр»: сумка в руке
    висит выше ступней, и старое расстояние давало 160 px там, где человек
    буквально держит её.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, bx1 - ax2, ax1 - bx2)
    dy = max(0, by1 - ay2, ay1 - by2)
    return math.hypot(dx, dy)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class AbObjDetector:
    """Отслеживает сумки и решает, когда они «забыты».

    `update()` возвращает словарь состояний (как раньше), а события копит
    внутри — забирать через `drain_events()`. Так логгер и GUI подключаются
    позже, без правок в логике.
    """

    def __init__(self):
        self.tracked_bags: dict[int, dict] = {}
        self.persons_state: dict[int, dict] = {}
        self.events: list[Event] = []

        self._next_lid = 0
        self._last_time: float | None = None
        self._frame_h = 480
        self._frame_area = 480 * 640
        self._rejected_ids: set[int] = set()

    # ------------------------------------------------------------------ события

    def _emit(self, etype: str, state: dict | None = None, **kw) -> Event:
        ev = Event(
            type=etype,
            time=kw.pop("time", self._last_time or 0.0),
            bag_id=kw.pop("bag_id", state["track_id"] if state else -1),
            status=kw.pop("status", state["status"] if state else ""),
            reason=kw.pop("reason", state.get("reason", "") if state else ""),
            owner_lid=kw.pop("owner_lid", state.get("owner_lid") if state else None),
            person_lid=kw.pop("person_lid", None),
            elapsed=kw.pop("elapsed", state.get("elapsed", 0.0) if state else 0.0),
            distance_px=kw.pop("distance_px", None),
            bbox=kw.pop("bbox", state.get("bbox") if state else None),
        )
        self.events.append(ev)
        return ev

    def drain_events(self) -> list[Event]:
        out, self.events = self.events, []
        return out

    # -------------------------------------------------------------------- время

    def _dt(self, video_time: float) -> float:
        if self._last_time is None:
            self._last_time = video_time
            return 0.0
        if video_time < self._last_time:
            # перемотка назад / рестарт источника — таймеры недействительны
            self._reset_timers()
            self._last_time = video_time
            return 0.0
        dt = min(video_time - self._last_time, cfg.MAX_DT_SEC)
        self._last_time = video_time
        return max(dt, 0.0)

    def _reset_timers(self) -> None:
        for st in self.tracked_bags.values():
            st["history"].clear()
            st["warning_since"] = None
            st["stationary_since"] = None
            st["elapsed"] = 0.0
            st["status"] = "SAFE"
            st["reason"] = "reset"

    # ------------------------------------------------------------------- люди

    def _link_persons(self, persons: dict) -> None:
        """Привязывает track_id ByteTrack к стабильным внутренним id (lid).

        ByteTrack теряет ID при пропуске кадров, а нам нужен стабильный
        «владелец». Поэтому новый track_id пытаемся приклеить к недавно
        пропавшему человеку по расстоянию.
        """
        used: set[int] = set()
        t = self._last_time or 0.0

        for track_id, info in persons.items():
            for lid, st in self.persons_state.items():
                if st["track_id"] == track_id:
                    st.update(
                        center=info["center"],
                        pos=info["feet"],
                        bbox=info["bbox"],
                        height=info["height"],
                        missing_sec=0.0,
                        last_seen=t,
                    )
                    used.add(lid)
                    break

        known = {st["track_id"] for st in self.persons_state.values()}
        for track_id, info in persons.items():
            if track_id in known:
                continue

            best_lid, best_d = None, float("inf")
            for lid, st in self.persons_state.items():
                if lid in used or st["missing_sec"] <= 0.0:
                    continue
                d = math.hypot(info["feet"][0] - st["pos"][0],
                               info["feet"][1] - st["pos"][1])
                radius = max(cfg.PERSON_LINK_K * st["height"], 40.0)
                if d < radius and d < best_d:
                    best_d, best_lid = d, lid

            if best_lid is not None:
                st = self.persons_state[best_lid]
                st.update(
                    track_id=track_id,
                    center=info["center"],
                    pos=info["feet"],
                    bbox=info["bbox"],
                    height=info["height"],
                    missing_sec=0.0,
                    last_seen=t,
                )
                used.add(best_lid)
            else:
                self.persons_state[self._next_lid] = {
                    "track_id": track_id,
                    "center": info["center"],
                    "pos": info["feet"],
                    "bbox": info["bbox"],
                    "height": info["height"],
                    "missing_sec": 0.0,
                    "last_seen": t,
                }
                used.add(self._next_lid)
                self._next_lid += 1

    def _drop_persons(self, dt: float) -> None:
        """Увеличивает «время без детекции» и забывает людей, пропавших надолго.

        Если пропал владелец сумки — снимаем привязку, иначе `owner_lid`
        навсегда указывает на несуществующего человека и тревога не снимается.
        """
        t = self._last_time or 0.0
        for lid in list(self.persons_state):
            st = self.persons_state[lid]
            if st["last_seen"] < t:
                st["missing_sec"] += dt
            if st["missing_sec"] > cfg.PERSON_MISSING_SEC:
                del self.persons_state[lid]
                for bag in self.tracked_bags.values():
                    if bag["owner_lid"] == lid:
                        bag["owner_lid"] = None
                        self._emit("owner_forgotten", bag)

    # ------------------------------------------------------------------ сумки

    def _near_radius(self, diag: float) -> float:
        r = cfg.OWNER_RADIUS_K * diag
        r = max(r, cfg.OWNER_RADIUS_MIN_PX)
        r = min(r, cfg.OWNER_RADIUS_MAX_FRAC * self._frame_h)
        return r

    def _plausible(self, info: dict) -> bool:
        """Отсекает боксы, которые физически не могут быть сумкой.

        Дверной проём в video1 — 18% кадра. Модель путает его с сумкой, и без
        этого фильтра логика начинает искать «владельца» у двери.
        """
        x1, y1, x2, y2 = info["bbox"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        return area <= cfg.MAX_BAG_AREA_FRAC * self._frame_area

    def _new_state(self, track_id: int, info: dict) -> dict:
        x1, y1, x2, y2 = info["bbox"]
        diag = math.hypot(x2 - x1, y2 - y1)
        return {
            "track_id": track_id,
            "bbox": info["bbox"],
            "center": info["center"],
            "diag": max(diag, 1.0),
            "conf": info.get("conf", 0.0),
            "first_seen": self._last_time or 0.0,
            "last_seen": self._last_time or 0.0,
            "missing_sec": 0.0,
            "owner_lid": None,
            "status": "SAFE",
            "reason": "new",
            "warning_since": None,
            "stationary_since": None,
            "stationary": False,
            "motion": "unknown",
            "was_moving": False,
            "age": 0.0,
            "stationary_sec": 0.0,
            "elapsed": 0.0,
            "owner_distance": None,
            "bystander_lid": None,
            "bystander_since": None,
            "bystander_active": False,
            "radius": self._near_radius(diag),
            "history": deque(maxlen=256),
        }

    def _apply_detection(self, st: dict, info: dict) -> None:
        x1, y1, x2, y2 = info["bbox"]
        t = self._last_time or 0.0
        # через пропуск детекций историю движения продолжать нельзя:
        # смещение «до» и «после» пропуска ничего не говорит о движении
        if st["missing_sec"] > cfg.STATIONARY_GAP_SEC:
            st["history"].clear()
        st["bbox"] = info["bbox"]
        st["center"] = info["center"]
        st["diag"] = max(math.hypot(x2 - x1, y2 - y1), 1.0)
        st["conf"] = info.get("conf", 0.0)
        st["last_seen"] = t
        st["missing_sec"] = 0.0
        st["history"].append((t, info["center"]))

    def _relink(self, info: dict) -> dict | None:
        """Ищет существующее состояние для нового track_id (ID switch)."""
        best, best_score = None, 0.0
        for st in self.tracked_bags.values():
            if st["missing_sec"] > cfg.BAG_RELINK_SEC:
                continue
            iou = _iou(st["bbox"], info["bbox"])
            d = math.hypot(info["center"][0] - st["center"][0],
                           info["center"][1] - st["center"][1])
            limit = max(st["diag"], 40.0)
            if iou < cfg.BAG_RELINK_IOU and d > limit:
                continue
            score = iou + max(0.0, 1.0 - d / limit) * 0.5
            if score > best_score:
                best, best_score = st, score
        return best

    def _motion_state(self, st: dict, t: float) -> str:
        """'unknown' (мало данных) | 'stationary' | 'moving'."""
        hist = st["history"]
        while hist and t - hist[0][0] > cfg.STATIONARY_WINDOW_SEC:
            hist.popleft()
        if len(hist) < 2:
            return "unknown"
        span = t - hist[0][0]
        if span < cfg.STATIONARY_WINDOW_SEC * 0.6:
            return "unknown"
        cx, cy = st["center"]
        shift = max(math.hypot(px - cx, py - cy) for _, (px, py) in hist)
        threshold = max(cfg.STATIONARY_SHIFT_K * st["diag"], cfg.STATIONARY_SHIFT_MIN_PX)
        return "stationary" if shift < threshold else "moving"

    # -------------------------------------------------------------- состояние

    def _update_state(self, st: dict) -> None:
        t = self._last_time or 0.0
        center = st["center"]
        radius = self._near_radius(st["diag"])
        st["radius"] = radius

        motion = self._motion_state(st, t)
        stationary = motion == "stationary"
        if motion == "moving":
            # только реальное движение означает «несут» и обнуляет таймер
            st["was_moving"] = True
            st["stationary_since"] = None
        elif motion == "stationary" and st["stationary_since"] is None:
            st["stationary_since"] = t
            self._emit("bag_stationary", st)
        st["motion"] = motion
        st["stationary"] = stationary
        st["stationary_sec"] = (t - st["stationary_since"]) if stationary else 0.0

        # ближайший человек вообще и ближайший «не владелец»
        nearest_lid, nearest_d = None, float("inf")
        bystander_lid, bystander_d = None, float("inf")
        for lid, pst in self.persons_state.items():
            if pst["missing_sec"] > 0.0:
                continue
            d = _box_gap(pst["bbox"], st["bbox"])
            if d < nearest_d:
                nearest_d, nearest_lid = d, lid
            if lid != st["owner_lid"] and d < bystander_d:
                bystander_d, bystander_lid = d, lid

        # владелец: только тот, кто несёт сумку. Стоящую сумку посторонний
        # «присвоить» не может, и статичное ложное срабатывание (дверь) тоже
        # не получает владельца — иначе система рисует SAFE у двери.
        if motion == "moving" and nearest_lid is not None \
                and nearest_d <= radius and nearest_lid != st["owner_lid"]:
            st["owner_lid"] = nearest_lid
            st["had_owner"] = True
            self._emit("owner_assigned", st, person_lid=nearest_lid,
                       distance_px=nearest_d)

        # владелец рядом? с гистерезисом, чтобы не мигало на границе радиуса
        owner_near = False
        owner_dist = None
        st["owner_pos"] = None
        if st["owner_lid"] is not None:
            pst = self.persons_state.get(st["owner_lid"])
            if pst is not None and pst["missing_sec"] == 0.0:
                owner_dist = _box_gap(pst["bbox"], st["bbox"])
                limit = radius * (cfg.OWNER_LEAVE_K if st["status"] == "SAFE" else 1.0)
                owner_near = owner_dist <= limit
                st["owner_pos"] = pst["pos"]
        st["owner_distance"] = owner_dist

        # посторонний рядом: должен простоять, а не пройти мимо
        bystander_sec = 0.0
        st["bystander_pos"] = None
        if bystander_lid is not None and bystander_d <= radius:
            if st["bystander_lid"] != bystander_lid:
                st["bystander_lid"] = bystander_lid
                st["bystander_since"] = t
            since = st["bystander_since"]
            bystander_sec = 0.0 if since is None else max(0.0, t - since)
            pst = self.persons_state.get(bystander_lid)
            if pst is not None:
                st["bystander_pos"] = pst["pos"]
        else:
            st["bystander_lid"] = None
            st["bystander_since"] = None

        bystander = bystander_sec >= cfg.BYSTANDER_CONFIRM_SEC
        if bystander and not st["bystander_active"]:
            self._emit("bystander_near", st, person_lid=st["bystander_lid"],
                       distance_px=bystander_d)
        st["bystander_active"] = bystander

        # ---- переходы состояний ----
        prev_status, prev_reason = st["status"], st["reason"]
        # «владелец был» помним даже после того, как его трек пропал:
        # иначе на экране «NO OWNER» вместо честного «владелец ушёл»
        had_owner = st["owner_lid"] is not None or st.get("had_owner", False)

        age = t - st["first_seen"]
        st["age"] = age

        if motion == "moving":
            # сумку несут — тревоги быть не может
            status, reason = "SAFE", "carried"
            st["warning_since"] = None
            st["elapsed"] = 0.0
        elif motion == "unknown" and age <= cfg.MOTION_GRACE_SEC:
            # только что появилась: окно движения ещё не заполнилось
            status, reason = "SAFE", "new"
            st["warning_since"] = None
            st["elapsed"] = 0.0
        elif owner_near:
            status, reason = "SAFE", "owner_near"
            st["warning_since"] = None
            st["elapsed"] = 0.0
        else:
            # motion == "stationary" ИЛИ "unknown" (пропуск детекций).
            # unknown НЕ обнуляет таймер: иначе мигающая детекция лежащей сумки
            # никогда не доживёт до ALARM — elapsed перезапускался бы каждый раз.
            if st["warning_since"] is None:
                st["warning_since"] = t
            st["elapsed"] = max(0.0, t - st["warning_since"])
            base_reason = "owner_left" if had_owner else "owner_unknown"

            if bystander and not cfg.BYSTANDER_ESCALATE:
                status, reason = "WARNING", "bystander"
            elif (st["elapsed"] >= cfg.ALARM_DELAY_SEC
                  and (st["was_moving"] or not cfg.REQUIRE_MOVED)):
                status, reason = "ALARM", base_reason
            else:
                status, reason = "WARNING", base_reason

        # тревога не гаснет сама: если уже ALARM, посторонний рядом её не отменяет.
        # Снять может только возврат владельца или то, что сумку унесли.
        if prev_status == "ALARM" and status == "WARNING":
            status, reason = "ALARM", prev_reason

        st["status"] = status
        st["reason"] = reason

        # события — уже с НОВЫМ статусом
        if status == "ALARM" and prev_status != "ALARM":
            self._emit("alarm", st, status=status, reason=reason,
                       elapsed=st["elapsed"], distance_px=owner_dist)
        elif prev_status == "ALARM" and status != "ALARM":
            self._emit("alarm_cleared", st, status=status, reason=reason,
                       elapsed=st["elapsed"])
        elif prev_status in ("WARNING", "ALARM") and status == "SAFE":
            if reason == "carried":
                self._emit("bag_taken", st, status=status, reason=reason)
            else:
                self._emit("owner_returned", st, status=status, reason=reason,
                           distance_px=owner_dist)
        elif prev_status == "SAFE" and status == "WARNING":
            self._emit("owner_left" if had_owner else "owner_unknown", st,
                       status=status, reason=reason, distance_px=owner_dist)
        elif prev_reason == "bystander" and reason != "bystander":
            self._emit("bystander_left", st, status=status, reason=reason,
                       distance_px=owner_dist)

    # ------------------------------------------------------------------ update

    def update(self, persons: dict, bags: dict, video_time: float,
               frame_size: tuple | None = None) -> dict:
        """Один шаг логики.

        persons: {track_id: {'center', 'feet', 'bbox', 'height', 'conf'}}
        bags:    {track_id: {'center', 'bbox', 'conf'}}
        video_time: время источника в секундах (не номер кадра!)
        frame_size: (h, w) — нужен, чтобы ограничить радиус относительно кадра
        """
        if frame_size is not None:
            self._frame_h = int(frame_size[0])
            self._frame_area = int(frame_size[0]) * int(frame_size[1])

        dt = self._dt(video_time)
        self._link_persons(persons)
        self._drop_persons(dt)

        # 0) отсев неправдоподобных боксов (дверь/стена/кусок пола)
        rejected: set[int] = set()
        filtered: dict[int, dict] = {}
        for track_id, info in bags.items():
            if self._plausible(info):
                filtered[track_id] = info
            else:
                rejected.add(track_id)
                if track_id not in self._rejected_ids:
                    x1, y1, x2, y2 = info["bbox"]
                    area_frac = ((x2 - x1) * (y2 - y1)) / max(self._frame_area, 1)
                    print(f"[logic] отброшен неправдоподобный бокс #{track_id}: "
                          f"{x2 - x1}x{y2 - y1} = {area_frac:.1%} кадра", flush=True)
        self._rejected_ids = rejected
        bags = filtered

        used_states: set[int] = set()
        pending: list[tuple[int, dict]] = []

        # 1) совпадение по track_id — основной путь
        for track_id, info in bags.items():
            st = self.tracked_bags.get(track_id)
            if st is not None and id(st) not in used_states:
                self._apply_detection(st, info)
                used_states.add(id(st))
            else:
                pending.append((track_id, info))

        # 2) новый track_id — возможно, ID switch: приклеиваем к старому состоянию
        for track_id, info in pending:
            st = self._relink(info)
            if st is not None and id(st) not in used_states:
                old_key = st["track_id"]
                if old_key in self.tracked_bags:
                    del self.tracked_bags[old_key]
                st["track_id"] = track_id
                self.tracked_bags[track_id] = st
                self._apply_detection(st, info)
                used_states.add(id(st))
                self._emit("bag_relinked", st, person_lid=None)
            else:
                st = self._new_state(track_id, info)
                st["history"].append((video_time, info["center"]))
                self.tracked_bags[track_id] = st
                used_states.add(id(st))
                self._emit("bag_appeared", st)

        # 3) дубли одной и той же сумки в кадре — оставляем более уверенный
        seen = [st for st in self.tracked_bags.values() if id(st) in used_states]
        for i, a in enumerate(seen):
            for b in seen[i + 1:]:
                if id(a) not in used_states or id(b) not in used_states:
                    continue
                if _iou(a["bbox"], b["bbox"]) >= cfg.BAG_DEDUP_IOU:
                    drop = a if a["conf"] < b["conf"] else b
                    keep = b if drop is a else a
                    self.tracked_bags.pop(drop["track_id"], None)
                    used_states.discard(id(drop))
                    self._emit("bag_dedup", keep, person_lid=None)

        # 4) состояния: стационарность, владелец, статус
        for st in self.tracked_bags.values():
            if id(st) in used_states:
                self._update_state(st)

        # 5) пропавшие сумки: помним BAG_MISSING_SEC, потом забываем
        for track_id, st in list(self.tracked_bags.items()):
            if id(st) in used_states:
                continue
            st["missing_sec"] += dt
            if st["missing_sec"] > cfg.BAG_MISSING_SEC:
                self._emit("bag_lost", st)
                del self.tracked_bags[track_id]

        return self.tracked_bags
