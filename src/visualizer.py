import cv2

import config as cfg


class Visualizer:
    """Рисует детекции и статусы.

    Важно: состояния, пропавшие дольше GHOST_GRACE_SEC, не рисуются —
    иначе на экране остаются «призрачные» рамки с застывшим таймером.
    """

    def __init__(self):
        self.status_colors = {
            "SAFE": cfg.COLOR_SAFE,
            "WARNING": cfg.COLOR_WARNING,
            "ALARM": cfg.COLOR_ALARM,
        }

    @staticmethod
    def _person_center(person):
        # поддерживаем и новый формат (dict), и старый (кортеж)
        if isinstance(person, dict):
            return person["center"]
        return person

    def draw(self, frame, persons, bags):
        """Сырые детекции без логики (для отладки трекера)."""
        for person_id, person in persons.items():
            center = self._person_center(person)
            cv2.circle(frame, center, 5, cfg.COLOR_PERSON, -1)
            cv2.putText(frame, f"P-{person_id}", (center[0] - 15, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.COLOR_PERSON, 1)

        for bag_id, bag_info in bags.items():
            x1, y1, x2, y2 = bag_info["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Bag #{bag_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return frame

    @staticmethod
    def _label(bag_id, status, reason, elapsed):
        if status == "SAFE":
            return f"Bag #{bag_id} [carried]" if reason == "carried" else f"Bag #{bag_id} [OK]"
        if status == "ALARM":
            return f"ALARM! Bag #{bag_id} [{elapsed:.1f}s]"
        if reason == "bystander":
            return f"Bag #{bag_id} [BYSTANDER {elapsed:.1f}s]"
        if reason == "owner_unknown":
            return f"Bag #{bag_id} [NO OWNER {elapsed:.1f}s]"
        return f"Bag #{bag_id} [LEFT {elapsed:.1f}s]"

    def draw_status(self, frame, persons, tracked_bags):
        for person_id, person in persons.items():
            center = self._person_center(person)
            cv2.circle(frame, center, 5, cfg.COLOR_PERSON, -1)
            cv2.putText(frame, f"P-{person_id}", (center[0] - 15, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.COLOR_PERSON, 1)

        for bag_id, bag_data in tracked_bags.items():
            if bag_data.get("missing_sec", 0.0) > cfg.GHOST_GRACE_SEC:
                continue

            # Объект, который так и не сдвинулся, не может быть забытой вещью.
            # Через STATIC_HIDE_SEC убираем его с экрана: иначе дверной проём
            # висит как «сумка без владельца» всю смену.
            if (not bag_data.get("was_moving", False)
                    and bag_data.get("age", 0.0) > cfg.STATIC_HIDE_SEC):
                continue

            status = bag_data["status"]
            reason = bag_data.get("reason", "")
            elapsed = bag_data.get("elapsed", 0.0)
            x1, y1, x2, y2 = bag_data["bbox"]
            color = self.status_colors.get(status, cfg.COLOR_SAFE)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # линия до владельца: сразу видно, кто «отвечает» за сумку
            owner_pos = bag_data.get("owner_pos")
            if owner_pos is not None:
                cv2.line(frame, center, owner_pos, color, 1, cv2.LINE_AA)

            # линия до постороннего, из-за которого тревога заморожена
            if reason == "bystander":
                bystander_pos = bag_data.get("bystander_pos")
                if bystander_pos is not None:
                    cv2.line(frame, center, bystander_pos, cfg.COLOR_WARNING, 1, cv2.LINE_AA)

            if status in ("WARNING", "ALARM"):
                radius = int(bag_data.get("radius", cfg.OWNER_RADIUS_MIN_PX))
                cv2.circle(frame, center, radius, color, 1, lineType=cv2.LINE_AA)

            thickness = 3 if status == "ALARM" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = self._label(bag_id, status, reason, elapsed)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            frame_h, frame_w = frame.shape[:2]
            pad = 10
            x1_label = max(0, min(x1, frame_w - w - pad))
            y_top = max(0, y1 - 20)

            cv2.rectangle(frame, (x1_label, y_top), (x1_label + w + pad, y_top + 20), color, -1)
            text_color = (0, 0, 0) if status == "WARNING" else cfg.COLOR_TEXT
            cv2.putText(frame, label, (x1_label + 5, y_top + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        return frame


if __name__ == "__main__":
    import os
    import sys

    import numpy as np

    sys.path.insert(0, os.path.dirname(__file__))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    persons = {
        1: {"center": (500, 240), "feet": (500, 400), "bbox": (480, 150, 520, 400),
            "height": 250, "conf": 0.9},
        5: {"center": (120, 200), "feet": (120, 340), "bbox": (100, 120, 140, 340),
            "height": 220, "conf": 0.8},
    }
    tracked_bags = {
        7: {"status": "WARNING", "reason": "owner_left", "elapsed": 12.3,
            "bbox": (100, 200, 160, 280), "radius": 90, "missing_sec": 0.0,
            "owner_pos": (500, 400)},
        9: {"status": "ALARM", "reason": "owner_left", "elapsed": 16.0,
            "bbox": (560, 300, 620, 380), "radius": 70, "missing_sec": 0.0,
            "owner_pos": None},
        11: {"status": "WARNING", "reason": "bystander", "elapsed": 8.0,
             "bbox": (300, 100, 350, 160), "radius": 60, "missing_sec": 0.0,
             "owner_pos": None, "bystander_pos": (120, 340)},
        13: {"status": "SAFE", "reason": "carried", "elapsed": 0.0,
             "bbox": (400, 60, 450, 130), "radius": 50, "missing_sec": 2.0},
    }

    vis = Visualizer()
    out = vis.draw_status(frame, persons, tracked_bags)
    cv2.imwrite("src/_viz_check.png", out)
    print("saved src/_viz_check.png")
