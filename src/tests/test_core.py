"""Автотесты логики, журнала и отчёта (без модели, без видео, без GPU).

Запуск из src/:
    python -m unittest discover -s tests -v
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import config as cfg  # noqa: E402
from event_logger import EventLogger  # noqa: E402
from logic import AbObjDetector  # noqa: E402
from tools.sim_scene import FPS, H, W, scene_at, render  # noqa: E402

FRAME_SHAPE = (H, W, 3)


def run_logic(scene, duration, stride=1, fps=FPS):
    det = AbObjDetector()
    events = []
    n = int(duration * fps)
    for i in range(1, n + 1, stride):
        t = i / fps
        persons, bags = scene(t)
        det.update(persons, bags, t, FRAME_SHAPE)
        events.extend(det.drain_events())
    return events


def times(events, etype):
    return [e.time for e in events if e.type == etype]


class LogicTests(unittest.TestCase):
    def test_left_bag_raises_single_alarm_after_delay(self):
        ev = run_logic(scene_at, 30)
        alarms = times(ev, "alarm")
        self.assertEqual(len(alarms), 1, "должна быть ровно одна тревога")
        left = times(ev, "owner_left")[0]
        self.assertAlmostEqual(alarms[0] - left, cfg.ALARM_DELAY_SEC, delta=0.5)

    def test_alarm_cleared_when_bag_taken(self):
        ev = run_logic(scene_at, 45)
        self.assertTrue(times(ev, "alarm_cleared"), "тревога должна сняться, когда сумку унесли")

    def test_carried_bag_never_alarms(self):
        def walker(t):
            # человек 40 с ходит туда-сюда с сумкой в руке
            persons, bags = {}, {}
            x = 100 + (t * 60) % 700
            persons[1] = {"center": (int(x), 315), "feet": (int(x), 430),
                          "bbox": (int(x) - 35, 200, int(x) + 35, 430), "height": 230, "conf": .9}
            bags[2] = {"center": (int(x) + 40, 360), "feet": (int(x) + 40, 382),
                       "bbox": (int(x) + 18, 338, int(x) + 62, 382), "height": 44, "conf": .8}
            return persons, bags
        self.assertEqual(times(run_logic(walker, 40), "alarm"), [])

    def test_static_false_positive_never_alarms(self):
        # «сумка», которая ни разу не двигалась (текстура пола) — REQUIRE_MOVED
        def static(t):
            return {}, {5: {"center": (200, 450), "feet": (200, 470),
                            "bbox": (180, 430, 220, 470), "height": 40, "conf": .5}}
        self.assertEqual(times(run_logic(static, 40), "alarm"), [])

    def test_timer_independent_of_frame_rate(self):
        a1 = times(run_logic(scene_at, 30, stride=1), "alarm")
        a3 = times(run_logic(scene_at, 30, stride=3), "alarm")
        self.assertEqual(len(a1), 1)
        self.assertEqual(len(a3), 1)
        self.assertLess(abs(a1[0] - a3[0]), 1.0, "логика должна считать секунды, а не кадры")


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_logger_writes_csv_snapshots_and_report(self):
        log = EventLogger(root=self.tmp, source="unit-test", model="none")
        det = AbObjDetector()
        for i in range(1, int(26 * FPS)):
            t = i / FPS
            persons, bags = scene_at(t)
            det.update(persons, bags, t, FRAME_SHAPE)
            evs = det.drain_events()
            if evs:
                frame = render(t, i)
                for e in evs:
                    log.log(e, frame, frame)
        report = log.close()

        d = log.dir
        self.assertTrue((d / "events.csv").exists())
        self.assertTrue(report.exists())
        rows = [json.loads(x) for x in (d / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        alarm = [r for r in rows if r["type"] == "alarm"]
        self.assertEqual(len(alarm), 1)
        self.assertTrue((d / alarm[0]["frame_path"]).exists(), "снимок кадра тревоги")
        self.assertTrue((d / alarm[0]["crop_path"]).exists(), "вырезка сумки")
        meta = json.loads((d / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["summary"]["alarms"], 1)
        self.assertIn("ТРЕВОГА", report.read_text(encoding="utf-8"))

    def test_numpy_values_are_json_safe(self):
        """Реальный YOLO отдаёт np.int64/np.float32: строки журнала должны
        сериализоваться без конвертеров (иначе /api/state падает с 500)."""
        import numpy as np
        log = EventLogger(root=self.tmp, source="unit-test", model="none")
        row = log.log({"type": "alarm", "time": np.float32(1.5), "bag_id": np.int64(7),
                       "status": "ALARM", "reason": "owner_left", "elapsed": np.float32(15.0),
                       "bbox": (np.int64(1), np.int64(2), np.int64(3), np.int64(4))})
        json.dumps(row)                     # без default= не должно падать
        self.assertIsInstance(row["bag_id"], int)
        self.assertIsInstance(row["bbox"][0], int)
        log.close()

    def test_abort_removes_empty_session(self):
        log = EventLogger(root=self.tmp, source="unit-test", model="none")
        d = log.dir
        log.abort()
        self.assertFalse(d.exists(), "пустая папка сессии не должна оставаться")


if __name__ == "__main__":
    unittest.main(verbosity=2)
