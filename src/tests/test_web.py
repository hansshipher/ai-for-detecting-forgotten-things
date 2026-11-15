"""Автотесты HTTP-слоя веб-панели — без модели и без запуска видео."""

import importlib.util
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

HAS_FLASK = importlib.util.find_spec("flask") is not None


@unittest.skipUnless(HAS_FLASK, "Flask не установлен: pip install -r requirements.txt")
class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import config as cfg
        import web_app
        cls.cfg, cls.web = cfg, web_app
        cls.c = web_app.app.test_client()

    def _sessions(self):
        d = self.cfg.BASE_DIR / "runs" / "sessions"
        return {p.name for p in d.iterdir()} if d.is_dir() else set()

    def test_index_and_options(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        r.close()                       # send_from_directory держит файл открытым
        opt = self.c.get("/api/options").get_json()
        self.assertIn("weights", opt)
        self.assertIn("settings", opt)

    def test_settings_reject_non_number(self):
        r = self.c.post("/api/settings", json={"alarm_delay": "abc"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("числом", r.get_json()["error"])

    def test_state_rejects_bad_since(self):
        self.assertEqual(self.c.get("/api/state?since=abc").status_code, 400)

    def test_model_must_live_in_weights(self):
        """.pt — это pickle: панель не должна грузить файл по произвольному пути."""
        before = self._sessions()
        r = self.c.post("/api/start", json={"source": str(Path(__file__).resolve()),
                                            "model": "../src/config.py"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("weights", r.get_json()["error"])
        self.assertEqual(self._sessions(), before, "неудачный старт не должен оставлять сессию")

    def test_missing_source_rejected(self):
        self.assertEqual(self.c.post("/api/start", json={}).status_code, 400)

    def test_session_path_traversal_blocked(self):
        self.assertEqual(self.c.get("/sessions/../config.py").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
