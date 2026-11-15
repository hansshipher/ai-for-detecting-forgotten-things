"""HTML-отчёт по сессии: итоги, настройки и журнал тревог с картинками.

Строится из файлов папки сессии (session.json + events.jsonl), поэтому его
можно пересобрать в любой момент:

    python report.py ../runs/sessions/2026-11-15_12-00-00
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

LEVEL_COLORS = {"alarm": "#d93025", "warning": "#e8a200", "info": "#1a7f37", "debug": "#888"}
STATUS_COLORS = {"ALARM": "#d93025", "WARNING": "#e8a200", "SAFE": "#1a7f37"}

SETTING_TITLES = {
    "ALARM_DELAY_SEC": "Задержка тревоги, с",
    "CONF_THRESHOLD": "Порог уверенности детектора",
    "IMGSZ": "Размер входа модели",
    "BYSTANDER_ESCALATE": "Тревога даже при постороннем рядом",
    "OWNER_RADIUS_K": "Радиус «владелец рядом» (диагоналей сумки)",
    "REQUIRE_MOVED": "Тревога только для принесённых сумок",
    "MAX_BAG_AREA_FRAC": "Макс. доля кадра для сумки",
    "BAG_MISSING_SEC": "Память о пропавшей сумке, с",
    "PERSON_MISSING_SEC": "Память о пропавшем человеке, с",
    "MODEL_NAME": "Модель",
}


def _fmt_time(sec) -> str:
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return ""
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _esc(v) -> str:
    return html.escape("" if v is None else str(v))


def build_report(session_dir: Path | str) -> Path:
    from event_logger import VISIBLE_EVENTS  # локально: report.py можно звать отдельно

    d = Path(session_dir)
    meta = {}
    if (d / "session.json").exists():
        meta = json.loads((d / "session.json").read_text(encoding="utf-8"))
    rows = []
    if (d / "events.jsonl").exists():
        for line in (d / "events.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("type") in VISIBLE_EVENTS:
                    rows.append(r)

    s = meta.get("summary", {})
    alarms = [r for r in rows if r["type"] == "alarm"]

    cards = [
        ("Тревог", s.get("alarms", len(alarms)), "#d93025"),
        ("Предупреждений", s.get("warnings", 0), "#e8a200"),
        ("Сумок замечено", s.get("bags_seen", "—"), "#1967d2"),
        ("Длительность видео", _fmt_time(s.get("video_time_sec", 0)), "#444"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="num" style="color:{c}">{_esc(v)}</div>'
        f'<div class="lbl">{_esc(t)}</div></div>' for t, v, c in cards)

    settings = meta.get("settings", {})
    settings_html = "".join(
        f"<tr><td>{_esc(SETTING_TITLES.get(k, k))}</td><td><code>{_esc(v)}</code></td></tr>"
        for k, v in settings.items())

    # Галерея тревог — главное, что хочет увидеть оператор/комиссия
    gallery = ""
    for r in alarms:
        img = r.get("crop_path") or r.get("frame_path")
        full = r.get("frame_path") or img
        if not img:
            continue
        gallery += (
            f'<figure><a href="{_esc(full)}" target="_blank"><img src="{_esc(img)}"></a>'
            f'<figcaption>Сумка #{_esc(r.get("bag_id"))} · {_fmt_time(r.get("video_time"))}'
            f' · без присмотра {float(r.get("elapsed_sec") or 0):.1f} с</figcaption></figure>')
    if not gallery:
        gallery = '<p class="muted">Тревог за сессию не было.</p>'

    trs = []
    for r in rows:
        color = LEVEL_COLORS.get(r.get("level"), "#888")
        st = r.get("status", "")
        thumb = ""
        if r.get("crop_path") or r.get("frame_path"):
            img = r.get("crop_path") or r.get("frame_path")
            full = r.get("frame_path") or img
            thumb = f'<a href="{_esc(full)}" target="_blank"><img class="thumb" src="{_esc(img)}"></a>'
        trs.append(
            f'<tr><td>{_esc(r.get("n"))}</td><td>{_esc(r.get("wall_time"))}</td>'
            f'<td>{_fmt_time(r.get("video_time"))}</td>'
            f'<td><span class="dot" style="background:{color}"></span>{_esc(r.get("title"))}</td>'
            f'<td>{_esc(r.get("bag_id")) if (r.get("bag_id") or 0) >= 0 else "—"}</td>'
            f'<td style="color:{STATUS_COLORS.get(st, "#444")};font-weight:600">{_esc(st)}</td>'
            f'<td>{_esc(r.get("reason"))}</td>'
            f'<td>{float(r.get("elapsed_sec") or 0):.1f}</td><td>{thumb}</td></tr>')
    table = "".join(trs) or '<tr><td colspan="9" class="muted">Событий нет</td></tr>'

    page = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Отчёт сессии {_esc(d.name)}</title>
<style>
 body{{font-family:Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f5f6f8;color:#222}}
 header{{background:#1f2937;color:#fff;padding:20px 32px}}
 header h1{{margin:0;font-size:22px}} header p{{margin:4px 0 0;opacity:.75}}
 main{{padding:24px 32px;max-width:1200px}}
 .cards{{display:flex;gap:16px;flex-wrap:wrap}}
 .card{{background:#fff;border-radius:10px;padding:16px 22px;min-width:160px;box-shadow:0 1px 3px #0001}}
 .num{{font-size:30px;font-weight:700}} .lbl{{color:#666;font-size:13px}}
 h2{{margin-top:32px;font-size:18px}}
 table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px #0001}}
 th,td{{padding:8px 10px;border-bottom:1px solid #eee;text-align:left;font-size:14px;vertical-align:middle}}
 th{{background:#fafafa;font-weight:600}}
 .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}}
 .thumb{{height:56px;border-radius:4px}}
 .gallery{{display:flex;flex-wrap:wrap;gap:16px}}
 figure{{margin:0;background:#fff;padding:8px;border-radius:10px;box-shadow:0 1px 3px #0001}}
 figure img{{height:180px;border-radius:6px;display:block}}
 figcaption{{font-size:13px;color:#444;margin-top:6px}}
 .muted{{color:#888}} code{{background:#f0f0f0;padding:1px 5px;border-radius:4px}}
 .meta td:first-child{{width:340px;color:#555}}
</style></head><body>
<header><h1>Обнаружение оставленных вещей — отчёт сессии</h1>
<p>Источник: {_esc(meta.get("source"))} · Модель: {_esc(meta.get("model"))} ·
Начало: {_esc(meta.get("started"))} · Конец: {_esc(meta.get("finished", "—"))}</p></header>
<main>
<div class="cards">{cards_html}</div>
<h2>Тревоги</h2><div class="gallery">{gallery}</div>
<h2>Журнал событий</h2>
<table><tr><th>#</th><th>Время</th><th>Время видео</th><th>Событие</th><th>Сумка</th>
<th>Статус</th><th>Причина</th><th>Без присмотра, с</th><th>Снимок</th></tr>{table}</table>
<h2>Параметры запуска</h2><table class="meta">{settings_html}</table>
<p class="muted">Полный машинный журнал: events.csv / events.jsonl в папке сессии.</p>
</main></body></html>"""
    out = d / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python report.py <папка сессии>")
        sys.exit(1)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    print(build_report(sys.argv[1]))
