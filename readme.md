# AI for Detecting Forgotten Things

Система компьютерного зрения на Python и YOLO: находит в видеопотоке людей и сумки/рюкзаки, отслеживает их между кадрами и определяет, оставлена ли вещь без присмотра. Результат — статус `SAFE` / `WARNING` / `ALARM` и события вида «владелец ушёл», «рядом посторонний», «тревога».

Технологический стек:

* **Python 3.11**
* **Ultralytics YOLO** (модель `yolo26n`)
* **ByteTrack** для трекинга объектов
* **OpenCV**

---

## Быстрый старт. Как развернуть проект у себя?

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/hansshipher/ai-for-detecting-forgotten-things.git
cd ai-for-detecting-forgotten-things
```

### 2. Создайте и активируйте виртуальное окружение

* **Windows (PowerShell)**:

```powershell
python -m venv .venv
.venv/Scripts/activate
```

* **macOS / Linux**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установите зависимости

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если есть NVIDIA GPU, поставьте torch с CUDA по инструкции с [pytorch.org](https://pytorch.org/get-started/locally/) — иначе всё будет считаться на CPU (работает, но медленнее).

### 4. Скачайте веса модели

Рабочая модель **не хранится в git** — её нужно скачать отдельно и положить в папку `weights/`.

* **Папка с моделями:** [Google Drive](https://drive.google.com/drive/folders/1A_ShEM1srlCQk-eEvo9lz4037k7ym1KV?usp=sharing)
* **Файл для демо:** `best12k.pt`
* **Куда положить:** `weights/best12k.pt`
* **Размер:** 5 443 141 байт
* **SHA256:** `DFE30F9B236831871AF2C3B73DCB6E1C6F02BBEE9FB944CE900A1B74E39CF0A1`

В той же папке лежит прежняя демо-модель `best26n_nodc_640.pt` (5 451 141 байт,
SHA256 `83AD9DC0BF1DC86AB9F424BCE7C740C7670A52D0D08971742A24BCAEC446DE93`) —
её можно взять, если нужно вернуть `MODEL_NAME` в `src/config.py` на прежнее имя.

Проверка целостности после скачивания:

* **Windows (PowerShell)**:

```powershell
Get-FileHash .\weights\best12k.pt -Algorithm SHA256
```

* **macOS / Linux**:

```bash
sha256sum weights/best12k.pt
```

Хеш должен совпасть посимвольно. Если не совпал — файл скачался не полностью или это другая модель, скачайте заново.

Что это за модель: `yolo26n`, `imgsz=640`, обучена на `dataset/12k_merged`
(65 эпох), 2 класса — `bag`, `person`. На своей валидации 12k: mAP50 0.847 /
mAP50-95 0.570. На честном `cctv_holdout` — 0.382 / 0.171: метрика низкая, но
именно эта модель находит лежащую сумку, и на `video1` логика даёт корректный
`ALARM` при нуле ложных тревог на остальных видео.

> Папка `weights/` добавлена в `.gitignore`. Обученные и скачанные веса в репозиторий не коммитим — делимся через облако.

### 5. Запустите проект

Все скрипты запускаются **из папки `src/`** — в них используются плоские импорты (`import config as cfg`). Это касается и подпапок: `python eval/eval_videos.py`, `python tools/build_12k_merged.py`.

```bash
cd src
python test.py
```

Полезные варианты запуска:

```bash
python test.py --source ../videos/video2.avi   # другое видео
python test.py --camera --source 0             # веб-камера (live-режим)
python test.py --camera --source rtsp://...    # IP-камера
```

Окно с результатом закрывается клавишей `q`.

---

## Как это работает

Пайплайн: кадр → детекция и трекинг → логика состояний → отрисовка.

* `tracker.py` — класс `ObjTracker`: YOLO + ByteTrack. На выходе люди и сумки с `track_id`, боксами и уверенностью.
* `logic.py` — класс `AbObjDetector`: состояние по каждой сумке. Хранит владельца, историю движения, посторонних рядом; считает таймеры **в секундах**, а не в кадрах.
* `visualizer.py` — рисует боксы, статусы и связи «человек ↔ сумка».
* `test.py` — точка входа: читает видео/камеру, инференс в отдельном потоке (всегда обрабатывается самый свежий кадр), показывает окно и FPS.

Правила статусов:

* **SAFE** — рядом владелец, либо сумку несут.
* **WARNING** — сумка стоит без владельца, но тревога ещё не положена: рядом посторонний (эскалация заморожена) или не истёк `ALARM_DELAY_SEC`.
* **ALARM** — сумка неподвижна, владельца нет, посторонних рядом нет, прошло не меньше `ALARM_DELAY_SEC` (по умолчанию 15 с).
* Тревога «защёлкивается»: снимается только если вернулся владелец или сумку забрали.
* Посторонний рядом **не** даёт `SAFE` — только `WARNING`, решение остаётся за оператором.
* Статичное ложное срабатывание (дверной проём, текстура) не может дать `ALARM`: тревожиться можно только за сумку, которую принесли и которая двигалась (`REQUIRE_MOVED`).

Все пороги и таймеры — в `src/config.py` (`ALARM_DELAY_SEC`, `BYSTANDER_ESCALATE`, радиусы близости, пороги стационарности и т.д.).

---

## Проверка качества

Все команды — из `src/`.

```bash
# Полевой замер по videos/: сколько сумок находит модель, медианный размер бокса, мс/кадр
python eval/eval_videos.py
python eval/eval_videos.py --models best26n_nodc_640 best26n --save-dir ../runs/field_dump

# Офлайн-прогон логики по видео (сколько тревог, какие события)
python eval/eval_logic.py
python eval/eval_logic.py --save-dir ../runs/logic_check --csv ../runs/logic_eval.csv

# Проверка, что логика считает секунды, а не кадры: сравнение stride 1 и 3
python eval/eval_logic.py --check
```

Метрики на `dataset/val` завышены (соседние кадры тех же видео попадают в train), поэтому честную оценку даёт только прогон по `videos/video1..4`.

Работа с датасетом (нужен локальный `dataset/`, в git его нет):

```bash
python tools/check_dataset.py --splits all      # аудит разметки: полигоны, битые боксы, дубликаты
python tools/merge_splits.py --apply            # пересборка dataset/all + data_all.yaml
python tools/audit_duplicates.py --root ../dataset/12k-dataset   # dHash-кластеры и утечки между сплитами
```

Обучение. Каждый прогон пишется в отдельную папку `runs/train/<name>`, рабочая модель
`weights/best26n_nodc_640.pt` при этом не перезаписывается:

```bash
python train.py --data ../dataset/12k_merged/data.yaml --name cctv_yolo26n_12k_640

# файнтюн на своём домене от предыдущего прогона
python train.py --data ../dataset/data_all.yaml --name finetune_all_640 \
    --weights ../runs/train/cctv_yolo26n_12k_640/weights/best.pt --epochs 40
```

---

## Структура проекта

* `src/` — исходный код:
  * **приложение (демо)**: `config.py` — конфиг (модель, классы, пороги, цвета); `tracker.py` — детекция и ByteTrack; `logic.py` — статусы и события; `visualizer.py` — отрисовка; `test.py` — точка входа (видео/камера).
  * **обучение**: `train.py` — обёртка над ultralytics с дефолтами проекта.
  * **оценка** (`src/eval/`): `eval_videos.py` — полевой замер по `videos/`; `eval_logic.py` — офлайн-прогон логики и FPS; `eval_dataset.py` — mAP и P/R по классам на сплите.
  * **инструменты** (`src/tools/`): `audit_duplicates.py` — dHash-кластеры и утечки между сплитами; `annotate_person.py` — авторазметка людей COCO-учителем; `build_12k_merged.py` — сборка `dataset/12k_merged`; `make_contact_sheets.py` — контактные листы для визуальной проверки; `check_dataset.py` — аудит разметки; `merge_splits.py` — сборка `dataset/all`; `download_model.py` — базовые веса Ultralytics.
* `configs/custom_tracker.yaml` — настройки ByteTrack.
* `weights/` — веса моделей (`.pt`), **в git не хранятся**.
* `videos/` — тестовые ролики для демо и оценки (`video1..11`, из ABODA); **в git не хранятся**, см. «Данные и атрибуция».
* `dataset/`, `runs/`, `mem/` — датасет, результаты обучений и локальные заметки, **в git не хранятся**.

---

## Данные и атрибуция

* **`videos/`** — 11 последовательностей из **ABODA** (ABandoned Objects DAtaset),
  [github.com/kevinlin311tw/ABODA](https://github.com/kevinlin311tw/ABODA).
  Файла лицензии в репозитории ABODA нет (GitHub API отдаёт `license: null`),
  поэтому ролики **не распространяются вместе с проектом**: скачайте их из
  исходного репозитория и положите в `videos/`.

  Citation: K. Lin, S.-C. Chen, C.-S. Chen, D.-T. Lin, Y.-P. Hung,
  "Abandoned Object Detection via Temporal Consistency Modeling and
  Back-Tracing Verification for Visual Surveillance", *IEEE Transactions on
  Information Forensics and Security*, vol. 10, no. 7, 2015,
  DOI [10.1109/TIFS.2015.2408263](https://doi.org/10.1109/TIFS.2015.2408263).

  Важно: 11 роликов — это **5 сцен**: `video1`; `video2/3/4` (одна запись);
  `video5/6` (одна камера, 5 — ИК-ночь, 6 — видимый свет); `video7/8/9/10`
  (одна запись); `video11`. Независимых сцен для проверки — пять, а не
  одиннадцать.
* **`dataset/cctv`** — экспорт Roboflow, лицензия **CC BY 4.0** (атрибуция —
  в `README.dataset.txt` внутри архива).
* **`dataset/12k-dataset`** — публичный Roboflow-экспорт; условия и атрибуция
  указаны в его `README.dataset.txt`.

Лицензии инструментов: Ultralytics YOLO — **AGPL-3.0**, это надо учитывать при
публикации проекта и весов. Код и веса сторонних репозиториев не копируются.

---

## Известные ограничения

* Сумка, лежащая на полу, детектится нестабильно: в обучающей выборке почти нет вида сверху. Модель `best12k.pt` её видит, а файнтюны на своём домене — часто нет.
* Иногда человек у края кадра (частично обрезанный) классифицируется как `bag` — путаница классов `person → bag`.
* Полный путь до `ALARM` воспроизведён на `video1` с моделью `best12k.pt`: владелец уходит, сумка стоит, тревога на 15-й секунде (`runs/demo_check_12k/video1_alarm.jpg`). На остальных видео сцены «положил и ушёл дольше 15 с» нет, поэтому тревог там ноль.
* Честная количественная оценка — только `dataset/cctv_holdout` (108+60 кадров). `videos/` разметки не имеют и служат качественной проверкой на 5 сценах.
* Ускорение через OpenVINO INT8 для слабых ноутбуков — в планах, пока не реализовано.
* Все подписи на кадре — латиницей: `cv2.putText` не умеет кириллицу.

---

## Памятка для работы с Git

Чтобы не ломать проект и не конфликтовать кодом, соблюдайте простой процесс разработки:

1. **Всегда обновляйте `main` перед началом работы**:

```bash
git checkout main
git pull origin main
```

2. **Создавайте отдельную ветку под свою задачу**:

```bash
git checkout -b feature/nazvanie-zadachi
```

3. **!!! Не коммитьте веса моделей и датасет !!!**
   * Папки `weights/`, `runs/`, `dataset/`, `mem/` и файлы `.pt` добавлены в `.gitignore`.
   * Не добавляйте их в коммиты вручную через `git add .`, чтобы не раздувать репозиторий. Обученные и скачанные веса храните локально или делитесь через облако.
   * Добавляйте файлы поимённо: `git add src/logic.py`.

4. **Фиксируйте изменения и отправляйте в репозиторий**:

```bash
git add path/to/your/files.py
git commit -m "Добавил логику проверки расстояния"
git push origin feature/nazvanie-zadachi
```

5. **Создайте Pull Request (PR):**
   Зайдите на GitHub, перейдите во вкладку **Pull Request** и создайте запрос на слияние вашей ветки в `main`.

6. **Закрытие ветки.**
   После закрытия PR(!) выполните следующие действия:
   * Переключитесь на главную ветку `main`:

```bash
git checkout main
# или
git switch main
```

   * Затяните обновлённый `main` с GitHub:

```bash
git pull origin main
```

   * При необходимости удалите вашу ветку у себя на компьютере:

```bash
git branch -d feature/nazvanie-zadachi
```