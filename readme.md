# AI for Detecting Forgotten Things
Система компьютерного зрения на базе Python и YOLO для автоматического обнаружения оставленных без присмотра вещей (сумок, рюкзаков) в видеопотоке.
---
Технологический стек:
* **Python 3.11**
* **Ultralitics YOLO**
* **ByteTrack / BoT-SORT** для трекинга обектов
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
  ``` PowerShell
  python -m venv .venv
  .venv/Scripts/activate
  ```
* **macOS / Linux**
  ``` Bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Установите зависимости
``` Bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Скачайте веса моделей
Чтобы автоматически загрузить базовую модель YOLO в папку `weights/`, запустите:
``` Bash
python download_model.py
```

### 5. Запустите проект
Убедитесь , что в папке `videos/` лежит тестовое видео (напримерб, `video1.avi`), после чего запустите скрипт проверки:
``` Bash
cd src
python test.py
```

## Структура проекта
* `src/` — папка с исходным кодом проекта:
  * `config.py` — центральный файл конфигурации (пути, классы COCO, параметры и цвета).
  * `tracker.py` — логика детекции и трекинга объектов (класс `ObjTracker`).
  * `visualizer.py` — отрисовка рамок, точек и ID людей/сумок на кадре.
  * `test.py` — точка входа, основной цикл обработки видеопотока.
  * `download_model.py` — скрипт автоматической загрузки весов модели.
* `configs/` — папка с кастомными настройками трекеров (`custom_tracker.yaml`).
* `weights/` — папка для хранения весов моделей (`.pt`) — *добавлена в .gitignore*.
* `videos/` — папка для тестовых видеофайлов.

## Памятка для работы с Git
Что бы не ломать проект и не конфликтовать кодом, соблюдайте простой процесс разработки:
1. **Всегда обновляйте `main` перед началом работы**:
   ``` Bash
   git checkout main
   git pull origin main
   ```
2. **Создавайте отдельную ветку под свою задачу:**
   ``` Bash
   git checkout -b feature/nazvanie-zadachi
   ```
3. **!!! Не коммитьте веса моделей! !!!**
   * Папка `weight/`, папка `runs/` и файлы с расширением `.pt` добавлены в `.gitignore`.
   * Не добавляйте их в коммиты вручную через `git add .`, чтобы не раздувать размер репозитория. Обученные или скаченные веса храните локально или делитесь через облако.
4. **Фиксируйте изменения и отправляйте в репозиторий:**
   ``` Bash
   git add path/to/your/files.py
   git commit -m "Добавил логику проверки расстояния"
   git push origin feature/nazvanie-zadachi
   ```
5. **Создайте Pull Request (PR):**
   Зайдите на GitHub, перейдите во вкладку **Pull Request** и создайте запрос на слияние вашей ветки в `main`.
6. **Закрытие ветки**
   После закрытия PR(!) выполните следующие действия:
   * Переключитесь на главную ветку `main`:
      ``` Bash
      git checkout main
      или
      git switch main
      ```
   * Затяните обновленный `main` с GitHub:
       ``` Bash
       git pull origin main
       ```
   * При необходимости удалите вашу ветку у себя на копьютере:
       ``` Bash
       git branch -d feature/nazvanie-zadachi
       ```