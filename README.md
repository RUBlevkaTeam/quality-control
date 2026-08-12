# E-CUP Quality Control

Репозиторий содержит два независимых контура:

1. **Локальные эксперименты** — быстрый text-only TF-IDF baseline, честные
   group-aware folds, OOF-предсказания и локальная Macro F1. Работает на CPU и
   подходит для Mac.
2. **Submission inference** — исходный мультимодальный Qwen pipeline для запуска
   проверяющей системой на NVIDIA H100.

Разделение намеренное: локальная модель даёт короткий цикл экспериментов, а
`run.py` сохраняет формат решения, которое можно отправить на проверку.

## 1. Подготовка окружения

Нужен Python 3.11–3.14 и `uv`.

```bash
uv sync
```

Для EDA-ноутбука:

```bash
uv sync --extra notebook
```

Для ручного запуска Qwen inference вне официального Docker-образа:

```bash
uv sync --extra submission
```

Train CSV должен находиться в `content/content_ecup/data.csv`, а изображения —
в `content/content_ecup/images/<id>/`.

## 2. Проверка кода

```bash
uv run python -m unittest discover -v
```

## 3. Быстрый локальный smoke test

```bash
uv run python -m scripts.run_text_baseline --quick --experiment-id E00_quick
```

`--quick` использует 3 folds и 30 тысяч TF-IDF-признаков. Это проверка всего
контура, а не финальная оценка модели.

## 4. Полный локальный baseline

```bash
uv run python -m scripts.run_text_baseline --experiment-id E01_tfidf
```

Pipeline:

```text
raw train
  -> нормализация текста и group_key для дублей
  -> StratifiedGroupKFold(category + label, 5 folds)
  -> word + char TF-IDF
  -> отдельная LogisticRegression для каждой категории
  -> OOF probabilities
  -> отдельный threshold для каждой категории
  -> F1 по категориям и их среднее
```

После запуска создаются:

- `reports/folds.csv` — сохранённое разбиение;
- `reports/oof/<experiment>.csv` — OOF probability и prediction каждой строки;
- `reports/<experiment>.json` — полный конфиг и метрики;
- `reports/experiments.csv` — таблица сравнения запусков;
- `reports/error_analysis.csv` — ошибки текущего эксперимента;
- `artifacts/text_baseline.joblib` — модель, обученная на всём train.

Главная метрика соревнования:

```text
score = (F1 БАД + F1 Легковоспламеняющиеся) / 2
```

Скрипт печатает метрику при пороге `0.5` и после подбора отдельного OOF-порога
для каждой категории. Для честного сравнения гипотез всегда используйте одни и
те же folds и записывайте новый `--experiment-id`.

Пример изменения параметров:

```bash
uv run python -m scripts.run_text_baseline \
  --experiment-id E02_more_char \
  --word-max-features 40000 \
  --char-max-features 100000 \
  --c 2.0
```

## 5. Локальное предсказание и проверка CSV

После обучения text baseline:

```bash
uv run python -m scripts.predict_text_baseline \
  -i content/content_ecup/data.csv \
  -o reports/local_submit.csv

uv run python -m scripts.validate_submission \
  --input content/content_ecup/data.csv \
  --output reports/local_submit.csv
```

Text-only CSV нужен для локальной отладки формата. Основной submission по-прежнему
генерирует `run.py` с Qwen embeddings и Qwen-комментариями.

## 6. Multimodal-эксперименты на Apple MPS

TF-IDF и `LogisticRegression` выполняются на CPU, поскольку scikit-learn не
поддерживает MPS. Qwen embeddings извлекаются на MPS и сохраняются на диск —
после этого подбор классификатора и порогов не требует повторного запуска VLM.

Проверьте MPS:

```bash
uv sync --extra submission
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

Сначала сделайте smoke test на 10 объектах. `--model` должен указывать на уже
скачанную локальную директорию `Qwen3-VL-Embedding-2B`:

```bash
mkdir -p models/Qwen
uv run --extra submission hf download Qwen/Qwen3-VL-Embedding-2B \
  --local-dir models/Qwen/Qwen3-VL-Embedding-2B

uv run --extra submission python -m scripts.extract_embeddings_mps \
  --model models/Qwen/Qwen3-VL-Embedding-2B \
  --limit 10 \
  --batch-size 1 \
  --pixel-preset S \
  --output cache/qwen3_vl_mps_smoke.npy
```

Полное извлечение:

```bash
uv run --extra submission python -m scripts.extract_embeddings_mps \
  --model models/Qwen/Qwen3-VL-Embedding-2B \
  --batch-size 2 \
  --pixel-preset S
```

На Mac с 24 ГБ unified memory начинайте с batch `1–2` и preset `S`. Скрипт явно
требует MPS и не переключается молча на CPU. После получения кэша:

```bash
uv run python -m scripts.run_embedding_baseline \
  --experiment-id E10_qwen_embedding_mps
```

Этот запуск использует те же group-aware folds и ту же соревновательную метрику,
что и text baseline.

## 7. WandB dashboard

WandB — опциональная зависимость и не попадает в submission ZIP. Установите её
вместе с MPS-зависимостями:

```bash
uv sync --extra submission --extra tracking
```

Один раз авторизуйтесь. Команда интерактивно запросит API key или прочитает его
из `WANDB_API_KEY`; ключ не передаётся аргументом и не сохраняется в Git:

```bash
uv run --extra tracking python -m scripts.setup_wandb \
  --project ecup-quality-control
```

API key создаётся на `https://wandb.ai/authorize` или в WandB User Settings.
Не открывайте `https://api.wandb.ai` в браузере: это SDK endpoint, его корневая
страница отвечает 404. Для CI используйте секрет
`WANDB_API_KEY`, а проект и команду можно указать через `WANDB_PROJECT` и
`WANDB_ENTITY`.

После авторизации новый experiment запускается так:

```bash
uv run --extra tracking python -m scripts.run_text_baseline \
  --experiment-id E02_tfidf \
  --wandb \
  --wandb-project ecup-quality-control \
  --wandb-tag baseline
```

Embedding baseline поддерживает те же `--wandb*` аргументы. На сайт уходят:

- параметры модели и folds;
- fold runtime и F1 при пороге 0.5;
- Macro F1 до и после threshold tuning;
- F1, precision и recall по категориям;
- thresholds и число OOF-ошибок;
- summary JSON, folds, OOF, error analysis и модель как versioned artifacts.

Загрузить уже рассчитанный полный `E01_tfidf` без повторного обучения:

```bash
uv run --extra tracking python -m scripts.log_report_wandb \
  --summary reports/E01_tfidf.json \
  --project ecup-quality-control \
  --tag baseline
```

Проверить интеграцию без входа и интернета:

```bash
uv run --extra tracking python -m scripts.setup_wandb \
  --project ecup-quality-control \
  --offline
```

Offline-запуски сохраняются в `wandb/`. После авторизации их можно отправить:

```bash
uv run --extra tracking wandb sync wandb/offline-run-*
```

## 8. Запуск submission pipeline

Нужны CUDA и две локальные модели:

```text
<SHARED_MODELS_PATH>/Qwen/Qwen3-VL-Embedding-2B/
<SHARED_MODELS_PATH>/Qwen/Qwen3.5-4B/
```

Генеративную модель можно скачать позже, когда классификационный MPS-контур уже
проверен:

```bash
uv run --extra submission hf download Qwen/Qwen3.5-4B \
  --local-dir models/Qwen/Qwen3.5-4B
```

Smoke test лучше делать на отдельном CSV из 10–20 строк и соответствующих папках
изображений:

```bash
SHARED_MODELS_PATH=/path/to/models uv run --extra submission python run.py \
  --test-data-path /path/to/smoke/data.csv \
  --output-path /path/to/smoke/submission.csv

uv run python -m scripts.validate_submission \
  --input /path/to/smoke/data.csv \
  --output /path/to/smoke/submission.csv
```

`run.py` не обучает модель и не считает локальную F1. Он загружает официальный
classifier artifact, строит Qwen embeddings и формирует финальные объяснения.

## 9. Сборка архива для отправки

```bash
uv run python -m scripts.build_submission
```

Получится `dist/quality-control-submission.zip`. В архив попадают только:

- `metadata.json`;
- `run.py`;
- `baseline_qwen3vl_bf16.joblib`;
- Python-модули из `src/`.

Train CSV, изображения, ноутбуки, локальные отчёты и кэши в архив не включаются.
Архив использует Docker image из `metadata.json`; модели ожидаются в
`SHARED_MODELS_PATH`, который предоставляет evaluator.

## 10. EDA

Готовый ноутбук находится в `experiments/eda.ipynb`:

```bash
uv run --extra notebook jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 experiments/eda.ipynb
```

EDA не обучает модель и не изменяет исходные данные.
