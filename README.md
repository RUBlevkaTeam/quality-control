# E-CUP Quality Control

В проекте один основной pipeline без TF-IDF:

```text
очищенный текст + изображения
  → Qwen3-VL-Embedding-2B
  → LogisticRegression отдельно для каждой категории
  → OOF-подбор threshold отдельно для каждой категории
  → среднее F1 по категориям
```

Локально Qwen работает через Apple MPS. В проверяющей системе тот же код
автоматически использует CUDA.

## 1. Установка

```bash
uv sync --extra submission --extra tracking
```

Для локального полного inference модели должны лежать в `models/Qwen/`:

```bash
mkdir -p models/Qwen
uv run --extra submission hf download Qwen/Qwen3-VL-Embedding-2B \
  --local-dir models/Qwen/Qwen3-VL-Embedding-2B
uv run --extra submission hf download Qwen/Qwen3.5-4B \
  --local-dir models/Qwen/Qwen3.5-4B
```

`Qwen3.5-4B` нужна только для комментариев. Для извлечения embeddings и OOF
достаточно первой модели. Модели, embeddings, отчёты и WandB-кэш добавлены в
`.gitignore`.

## 2. Быстрая проверка MPS

Полный датасет для этого не нужен:

```bash
uv run --extra submission python -m scripts.extract_embeddings_mps \
  --model models/Qwen/Qwen3-VL-Embedding-2B \
  --limit 10 \
  --batch-size 1 \
  --pixel-preset S \
  --output cache/qwen3_vl_mps_smoke.npy
```

Команда создаёт рядом три файла:

- `qwen3_vl_mps_smoke.npy` — матрица embeddings;
- `qwen3_vl_mps_smoke.ids.csv` — порядок `id`;
- `qwen3_vl_mps_smoke.json` — параметры и время запуска.

## 3. Полные embeddings и локальная метрика

Когда понадобится полноценный эксперимент, embeddings считаются один раз:

```bash
uv run --extra submission python -m scripts.extract_embeddings_mps \
  --model models/Qwen/Qwen3-VL-Embedding-2B \
  --batch-size 2 \
  --pixel-preset S \
  --output cache/qwen3_vl_mps.npy
```

После этого LogisticRegression можно перезапускать быстро, не загружая Qwen:

```bash
uv run python -m scripts.run_embedding_baseline \
  --embeddings cache/qwen3_vl_mps.npy \
  --experiment-id E10_qwen_embedding_mps
```

Опциональный group-aware поиск `C` и L2-нормализации:

```bash
uv run python -m scripts.run_embedding_baseline \
  --embeddings cache/qwen3_vl_mps.npy \
  --experiment-id E11_qwen_s_grid \
  --c-grid 0.01 0.1 1 10 \
  --search-normalization
```

Лучшая конфигурация выбирается отдельно для каждой категории по OOF F1. Все
кандидаты и выбранные параметры сохраняются в отчёте и WandB.

Разбиение — `StratifiedGroupKFold`: дубликаты одной карточки остаются в одном
fold. Метрика считается как:

```text
score = (F1 БАД + F1 Легковоспламеняющиеся) / 2
```

После запуска появляются:

- `reports/E10_qwen_embedding_mps.json` — итоговые метрики и thresholds;
- `reports/oof/E10_qwen_embedding_mps.csv` — OOF probabilities, predictions и ошибки;
- `reports/folds.csv` — fold для каждого `id`;
- `reports/experiments.csv` — общая таблица экспериментов;
- `baseline_qwen3vl_bf16.joblib` — sklearn artifact для локального анализа;
- `baseline_qwen3vl_bf16.npz` — переносимый линейный artifact для submission;
- `artifacts/backups/` — резервная копия предыдущего классификатора перед обучением.

## 4. WandB

Проект: [ecup-quality-control](https://wandb.ai/nikitamarchenko134-mirea-russian-technological-university/ecup-quality-control).

Один раз проверить авторизацию:

```bash
uv run --extra tracking python -m scripts.setup_wandb \
  --entity nikitamarchenko134-mirea-russian-technological-university
```

Запуск эксперимента с отправкой на сайт:

```bash
uv run --extra tracking python -m scripts.run_embedding_baseline \
  --embeddings cache/qwen3_vl_mps.npy \
  --experiment-id E10_qwen_embedding_mps \
  --wandb \
  --wandb-project ecup-quality-control \
  --wandb-entity nikitamarchenko134-mirea-russian-technological-university
```

Логируются F1/precision/recall по каждому fold и категории, mean F1 при
threshold `0.5`, mean F1 после подбора thresholds, сами thresholds, время,
количество ошибок, проверенные значения `C`, варианты L2-нормализации,
выбранная конфигурация каждой категории, OOF-таблица, folds и оба формата
финального классификатора.

## 5. Submission

`run.py` — entrypoint проверяющей системы. Модели Qwen не кладутся в ZIP:
проверяющая система предоставляет их в `SHARED_MODELS_PATH`:

```text
Qwen/Qwen3-VL-Embedding-2B/
Qwen/Qwen3.5-4B/
```

Собрать submission с текущим классификатором:

```bash
uv run python -m scripts.build_submission
```

До первого обучения доступны исходный joblib и эквивалентный NPZ. После обучения
`run_embedding_baseline` создаёт резервные копии в `artifacts/backups/`, заменяет
оба формата, а сборщик предпочитает компактный NPZ. Если NPZ отсутствует, старый
joblib остаётся полностью поддержан.

После локального inference проверить CSV можно так:

```bash
uv run python -m scripts.validate_submission \
  --input /path/to/test.csv \
  --output /path/to/submission.csv
```

Полный локальный запуск автоматически использует `./models`, MPS и небольшие
batch sizes:

```bash
uv run --extra submission python run.py \
  --test-data-path /path/to/test.csv \
  --output-path /path/to/submission.csv
```

В проверяющей системе тот же entrypoint берёт модели из `SHARED_MODELS_PATH`,
использует CUDA и увеличенные batch sizes. Train и inference по умолчанию
используют один pixel preset `S`.

Генеративная модель `Qwen3.5-4B` нужна только для финальных комментариев в
`run.py`; для локальной F1-метрики и экспериментов с классификатором она не нужна.

## 6. Проверка кода

```bash
uv run python -m unittest discover -v
uv run python -m scripts.build_submission
```
