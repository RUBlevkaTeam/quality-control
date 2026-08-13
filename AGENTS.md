# AGENTS.md

Инструкции для coding-агентов, работающих с репозиторием E-CUP Quality Control.

## Контекст проекта

Основной pipeline текущей ветки:

```text
очищенный текст + изображения
  → Qwen3-VL-Embedding-2B
  → отдельная LogisticRegression для каждой категории
  → category-specific OOF thresholds
  → бинарный verdict
  → Qwen3.5-4B генерирует комментарий
  → submission CSV
```

Категории:

- `БАД`;
- `Легковоспламеняющиеся`.

Семантика класса:

- `1` → `не бан`;
- `0` → `бан`.

Генеративная модель не должна менять verdict. Она получает уже готовое
предсказание и отвечает только за комментарий.

## Основные документы

Перед крупными изменениями прочитать:

- `README.md` — быстрый запуск;
- `PIPELINE_GUIDE.md` — полное описание команд и pipeline;
- `ULTRA_COMPARISON.md` — сравнение с веткой `ultra`;
- `PROJECT_STATUS.md` — текущее состояние и следующие эксперименты.

## Окружение

Проект использует `uv` и Python 3.11–3.14.

Базовое окружение:

```bash
uv sync
```

Qwen inference:

```bash
uv sync --extra submission
```

WandB:

```bash
uv sync --extra tracking
```

Все extras:

```bash
uv sync --extra submission --extra tracking
```

Не добавлять зависимости без необходимости. Новую зависимость нужно добавить в
`pyproject.toml`, обновить `uv.lock` и объяснить, для какого runtime-контура она
нужна.

## Локальные модели

Ожидаемая структура:

```text
models/Qwen/Qwen3-VL-Embedding-2B/
models/Qwen/Qwen3.5-4B/
```

`models/` игнорируется Git. Никогда не добавлять веса Qwen в коммит или
submission ZIP.

Локально `run.py` автоматически использует `./models`. На проверке модели
предоставляются через `SHARED_MODELS_PATH`.

## Инварианты pipeline

Следующие условия нельзя нарушать без отдельного обоснования и измерения:

1. Train и inference используют один preprocessing текста.
2. Train и inference используют один pixel preset. Текущий default — `S`.
3. Порядок embeddings должен точно соответствовать manifest-файлу `id`.
4. Точные текстовые дубли не должны попадать в разные folds основного CV.
5. F1 считается отдельно по категориям, затем усредняется без весов.
6. Threshold подбирается отдельно для каждой категории только по OOF
   probabilities.
7. Финальный classifier обучается на всём train после OOF-оценки.
8. `run.py`, локальное обучение и submission используют одну линейную модель в
   двух эквивалентных форматах: `.joblib` для sklearn и `.npz` для переносимого
   inference. При наличии NPZ submission предпочитает его.
9. Комментарий не влияет на класс и не может изменить verdict.
10. Output должен содержать ровно столбцы `id` и `result`.

Формат `result`:

```text
<комментарий>текст длиной 50–300 символов<вердикт>бан
```

или:

```text
<комментарий>текст длиной 50–300 символов<вердикт>не бан
```

## Основные команды

Unit-тесты:

```bash
uv run python -m unittest discover -v
```

Короткий MPS smoke test:

```bash
uv run --extra submission python -m scripts.extract_embeddings_mps \
  --model models/Qwen/Qwen3-VL-Embedding-2B \
  --limit 2 \
  --batch-size 1 \
  --pixel-preset S \
  --output cache/qwen3_vl_mps_smoke.npy
```

Полные train embeddings:

```bash
uv run --extra submission python -m scripts.extract_embeddings_mps \
  --model models/Qwen/Qwen3-VL-Embedding-2B \
  --batch-size 2 \
  --pixel-preset S \
  --output cache/qwen3_vl_mps.npy
```

OOF и обучение classifier:

```bash
uv run python -m scripts.run_embedding_baseline \
  --embeddings cache/qwen3_vl_mps.npy \
  --experiment-id E01_qwen_embedding_s
```

Group-aware поиск гиперпараметров:

```bash
uv run python -m scripts.run_embedding_baseline \
  --embeddings cache/qwen3_vl_mps.npy \
  --experiment-id E02_qwen_s_grid \
  --c-grid 0.01 0.1 1 10 \
  --search-normalization
```

Полный локальный inference:

```bash
uv run --extra submission python run.py \
  --test-data-path /path/to/test.csv \
  --output-path /path/to/submission.csv
```

Валидация CSV:

```bash
uv run python -m scripts.validate_submission \
  --input /path/to/test.csv \
  --output /path/to/submission.csv
```

Сборка submission:

```bash
uv run python -m scripts.build_submission
```

Готовый архив:

```text
dist/quality-control-submission.zip
```

## Структура кода

```text
run.py                              локальный и submission entrypoint
src/constants.py                    общие defaults и имена моделей
src/utils_data_prep.py              чтение и подготовка данных
src/utils_embed_cuda.py             embeddings на MPS/CUDA/CPU
src/utils_logreg.py                 category-specific LogisticRegression
src/evaluation.py                   folds, метрики и thresholds
src/utils_generate_cuda.py          комментарии через Qwen3.5
src/utils_postprocess.py             финальный формат result
src/wandb_tracking.py               опциональная интеграция WandB
scripts/extract_embeddings_mps.py   кеширование embeddings
scripts/run_embedding_baseline.py   OOF и обучение artifact
scripts/validate_submission.py      проверка CSV
scripts/build_submission.py         минимальный ZIP по allowlist
```

## Правила изменения кода

- Не возвращать удалённый TF-IDF baseline в основной pipeline без явного
  запроса. TF-IDF cascade исследуется отдельно в ветке `ultra`.
- Не дублировать preprocessing в scripts: использовать `src/utils_data_prep.py`.
- Не создавать несколько конкурирующих classifier моделей в корне. Joblib и
  NPZ с одинаковым базовым именем считаются двумя представлениями одной модели.
- Не хардкодить локальные абсолютные пути.
- Не добавлять модели, embeddings, reports, cache, WandB data и временные файлы
  в Git.
- Не менять смысл `0/1` и `бан/не бан`.
- Не смешивать threshold `0.5` с category-specific thresholds из artifact.
- Не подбирать thresholds на train predictions; использовать OOF probabilities.
- Не менять одновременно preprocessing, embedding settings и классификатор без
  отдельного эксперимента: иначе нельзя понять источник изменения метрики.
- Для неизвестной категории завершаться с понятной ошибкой, а не выдавать
  скрытый fallback verdict.
- Сохранять lazy imports тяжёлых GPU-зависимостей там, где CLI должен работать в
  базовом окружении.

## Работа с экспериментами

Каждому эксперименту задавать уникальный и содержательный `--experiment-id`:

```text
E01_qwen_s_c1_seed42
E02_qwen_s_c03_seed42
E03_qwen_s_c3_seed42
```

Фиксировать минимум:

- источник embeddings;
- pixel preset;
- batch size извлечения;
- `C`;
- random seed;
- количество folds;
- Mean F1 и метрики по категориям;
- thresholds;
- runtime.

При WandB-запуске использовать experiment ID как имя run и добавлять tags для
ключевого изменения.

## Артефакты

После `scripts.run_embedding_baseline` создаются:

```text
reports/<experiment-id>.json
reports/oof/<experiment-id>.csv
reports/folds.csv
reports/experiments.csv
baseline_qwen3vl_bf16.joblib
baseline_qwen3vl_bf16.npz
artifacts/backups/*.joblib
artifacts/backups/*.npz
```

Скрипт обучения заменяет рабочий classifier и предварительно создаёт backup.
Не запускать его на непроверенных embeddings без понимания, что будет обновлён
artifact для следующей сборки submission.

## Минимальная проверка перед завершением задачи

Для изменений обычного Python-кода:

```bash
uv run python -m unittest discover -v
git diff --check
```

Для изменений `run.py`, `src/` submission-модулей или сборщика дополнительно:

```bash
uv run python -m scripts.build_submission
unzip -l dist/quality-control-submission.zip
```

Для изменений preprocessing или embedding-кода дополнительно выполнить MPS
smoke test хотя бы на одной реальной карточке с изображением.

Для изменений output formatting дополнительно запустить
`scripts.validate_submission` на тестовом CSV.

## Известное текущее ограничение

Локальный и submission runtime технически проверены, но текущий
`baseline_qwen3vl_bf16.joblib` был создан до последних изменений preprocessing.
Перед оценкой качества нужно получить полные embeddings текущим кодом, выполнить
новый OOF-эксперимент и переобучить artifact.
