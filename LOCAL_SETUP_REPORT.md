# Отчёт о подготовке локального и submission-контуров

Дата проверки: 12 августа 2026 года.

## Результат

Репозиторий разделён на два воспроизводимых сценария:

- локальные CPU/MPS-эксперименты с честной OOF-метрикой;
- сборка и запуск исходного Qwen submission на CUDA/H100.

Проверенный полный text baseline (`E01_tfidf`, 5 folds):

| Метрика | Значение |
|---|---:|
| F1 БАД | 0.939908 |
| F1 Легковоспламеняющиеся | 0.837466 |
| Macro F1 | **0.888687** |
| Threshold БАД | 0.35 |
| Threshold Легковоспламеняющиеся | 0.64 |

Это локальная group-aware OOF-оценка, а не leaderboard score.

## Изменения по файлам

### Окружение

- `pyproject.toml` — зависимости разделены на базовые, notebook и submission;
  указаны поддерживаемые версии Python и конфигурация тестов.
- `uv.lock` — зафиксированы точные версии всех зависимостей.
- `.gitignore` — исключены локальные модели, кэши embeddings и отчёты запусков.

### Локальная оценка

- `src/metrics.py` — соревновательная F1 по категориям, Macro F1, подбор и
  применение отдельных thresholds.
- `src/folds.py` — нормализация дублей и детерминированный
  `StratifiedGroupKFold`, не допускающий пересечения одной группы между folds.
- `src/text_baseline.py` — очистка HTML, объединённый word/char TF-IDF и
  отдельная logistic regression по каждой категории.
- `scripts/run_text_baseline.py` — полный OOF-цикл, обучение финальной модели,
  сохранение folds, probabilities, ошибок, метрик и experiment registry.
- `scripts/predict_text_baseline.py` — локальный inference сохранённой text
  модели и формирование submission-shaped CSV.

### MPS

- `scripts/extract_embeddings_mps.py` — явный MPS-only extractor Qwen
  embeddings, безопасные настройки для Mac 24 ГБ, сохранение `.npy`, ID manifest
  и metadata.
- `scripts/run_embedding_baseline.py` — group-aware OOF logistic regression на
  сохранённых Qwen embeddings с той же метрикой и thresholds.
- `src/utils_embed_cuda.py` — добавлен выбор `auto/cuda/mps/cpu`, корректная
  очистка MPS cache, обработка MPS/CUDA OOM и исправлено накопление нескольких
  batch embeddings.

### Submission

- `run.py` — поддержаны все варианты CLI-аргументов; тяжёлые импорты перенесены
  после argparse; добавлены проверки входа, изображений, моделей и artifact;
  пути больше не зависят от текущей директории.
- `src/utils_data_prep.py` — удаление `Unnamed:*`, проверка `id`, fallback для
  отсутствующих текстовых колонок и фильтрация только настоящих image files.
- `scripts/validate_submission.py` — проверка колонок, ID, формата тегов,
  бинарного вердикта и длины комментария 50–300 символов.
- `scripts/build_submission.py` — создаёт минимальный ZIP только из файлов,
  необходимых evaluator; train, изображения и локальные модули не включаются.

### Тесты и документация

- `tests/test_metrics.py` — тесты идеальной/нулевой F1, Macro averaging,
  некорректных входов и thresholds.
- `tests/test_folds.py` — тесты отсутствия group leakage и уникальности ID.
- `tests/test_submission.py` — тесты корректного CSV и отклонения короткого
  комментария.
- `README.md` — полная инструкция: установка, quick/full baseline, MPS,
  локальный inference, валидация и сборка submission.
- `scripts/__init__.py`, `tests/__init__.py` — запуск CLI и тестов как Python
  modules.

### WandB

- `src/wandb_tracking.py` — opt-in WandB adapter, единые metric keys, fold/final
  logging и versioned report/model artifacts.
- `scripts/setup_wandb.py` — безопасная команда авторизации и offline smoke test;
  API key не принимается аргументом и не записывается в репозиторий; команда
  явно показывает UI-адрес `https://wandb.ai/authorize` и отличает его от API
  endpoint `https://api.wandb.ai`.
- `scripts/log_report_wandb.py` — загрузка уже рассчитанного JSON/OOF-отчёта на
  сайт без повторного обучения.
- `tests/test_wandb_tracking.py` — проверка имён dashboard metrics и flattening
  итоговых значений без сетевых вызовов.
- `scripts/run_text_baseline.py` и `scripts/run_embedding_baseline.py` — добавлены
  флаги `--wandb*`, логирование каждого fold, итоговой F1, thresholds, runtime,
  ошибок и artifacts.

## Что проверено

- 11 unit-тестов проходят;
- `run.py --help` работает без установленных Qwen-моделей;
- quick OOF baseline: Macro F1 `0.866127` за 59.8 секунды;
- full 5-fold OOF baseline: Macro F1 `0.888687`;
- локальный CSV создан на 12 971 товар и принят валидатором;
- submission ZIP собирается и проходит проверку целостности;
- MPS доступен вне sandbox, PyTorch matrix smoke test прошёл.
- WandB 0.28.2 offline setup и полный quick experiment с artifacts завершились
  с exit code 0; online-синхронизация требует личной авторизации владельца.

## Внешний артефакт, которого пока нет

Настоящее извлечение мультимодальных embeddings не запускалось, потому что в
репозитории отсутствует директория весов `Qwen3-VL-Embedding-2B`. После её
появления сначала нужно выполнить MPS smoke test на 10 строках по инструкции в
`README.md`, затем извлечь полный кэш.
