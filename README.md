# E-CUP 2026 Quality Control — Ultra

Ветка `ultra` — отдельная экспериментальная версия поверх официального baseline.
Зафиксированная точка отсчёта: public score голого baseline
`0.5077170722469463` (commit `73b2d15f`).

## Архитектура

Решение асимметрично для двух категорий:

- **БАД**: word/char TF-IDF и признаки явной маркировки, отрицания и
  исключённых товарных семейств; точная память дублей используется как
  дополнительный сигнал.
- **Легковоспламеняющиеся**: отдельный rare-class классификатор с признаками
  комплектации (`входит`, `без баллона`, `приобретается отдельно`, встроенный
  поджиг), high-precision negative veto и собственным F1-порогом.

Общий inference-каскад:

1. нормализация HTML/Unicode и category-specific sparse text model;
2. точный one-hop retrieval по SHA-256 изображений, нормализованному названию и
   описанию без транзитивного Union-Find;
3. проверенные relation-признаки и конфликтные gates;
4. Qwen3-VL-Embedding-2B только на спорной доле товаров как консервативный
   мультимодальный tie-breaker;
5. два grounded-шаблона с одинаковым фактическим смыслом; Qwen3.5 выбирает
   только более ясный вариант `A/B`, затем выполняется строгая валидация CSV.

В отличие от baseline, сырой текст Qwen3.5 никогда не попадает в результат и
модель не влияет на класс: она получает фиксированный verdict и выбирает между
двумя заранее проверенными формулировками. Любой ответ кроме ровно `A` или `B`,
а также любой сбой, автоматически оставляет детерминированный вариант `A`.
Открытые LLM и VLM остаются частью pipeline.

## Локальная оценка

Результаты хранятся в `reports/`.

| Протокол | F1 БАД | F1 легковоспл. | Среднее |
|---|---:|---:|---:|
| 5-fold deployment-like OOF | 0.9566 | 0.8901 | 0.9233 |
| 5-fold exact-text-grouped OOF | 0.9521 | 0.8167 | 0.8844 |

Первая оценка моделирует вероятный случайный отбор test из общего пула и
разрешает законный train→test duplicate retrieval. Вторая не допускает
разделения карточек с одинаковыми нормализованными `name+description` между
фолдами и измеряет только text/rule fallback. Это не полноценный
family-disjoint тест: похожие SKU с изменённым описанием или изображением всё
ещё могут оказаться в разных фолдах. Вручную отобранные relation-признаки также
смотрели на весь train, поэтому обе цифры оптимистичны и не являются обещанием
leaderboard.

В train 49 456 изображений, но только 32 951 уникальный SHA-256; 7 084 товара
имеют хотя бы одного exact-image соседа. Поэтому точная память дублей —
измеренный сигнал, а не предположение. Perceptual hash сознательно не включён в
первую отправляемую ultra-версию: без геометрической/OCR-верификации он может
склеить разные варианты упаковки.

## Воспроизведение

Локальное обучение не требует GPU:

```bash
python -m venv .venv
.venv/bin/python -m pip install numpy pandas scipy scikit-learn pillow joblib
.venv/bin/python scripts/train_ultra.py --folds 5
.venv/bin/python scripts/evaluate_ultra_strict.py --folds 5
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/build_ultra_submission.py
```

Train-изображения ожидаются в `content/content_ecup/images/<id>/`. Кеш их
SHA-256 создаётся в `cache/` и не попадает в Git или submission.

## Submission

Готовый архив создаётся только по явному allowlist:

```text
dist/ecup_quality_ultra.zip
```

В корне ZIP находятся `metadata.json`, `run.py`, два joblib-артефакта и только
необходимые модули `src/`. Train, изображения, `.git`, кеши и macOS-мусор не
архивируются.

Официальный CLI поддержан во всех формах:

```bash
python -u run.py --test_data_path /data/test.csv --output-path /output/result.csv
python -u run.py -i /data/test.csv -o /output/result.csv
```

Используется официальный образ `odsai/ecup26-quality-baseline:1.0` и shared
модель `Qwen/Qwen3-VL-Embedding-2B`; сетевые загрузки принудительно отключены.

## Ограничения проверки

- Локально отсутствует NVIDIA GPU и Docker daemon, поэтому реальный H100
  inference официального образа необходимо подтвердить отправкой Check/Public.
- VLM rescue намеренно очень консервативен, потому что для baseline embeddings
  нет сохранённых train OOF probabilities; его пороги `0.95/0.98` отдельно не
  валидированы и могут как помочь, так и навредить.
- Empirical absence-veto идеально разделяет найденные train-примеры, но является
  dataset-specific правилом и может ошибиться на новой формулировке аксессуара.
- Любой сбой VLM оставляет полностью валидный sparse/retrieval prediction вместо
  падения всего submission.
