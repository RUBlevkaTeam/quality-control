# E-CUP 2026 Quality Control

Текущая ветка `main` развивает baseline небольшими измеримыми шагами. Public
точка отсчёта перед нативным retrieval: `0.7053364806`.

## Текущий inference

1. БАД: category-specific word TF-IDF + 38 rule-признаков;
2. легковоспламеняющиеся: word TF-IDF + char_wb 3–6 по карточке + отдельный
   char_wb 2–6 по названию + те же rule-признаки;
3. точный SHA-256 всех изображений через нативный OpenSSL (`hashlib`);
4. dHash-256 с пакетным C++ Hamming top-k;
5. только консервативные retrieval-overrides, прошедшие 70/30 stress-CV;
6. Qwen embedding fallback и Qwen3.5-комментарии из официального baseline.

Нативный `x86_64` ELF не зависит от `libstdc++`, OpenSSL или компилятора в
контейнере. Если он не загрузится, код автоматически использует точный NumPy
fallback и не роняет submission.

## Измерения image retrieval

- 49 403 уникальных внутри товара train-изображения;
- C++ поиск dHash по всему train: около `0.58 с` локально;
- ускорение Hamming top-k относительно NumPy: примерно `65x`;
- leave-one-product-out: БАД `0.9872` accuracy на 16.8% покрытия,
  легковоспламеняющиеся `1.0000` на 1.4% покрытия;
- пять selection stress-folds: средний F1 `0.89750 → 0.89843`;
- три независимых confirmation-fold: `0.87076 → 0.87130`.

Прирост небольшой, поэтому агрессивные exact-image overrides для редкого класса
отключены: высокая standalone accuracy там ухудшала F1 поверх текстовой модели.
Полный отчёт находится в `reports/image_retrieval_loo.json`.

## Rich text head для редкого класса

На одинаковых dedup + family-disjoint фолдах новая голова категории
`Легковоспламеняющиеся` дала F1 `0.7612 → 0.7893`. На трёх дополнительных
seed средний F1 вырос `0.78269 → 0.79248`; БАД намеренно оставлен на прежней
голове, её вероятности после пересборки совпадают с предыдущим артефактом.

## Сборка

```bash
.venv/bin/python -m src.text_model data.csv text_model.joblib families.csv
.venv/bin/python scripts/build_native.py --local --linux-x86-64
.venv/bin/python scripts/build_image_retrieval.py --workers 4
.venv/bin/python scripts/evaluate_image_retrieval.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/build_submit.py
```

Для Linux-кросс-сборки нужен только development-пакет `ziglang`; в evaluator
он не устанавливается. Отправлять нужно готовый `submit.zip` целиком.
