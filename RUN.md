# Запуск LoRA-обучения в Colab/Kaggle

Локально ничего не считается: ноутбук только заливает данные.

## Шаг 0. Залить данные (один раз, из браузера)

1. **Картинки**: папку `images/` (9 ГБ) залить как Kaggle Dataset
   (kaggle.com → New Dataset → перетащить папку) ИЛИ в Google Drive
   (drive.google.com → перетащить папку). Название датасета: `qc-images`.
2. **Мета-пак**: `qc_pack.zip` уже собран в корне репо.

## Шаг 1. Ноутбук Colab (T4) или Kaggle (P100/T4x2)

```python
# 1) зависимости
!pip -q install peft transformers accelerate scikit-learn pandas

# 2) данные
#    Kaggle: датасет qc-images примонтирован в /kaggle/input/qc-images
#    Colab:  !gdown --folder <ссылка>  или распаковка из Drive
!unzip -q qc_pack.zip -d /content/qc_pack
!ln -s /kaggle/input/qc-images/images /content/images   # путь подстроить

# 3) обучение фолда 0 (валидация)
!python train_lora_qc.py --pack /content/qc_pack --images /content/images \
    --fold 0 --epochs 1 --out /content/out
```

`train_lora_qc.py` лежит в репо (`scripts/`) — скопировать содержимое в ячейку
или загрузить через файловую панель.

Ожидаемое время на T4: ~4-7 ч на фолд (13k товаров после дедупа, батч 1 ×
accum 16). Если медленно — сначала прогнать только fire:
добавить в jsonl-фильтр по категории (скрипт поддерживает правку руками).

## Шаг 2. Забрать результаты

Скачать `out/oof_fold0.csv` и `out/adapter_fold0/` (адаптер ~100-200 МБ).

## Шаг 3. Локально (секунды, не GPU)

1. Положить `oof_fold0.csv` в `reports/`;
2. Прогнать текстовый OOF **в колабе тоже** (CPU, бесплатно):
   загрузить туда же `text_model.joblib`, запустить `dump_text_oof.py`
   → скачать `text_oof.csv` в `reports/`;
3. `venv312/bin/python scripts/tune_blend.py` → напечатает F1 бленда и
   создаст `lora_blend.json`.

## Критерий продолжения

- Средняя метрика бленда на OOF ≥ text + 0.01 → адаптер едет в сабмит №2;
- ниже → документируем отрицательный результат, остаётся план без VLM.

## Финальный адаптер

Если фолд-проверка удачна: повторить с `--train-all` (учится на всех данных),
положить `adapter_model.safetensors` + `adapter_config.json` в
`qc_lora_adapter/` репо, `lora_blend.json` — в корень,
`scripts/build_submit.py` сам упакует.
