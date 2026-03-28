# Feature Importance Analysis - Quick Start

## Описание

Скрипт для анализа важности признаков методом перестановки (Permutation Importance).
Помогает выявить неинформативные уровни стакана для оптимизации входного тензора.

## Быстрый старт

```bash
# Базовый запуск
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt

# С дополнительными параметрами
python python_lab/scripts/feature_importance.py \
  --symbol FARTCOINUSDT \
  --model_path bots/FARTCOINUSDT/model/best_model.ckpt \
  --n_repeats 10 \
  --batch_size 512 \
  --device cuda
```

## Выходные файлы

После выполнения в `bots/{SYMBOL}/model/` будут созданы:

1. `feature_importance.json` - результаты анализа
2. `feature_importance_bar.png` - топ-20 признаков
3. `lob_importance_heatmap.png` - heatmap 50 уровней × 4 канала

## Интерпретация

- **Положительная важность**: Признак информативен
- **Нулевая/отрицательная**: Признак можно удалить
- **Яркие области на heatmap**: Важные уровни стакана

## Подробная документация

См. `python_lab/FEATURE_IMPORTANCE_GUIDE.md`
