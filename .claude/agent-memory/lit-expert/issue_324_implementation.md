# Задача 324: Стабилизация dynamic feature contract

## Дата: 2026-03-25

## Реализованные изменения

### 1. Dataset (dataset.py)

**Новые функции:**
- `compute_delta_imb_from_lob_cache()` - event-consistent расчет DeltaImb
- `compute_delta_spread_from_lob_cache()` - event-consistent расчет DeltaSpread

**Изменения в `LOBDataset._init_memory_mode`:**
- Добавлены кэши: `self.delta_imb_cache` и `self.delta_spread_cache`
- Расчет использует `update_id_raw` для event-consistency (только при обновлении стакана)
- Восстановление сырых объемов из log1p перед расчетом Imbalance

**Изменения в `_calculate_6_channels_raw`:**
- Добавлены параметры `di_precomputed` и `ds_precomputed`
- DeltaImb/DeltaSpread берутся из кэшей вместо `torch.diff`
- Fallback на `torch.diff` если кэши недоступны

**Изменения в `_compute_channels_for_normalization`:**
- Получение `di_precomp` и `ds_precomp` из кэшей
- Передача в `_calculate_6_channels_raw`

**Изменения в `_process_sample`:**
- Получение `di_precomp` и `ds_precomp` из кэшей по индексу `idx:idx+seq_len`
- Передача в `_calculate_6_channels_raw`

### 2. Train Data (train_data.py)

**Изменения в `_fit_normalizer_on_train`:**
- Добавлена **агрегированная диагностика** по train split после fit:
  - min, max, mean, std
  - p1, p50, p99
  - saturation (±4.0), zero%
  - Выводится один раз перед стартом обучения

- Добавлен **hard guard** (защита от плохого scale):
  - Проверяет saturation > 10%
  - Проверяет zero% > 95%
  - Проверяет range (p99-p01) < 0.01
  - При неудаче выбрасывает RuntimeError
  - Есть флаг `--allow-bad-dynamic-scale` для override

### 3. CLI (train_cli.py)

**Новый аргумент:**
- `--allow-bad-dynamic-scale` (store_true) - разрешает обучение при плохом качеству scale

### 4. Normalizer (normalization.py)

**Без изменений** - уже поддерживает `dynamic_params` и их сериализацию.

## Контракт dynamic-каналов

- **Построение**: все три канала (OFI, DeltaImb, DeltaSpread) строятся из **одного event-consistent источника** с учетом `update_id_raw`
- **Fit нормализатора**: используется полное train распределение (все значения из кэшей), а не три прокси-столбца
- **Runtime pipeline**: идентичен fit pipeline через общий `_calculate_6_channels_raw`
- **Диагностика**: агрегированная по всему train split (не sample-local)
- **Guard**: автоматическая остановка при явно плохом scale

## Ожидаемый эффект

- Снижение saturation для OFI (было ~29%), DeltaImb (~33%), DeltaSpread (~16%)
- Улучшение MCC и Directional Accuracy
- Устойчивый микроструктурный сигнал без артефактов

## Критерии acceptance

- Все три канала согласованы по event-time контракту
- Fit normalizer использует полное train распределение
- Есть агрегированная диагностика перед обучением
- Guard останавливает заведомо плохие запуски
- Повторный запуск показывает улучшение метрик без изменения архитектуры LiT

## Следующие шаги

1. Провести повторный запуск обучения (324.6)
2. Зафиксировать метрики до/после в `docs/train_logs.md`
3. Провести ablation эксперимент (324.7)
4. Отправить изменения через giter
