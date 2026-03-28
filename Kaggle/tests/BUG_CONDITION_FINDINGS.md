# Bug Condition Exploration Test - Найденные контрпримеры

## Обзор

Тест `test_bug_conditions.py` выявляет все 9 категорий критических ошибок в python_lab на неисправленном коде.

**ОЖИДАЕМЫЙ РЕЗУЛЬТАТ**: Тест ПРОВАЛИВАЕТСЯ (это подтверждает наличие ошибок)

---

## Найденные ошибки и контрпримеры

### Ошибка 1: Неправильные пути импортов в calibrate.py

**Файл**: `python_lab/calibrate.py`  
**Строка**: 26  
**Проблема**: Использование `from src.train import` без добавления src в sys.path

```python
# ОШИБОЧНЫЙ КОД (строка 26):
from src.train import LiTModule
from src.dataset import LOBDataset, LOBDataLoader
from src.features import FeatureEngineer
from src.labels import Labeler
from src.normalization import Normalizer
```

**Контрпример**: При попытке импорта файла возникает `ModuleNotFoundError: No module named 'src'`

**Тест**: `test_bug_condition_1_import_errors_calibrate`

---

### Ошибка 2: Неправильные пути импортов в evaluate.py

**Файл**: `python_lab/evaluate.py`  
**Строка**: 9-15  
**Проблема**: Использование `from src.` импортов без добавления src в sys.path

```python
# ОШИБОЧНЫЙ КОД (строки 9-15):
from src.train import LiTModule
from src.dataset import LOBDataset, LOBPyTorchDataset
from src.features import FeatureEngineer
from src.labels import Labeler
from src.normalization import Normalizer
from src.utils import calculate_uncertainty
from src.interpretability import run_shap_analysis, prune_features, plot_shap_results
```

**Контрпример**: При попытке импорта файла возникает `ModuleNotFoundError: No module named 'src'`

**Тест**: `test_bug_condition_1_import_errors_evaluate`

---

### Ошибка 3: Жёстко закодированный путь в test_patching_fix.py

**Файл**: `python_lab/test_patching_fix.py`  
**Строка**: 9  
**Проблема**: Использование жёстко закодированного пути `'python_lab/src'` вместо кроссплатформенного пути

```python
# ОШИБОЧНЫЙ КОД (строка 9):
sys.path.insert(0, 'python_lab/src')
```

**Контрпример**: 
- На Windows: Путь не работает, так как использует `/` вместо `\`
- На Linux/macOS: Путь работает только если проект находится в определённом месте
- При перемещении проекта: Путь становится неправильным

**Тест**: `test_bug_condition_2_hardcoded_path`

---

### Ошибка 4: Несуществующие версии пакетов в requirements.txt

**Файл**: `python_lab/requirements.txt`  
**Проблема**: Требуемые версии пакетов никогда не были выпущены

```
# ОШИБОЧНЫЕ ВЕРСИИ:
scipy>=1.17.0          # Реально существует только до 1.12.x
matplotlib>=3.10.8     # Реально существует только до 3.9.x
plotly>=6.2.0          # Реально существует только до 5.x.x
seaborn>=0.13.0        # Реально существует только до 0.13.x
```

**Контрпримеры**:
- `scipy>=1.17.0`: Версия 1.17.0 не существует (последняя 1.12.x)
- `matplotlib>=3.10.8`: Версия 3.10.8 не существует (последняя 3.9.x)
- `plotly>=6.2.0`: Версия 6.2.0 не существует (последняя 5.x.x)
- `seaborn>=0.13.0`: Версия 0.13.0 не существует (последняя 0.12.x)

**Тест**: `test_bug_condition_3_incompatible_versions`

---

### Ошибка 5: Пустой файл src/types.py

**Файл**: `python_lab/src/types.py`  
**Проблема**: Файл не содержит необходимых типов для проекта

```python
# ОШИБОЧНЫЙ КОД (текущее содержимое):
# Common Types
```

**Контрпример**: Отсутствие типов `ArrayLike`, `PathLike` и импортов для типизации

**Тест**: `test_bug_condition_4_empty_types_file`

---

### Ошибка 6: Утечка данных в labels.py

**Файл**: `python_lab/src/labels.py`  
**Строка**: 61  
**Проблема**: Использование `fill_null(strategy="backward")` заполняет пропуски будущими значениями

```python
# ОШИБОЧНЫЙ КОД (строка 61):
returns_std=pl.col("mid_price").pct_change().rolling_std(window_size=self.window).fill_null(strategy="backward")
```

**Контрпример**: 
- Если есть пропуск в позиции i, `backward` заполняет его значением из позиции i+1 (будущее значение)
- Это приводит к утечке информации из будущего в прошлое
- Модель получает информацию, которая не должна быть доступна в момент времени i

**Тест**: `test_bug_condition_5_data_leakage_labels`

---

### Ошибка 7: Отсутствие защиты от None в normalization.py

**Файл**: `python_lab/src/normalization.py`  
**Строка**: 118  
**Проблема**: Условие `if winsor_limits` не проверяет на None перед доступом к элементам

```python
# ОШИБОЧНЫЙ КОД (строка 118):
wlows = np.quantile(data, winsor_limits[0], axis=0) if winsor_limits else None
```

**Контрпример**: 
- Если `winsor_limits = None`, условие `if winsor_limits` вычисляется как `False`
- Но если `winsor_limits = []` (пустой список), условие вычисляется как `False`
- Однако если `winsor_limits = [None]`, условие вычисляется как `True`, и попытка доступа к `winsor_limits[0]` вернёт `None`
- Затем `np.quantile(data, None, axis=0)` вызовет ошибку

**Тест**: `test_bug_condition_6_none_protection_normalization`

---

### Ошибка 8: Некорректная compute_intensity в dataset.py

**Файл**: `python_lab/src/dataset.py`  
**Строка**: 145  
**Проблема**: Функция использует счетчик вместо временных меток

```python
# ОШИБОЧНЫЙ КОД (строка 145):
def compute_intensity(timestamps, window=1000):
    n = len(timestamps)
    intensity = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        # Интенсивность = количество событий в окне
        intensity[i] = i - start_idx + 1  # ← ОШИБКА: использует счетчик, не временные метки!
    
    return intensity
```

**Контрпример**: 
- Функция должна рассчитывать интенсивность на основе временных меток
- Вместо этого она просто возвращает счетчик (1, 2, 3, ..., n)
- Это не отражает реальную интенсивность событий во времени

**Тест**: `test_bug_condition_7_compute_intensity_no_timestamps`

---

### Ошибка 9: Отсутствие защиты от деления на ноль в features.py

**Файл**: `python_lab/src/features.py`  
**Строка**: 32-36  
**Проблема**: Вычисление mid_price не проверяет на ноль для ask_p_0 и bid_p_0

```python
# ОШИБОЧНЫЙ КОД (строки 32-36):
df = df.with_columns(
    mid_price=(pl.col("ask_p_0") + pl.col("bid_p_0")) / 2
).with_columns(
    mid_price=pl.when(pl.col("mid_price") == 0).then(1.0).otherwise(pl.col("mid_price"))
)
```

**Контрпример**: 
- Если `ask_p_0 = 0` и `bid_p_0 = 0`, то `mid_price = 0 / 2 = 0`
- Проверка `if mid_price == 0` заменяет результат на 1.0
- Но это не защищает от случаев, когда `ask_p_0 = 0` или `bid_p_0 = 0` по отдельности
- Позже в коде может быть деление на `ask_p_0` или `bid_p_0`, что вызовет ошибку

**Тест**: `test_bug_condition_8_division_by_zero_protection`

---

### Ошибка 10: Скрытие ошибок при импорте в dataset.py

**Файл**: `python_lab/src/dataset.py`  
**Строка**: 1543  
**Проблема**: Использование `except: pass` скрывает реальные ошибки

```python
# ОШИБОЧНЫЙ КОД (строка 1543):
try:
    loader = LOBDataLoader("bots/BTCUSDT/data/raw", "BTCUSDT")
    print(loader.load_data().select(["timestamp_ms", "ask_p_0", "bid_p_0"]).head())
except: pass
```

**Контрпример**: 
- Если возникает ошибка (например, файл не найден), она молча игнорируется
- Разработчик не видит, что что-то пошло не так
- Это затрудняет отладку и скрывает реальные проблемы

**Тест**: `test_bug_condition_9_exception_hiding_import`

---

## Резюме

Все 9 категорий критических ошибок подтверждены в неисправленном коде:

| № | Ошибка | Файл | Строка | Статус |
|---|--------|------|--------|--------|
| 1 | Неправильные импорты (calibrate.py) | calibrate.py | 26 | ✓ Найдена |
| 2 | Неправильные импорты (evaluate.py) | evaluate.py | 9-15 | ✓ Найдена |
| 3 | Жёстко закодированный путь | test_patching_fix.py | 9 | ✓ Найдена |
| 4 | Несовместимые версии | requirements.txt | - | ✓ Найдена |
| 5 | Пустой types.py | src/types.py | - | ✓ Найдена |
| 6 | Утечка данных | src/labels.py | 61 | ✓ Найдена |
| 7 | Отсутствие защиты от None | src/normalization.py | 118 | ✓ Найдена |
| 8 | Некорректная compute_intensity | src/dataset.py | 145 | ✓ Найдена |
| 9 | Отсутствие защиты от деления на ноль | src/features.py | 32-36 | ✓ Найдена |
| 10 | Скрытие ошибок при импорте | src/dataset.py | 1543 | ✓ Найдена |

**ВЫВОД**: Тест `test_bug_conditions.py` успешно выявляет все критические ошибки на неисправленном коде.
