# Задача 095: Анализ распределения меток (Label Distribution Analysis)

**Цель**: Реализовать статистический анализ распределения классов (Up/Down/Flat) и матрицы переходов. Это необходимо для калибровки весов потерь (задача 052) и детекции дисбаланса, где `Flat` может занимать >90% данных.

## 1. Изменения в Python-лаборатории

### 1.1. Создание [./python_lab/src/utils.py](./python_lab/src/utils.py) (или обновление)
Добавить функцию `analyze_labels(df: pl.DataFrame)`, реализующую следующую логику:
*   **Распределение**: 
    ```python
    counts = df.group_by("label").count()
    percentages = counts.with_columns((pl.col("count") / pl.col("count").sum() * 100).alias("%"))
    # Imbalance Ratio: Max class count / Min class count
    imbalance_ratio = counts["count"].max() / counts["count"].min()
    ```
*   **Матрица переходов (Transition Matrix)**: Анализ вероятности смены состояний (например, Up -> Flat).
    ```python
    # Используем pandas для crosstab, так как это удобнее для визуализации
    import pandas as pd
    labels_ser = df["label"].to_pandas()
    transition = pd.crosstab(labels_ser.shift(1), labels_ser, normalize='index')
    ```

### 1.2. Создание [./python_lab/scripts/analyze_labels.py](./python_lab/scripts/analyze_labels.py)
Скрипт для запуска анализа из командной строки:
*   **Аргументы**: `--data_path` (путь к parquet-файлам).
*   **Действия**: Загрузка данных через `polars`, вызов `analyze_labels`, сохранение итогов в `metadata.json` и генерация графиков.
*   **Визуализация**: 
    *   Гистограмма частот классов (`sns.barplot`).
    *   Тепловая карта (Heatmap) матрицы переходов (`sns.heatmap`).
    *   **Важно**: Удален график накопленной доходности (перенесен в задачу 058).

## 2. Почему это важно (Аргументы Grok)
*   **Severe Imbalance**: В LOB-данных класс `Flat` доминирует. Без точных цифр дисбаланса невозможно корректно настроить `pos_weight` в `CrossEntropyLoss` (задача 052).
*   **Transition Bias**: Матрица переходов показывает, не "залипает" ли модель в одном состоянии. Если `Flat -> Flat` составляет 99%, модель будет крайне консервативна.
*   **Multi-horizon (Optional)**: Если в данных есть колонка `horizon`, анализ проводится в разрезе горизонтов: `df.group_by(["horizon", "label"]).count()`.

## 3. Критические требования
*   **Библиотеки**: Использовать `polars` для расчетов, `seaborn` и `matplotlib` для графиков.
*   **Metadata**: Результаты (counts, imbalance_ratio) должны сохраняться в JSON для последующего использования в `train.py`.
*   **Zero-Division**: Предусмотреть проверку, если какой-то класс полностью отсутствует в выборке.

## 4. Тестирование
*   **Consistency**: Сравнение распределения меток в `train.parquet` и `val.parquet`. Резкие отличия (>5%) должны вызывать `warning`.
*   **Visual Verify**: Проверка heatmap: сумма строк в матрице переходов должна быть равна 1.0.

---
**Напиши следующую задачу: 096-python-lab-cross-validation-logic.md**