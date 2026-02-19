# Задача 240: Устойчивое масштабирование и винзоризация (Robust Scaler & Winsorization)

Реализация продвинутых методов нормализации данных в `python_lab`, устойчивых к аномальным выбросам (Outliers) и «спуфингу» в стакане. Это критично для предотвращения смещения весов модели при резких скачках волатильности.

## 1. Цель задачи
Добавить поддержку `RobustScaler` (медиана и IQR) и предварительной **винзоризации** (клиппинг экстремумов) в пайплайн подготовки признаков. Обеспечить переключение между методами через конфигурацию.

## 2. Инструкции по реализации для Gemini

### А. Python: Расширение [./python_lab/src/dataset.py](./python_lab/src/dataset.py)
1.  **Функция Winsorization**:
    Реализовать ограничение экстремальных значений (например, 1-й и 99-й перцентили) через нативные средства **Polars**:
    ```python
    def apply_winsorization(df: pl.DataFrame, limits: tuple[float, float] = (0.01, 0.99)) -> pl.DataFrame:
        cols = df.columns
        for c in cols:
            low = df[c].quantile(limits[0])
            high = df[c].quantile(limits[1])
            df = df.with_columns(pl.col(c).clip(lower=low, upper=high))
        return df
    ```

2.  **Robust Scaling (Fit & Apply)**:
    Использовать медиану и межквартильный размах (IQR) вместо среднего и стандартного отклонения:
    ```python
    def fit_robust_params(df_train: pl.DataFrame) -> dict:
        med = df_train.median().to_dicts()[0]
        q25 = df_train.quantile(0.25).to_dicts()[0]
        q75 = df_train.quantile(0.75).to_dicts()[0]
        iqr = {k: q75[k] - q25[k] for k in med}
        return {"type": "robust", "median": med, "iqr": iqr}

    def apply_robust_scaling(df: pl.DataFrame, params: dict) -> pl.DataFrame:
        eps = 1e-8
        return df.with_columns([
            ((pl.col(c) - params['median'][c]) / (params['iqr'][c] + eps)).alias(c)
            for c in df.columns
        ])
    ```

### Б. Rust: Обновление [./src/ml/tensor.rs](./src/ml/tensor.rs)
1.  Обновить логику нормализации тензора, чтобы она поддерживала разные типы скейлеров на основе метаданных модели:
    ```rust
    pub fn apply_normalization(tensor: &mut Array2<f32>, meta: &ModelMetadata) {
        match meta.scaler_type.as_str() {
            "zscore" => { /* существующая логика из 034 */ },
            "robust" => {
                // tensor[i, j] = (val - median[j]) / (iqr[j] + eps)
            },
            _ => panic!("Unknown scaler type"),
        }
    }
    ```

## 3. Конфигурация
Добавить в `DatasetConfig`:
-   **scaler_type**: `"zscore"`, `"robust"`, `"winsor_robust"` (сначала клиппинг, затем Robust).
-   **winsor_limits**: `[0.01, 0.99]`.

## 4. Ожидаемый результат
1.  Модель успешно обучается на данных с «грязными» выбросами, не теряя точности на основном распределении.
2.  Параметры нормализации (`median`, `iqr`) корректно сохраняются в `metadata.json` (задача 056).
3.  Rust-ядро применяет выбранный метод масштабирования с околонулевыми накладными расходами.

## 5. Необходимые зависимости
-   **Python**: `polars >= 0.19`, `numpy`.
-   **Rust**: `ndarray`, `serde_json`.