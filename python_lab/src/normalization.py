import polars as pl
import json
import math
import numpy as np
from pathlib import Path
from typing import Union, Dict, List

class Normalizer:
    """
    Класс для расчета и применения параметров нормализации Z-score (mean, std).
    Обеспечивает идентичность предобработки данных в Python (обучение) и Rust (инференс).
    """
    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        self.params: Dict[str, Dict[str, float]] = {}

    def fit(self, data: Union[pl.DataFrame, pl.LazyFrame, np.ndarray], feature_names: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Рассчитывает среднее, стандартное отклонение, медиану и IQR для всех признаков.
        Поддерживает Polars DataFrame/LazyFrame и Numpy arrays.
        """
        import numpy as np
        
        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            
            feat_cols = [c for c in data.columns if c.startswith("feat_")]
            
            # Эффективный расчет агрегатов через Polars
            summary = data.select([
                pl.col(c).mean().alias(f"{c}_mean") for c in feat_cols
            ] + [
                pl.col(c).std().alias(f"{c}_std") for c in feat_cols
            ])

            results = summary.to_dicts()[0]

            # Задача 240: Расчет робастных параметров (медиана и IQR)
            robust_summary = data.select([
                pl.col(c).median().alias(f"{c}_median") for c in feat_cols
            ] + [
                pl.col(c).quantile(0.25).alias(f"{c}_q25") for c in feat_cols
            ] + [
                pl.col(c).quantile(0.75).alias(f"{c}_q75") for c in feat_cols
            ])
            
            robust_results = robust_summary.to_dicts()[0]

            for c in feat_cols:
                self.params[c] = {
                    "mean": float(results[f"{c}_mean"]) if not math.isnan(results[f"{c}_mean"]) else 0.0,
                    "std": float(results[f"{c}_std"]) if not (math.isnan(results[f"{c}_std"]) or results[f"{c}_std"] == 0) else 1.0,
                    # Задача 240: Добавляем медиану и IQR
                    "median": float(robust_results[f"{c}_median"]) if not math.isnan(robust_results[f"{c}_median"]) else 0.0,
                    "iqr": float(robust_results[f"{c}_q75"] - robust_results[f"{c}_q25"]) if not math.isnan(robust_results[f"{c}_q75"] - robust_results[f"{c}_q25"]) else 1.0
                }
        
        elif isinstance(data, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names must be provided when fitting on a numpy array")
            
            if data.shape[1] != len(feature_names):
                raise ValueError(f"Data shape {data.shape} mismatch with feature_names length {len(feature_names)}")
            
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            
            # Задача 240: Расчет робастных параметров
            medians = np.median(data, axis=0)
            q25 = np.quantile(data, 0.25, axis=0)
            q75 = np.quantile(data, 0.75, axis=0)
            iqrs = q75 - q25
            
            for i, name in enumerate(feature_names):
                m = float(means[i])
                s = float(stds[i])
                med = float(medians[i])
                iqr = float(iqrs[i])
                
                self.params[name] = {
                    "mean": m if not math.isnan(m) else 0.0,
                    "std": s if not (math.isnan(s) or s == 0) else 1.0,
                    # Задача 240: Добавляем медиану и IQR
                    "median": med if not math.isnan(med) else 0.0,
                    "iqr": iqr if not math.isnan(iqr) else 1.0
                }
        
        return self.params

    def save(self):
        """Сохраняет параметры нормализации в JSON файл для последующего использования в Rust."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.params, f, indent=4)
        print(f"[{self.__class__.__name__}] Normalization params saved to {self.output_path}")

    def load(self):
        """Загружает параметры из существующего JSON файла."""
        with open(self.output_path, 'r') as f:
            self.params = json.load(f)
        print(f"[{self.__class__.__name__}] Normalization params loaded from {self.output_path}")

    def transform(self, data: Union[pl.DataFrame, pl.LazyFrame, np.ndarray]) -> Union[pl.DataFrame, pl.LazyFrame, np.ndarray]:
        """
        Применяет Z-score нормализацию: (x - mean) / std.
        Приводит результат к Float32 для совместимости с ONNX/Rust.
        Поддерживает Polars DataFrame/LazyFrame и Numpy arrays (2D и 3D).
        """
        if not self.params:
            raise ValueError("Normalizer not fitted or loaded. Call fit() or load() first.")

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            # Для Polars используем существующую логику
            exprs = [
                ((pl.col(c) - self.params[c]["mean"]) / self.params[c]["std"])
                .cast(pl.Float32)
                .alias(c)
                for c in self.params
            ]
            return data.with_columns(exprs)

        elif isinstance(data, np.ndarray):
            # Поддержка 2D (N, F) и 3D (N, S, F) массивов
            # Работаем с последней размерностью (признаки)
            if data.shape[-1] != len(self.params):
                raise ValueError(f"Data shape {data.shape} mismatch with params length {len(self.params)}")
            
            res = data.copy().astype(np.float32)

            # Предполагаем, что порядок признаков в params совпадает с порядком колонок в массиве
            for i, (name, p) in enumerate(self.params.items()):
                if data.ndim == 2:
                    res[:, i] = (data[:, i] - p["mean"]) / p["std"]
                elif data.ndim == 3:
                    res[:, :, i] = (data[:, :, i] - p["mean"]) / p["std"]
                else:
                    raise ValueError(f"Unsupported array dimension: {data.ndim}. Expected 2D or 3D.")

            return res

        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected Polars DataFrame/LazyFrame or Numpy array.")

if __name__ == "__main__":
    # Тестовый пример
    test_df = pl.DataFrame({
        "feat_price_0": [10.0, 20.0, 30.0],
        "feat_vol_0": [100.0, 200.0, 300.0],
        "other_col": [1, 2, 3]
    })
    
    norm = Normalizer("norm_test.json")
    params = norm.fit(test_df)
    print("Fitted params:", params)
    
    transformed = norm.transform(test_df)
    if isinstance(transformed, pl.DataFrame):
        print("\nTransformed data (mean should be ~0, std ~1):")
        print(transformed.select(["feat_price_0", "feat_vol_0"]))
    
    # Clean up test file
    if Path("norm_test.json").exists():
        Path("norm_test.json").unlink()
