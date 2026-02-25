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
        self.scaler_type = "zscore"
        self.winsor_limits = None

    def fit(self, data: Union[pl.DataFrame, pl.LazyFrame, np.ndarray], feature_names: List[str] = None, winsor_limits: List[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Рассчитывает среднее, стандартное отклонение, медиану, IQR и границы винзоризации.
        Поддерживает Polars DataFrame/LazyFrame и Numpy arrays.
        """
        import numpy as np
        
        self.winsor_limits = winsor_limits

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            
            feat_cols = [c for c in data.columns if c.startswith("feat_")]
            
            # Эффективный расчет агрегатов через Polars
            summary_exprs = [
                pl.col(c).mean().alias(f"{c}_mean") for c in feat_cols
            ] + [
                pl.col(c).std().alias(f"{c}_std") for c in feat_cols
            ] + [
                pl.col(c).median().alias(f"{c}_median") for c in feat_cols
            ] + [
                pl.col(c).quantile(0.25).alias(f"{c}_q25") for c in feat_cols
            ] + [
                pl.col(c).quantile(0.75).alias(f"{c}_q75") for c in feat_cols
            ]

            if winsor_limits:
                summary_exprs += [
                    pl.col(c).quantile(winsor_limits[0]).alias(f"{c}_wlow") for c in feat_cols
                ] + [
                    pl.col(c).quantile(winsor_limits[1]).alias(f"{c}_whigh") for c in feat_cols
                ]

            summary = data.select(summary_exprs)
            results = summary.to_dicts()[0]

            for c in feat_cols:
                q25 = results[f"{c}_q25"]
                q75 = results[f"{c}_q75"]
                
                self.params[c] = {
                    "mean": float(results[f"{c}_mean"]) if not math.isnan(results[f"{c}_mean"]) else 0.0,
                    "std": float(results[f"{c}_std"]) if not (math.isnan(results[f"{c}_std"]) or results[f"{c}_std"] == 0) else 1.0,
                    "median": float(results[f"{c}_median"]) if not math.isnan(results[f"{c}_median"]) else 0.0,
                    "iqr": float(q75 - q25) if not math.isnan(q75 - q25) else 1.0
                }
                
                if winsor_limits:
                    self.params[c]["winsor_low"] = float(results[f"{c}_wlow"])
                    self.params[c]["winsor_high"] = float(results[f"{c}_whigh"])
        
        elif isinstance(data, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names must be provided when fitting on a numpy array")
            
            if data.shape[1] != len(feature_names):
                raise ValueError(f"Data shape {data.shape} mismatch with feature_names length {len(feature_names)}")
            
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            medians = np.median(data, axis=0)
            q25s = np.quantile(data, 0.25, axis=0)
            q75s = np.quantile(data, 0.75, axis=0)
            
            wlows = np.quantile(data, winsor_limits[0], axis=0) if winsor_limits else None
            whighs = np.quantile(data, winsor_limits[1], axis=0) if winsor_limits else None
            
            for i, name in enumerate(feature_names):
                self.params[name] = {
                    "mean": float(means[i]) if not math.isnan(means[i]) else 0.0,
                    "std": float(stds[i]) if not (math.isnan(stds[i]) or stds[i] == 0) else 1.0,
                    "median": float(medians[i]) if not math.isnan(medians[i]) else 0.0,
                    "iqr": float(q75s[i] - q25s[i]) if not math.isnan(q75s[i] - q25s[i]) else 1.0
                }
                
                if winsor_limits:
                    self.params[name]["winsor_low"] = float(wlows[i])
                    self.params[name]["winsor_high"] = float(whighs[i])
        
        return self.params

    def save(self, scaler_type: str = "zscore", winsor_limits: List[float] = None):
        """Сохраняет параметры нормализации в JSON файл для последующего использования в Rust."""
        self.scaler_type = scaler_type
        self.winsor_limits = winsor_limits
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "params": self.params,
            "scaler_type": self.scaler_type,
            "winsor_limits": self.winsor_limits
        }
        with open(self.output_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"[{self.__class__.__name__}] Normalization params saved to {self.output_path} (Type: {self.scaler_type})")

    def load(self):
        """Загружает параметры из существующего JSON файла."""
        with open(self.output_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and "params" in data:
                self.params = data["params"]
                self.scaler_type = data.get("scaler_type", "zscore")
                self.winsor_limits = data.get("winsor_limits")
            else:
                # Compatibility with old format
                self.params = data
                self.scaler_type = "zscore"
                self.winsor_limits = None
                
        print(f"[{self.__class__.__name__}] Normalization params loaded from {self.output_path} (Type: {self.scaler_type})")

    def transform(self, data: Union[pl.DataFrame, pl.LazyFrame, np.ndarray]) -> Union[pl.DataFrame, pl.LazyFrame, np.ndarray]:
        """
        Применяет нормализацию в соответствии с scaler_type.
        Приводит результат к Float32 для совместимости с ONNX/Rust.
        Поддерживает Polars DataFrame/LazyFrame и Numpy arrays (2D и 3D).
        """
        if not self.params:
            raise ValueError("Normalizer not fitted or loaded. Call fit() or load() first.")

        # Задача 240: Предварительная винзоризация для winsor_robust
        if self.scaler_type == "winsor_robust" and self.winsor_limits:
            data = self.winsorize(data, self.winsor_limits)

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if self.scaler_type in ("robust", "winsor_robust"):
                exprs = [
                    ((pl.col(c) - self.params[c]["median"]) / (self.params[c]["iqr"] + 1e-8))
                    .cast(pl.Float32)
                    .alias(c)
                    for c in self.params
                ]
            else:  # zscore
                exprs = [
                    ((pl.col(c) - self.params[c]["mean"]) / self.params[c]["std"])
                    .cast(pl.Float32)
                    .alias(c)
                    for c in self.params
                ]
            return data.with_columns(exprs)

        elif isinstance(data, np.ndarray):
            if data.shape[-1] != len(self.params):
                raise ValueError(f"Data shape {data.shape} mismatch with params length {len(self.params)}")
            
            res = data.copy().astype(np.float32)
            param_names = list(self.params.keys())

            for i, name in enumerate(param_names):
                p = self.params[name]
                if self.scaler_type in ("robust", "winsor_robust"):
                    center = p["median"]
                    scale = p["iqr"] + 1e-8
                else:  # zscore
                    center = p["mean"]
                    scale = p["std"]

                if data.ndim == 2:
                    res[:, i] = (data[:, i] - center) / scale
                elif data.ndim == 3:
                    res[:, :, i] = (data[:, :, i] - center) / scale
                else:
                    raise ValueError(f"Unsupported array dimension: {data.ndim}")

            return res

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def winsorize(self, data: Union[pl.DataFrame, pl.LazyFrame, np.ndarray], limits: List[float]) -> Union[pl.DataFrame, pl.LazyFrame, np.ndarray]:
        """
        Применяет винзоризацию (клиппинг экстремальных значений).
        """
        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            
            # Мы используем лимиты, вычисленные при fit, если они есть,
            # но задача говорит использовать перцентили.
            # В инференсе (Rust) мы будем использовать фиксированные значения (low/high),
            # рассчитанные на этапе обучения.
            
            for c in self.params:
                if "winsor_low" in self.params[c] and "winsor_high" in self.params[c]:
                    data = data.with_columns(pl.col(c).clip(lower_bound=self.params[c]["winsor_low"], upper_bound=self.params[c]["winsor_high"]))
                else:
                    # Если параметров нет, вычисляем на лету (только для обучения)
                    low = data[c].quantile(limits[0])
                    high = data[c].quantile(limits[1])
                    data = data.with_columns(pl.col(c).clip(lower_bound=low, upper_bound=high))
            return data
        
        elif isinstance(data, np.ndarray):
            # Для numpy аналогично
            res = data.copy()
            param_names = list(self.params.keys())
            for i, name in enumerate(param_names):
                if "winsor_low" in self.params[name] and "winsor_high" in self.params[name]:
                    low = self.params[name]["winsor_low"]
                    high = self.params[name]["winsor_high"]
                else:
                    low = np.quantile(data[..., i], limits[0])
                    high = np.quantile(data[..., i], limits[1])
                
                res[..., i] = np.clip(data[..., i], low, high)
            return res
        
        return data

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
