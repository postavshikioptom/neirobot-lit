# 024 - Python Lab Normalization Params

Цель задачи: Реализовать расчет глобальных параметров нормализации (Z-score) для признаков. Необходимо вычислить среднее (mean) и стандартное отклонение (std) для каждой колонки feat_..., используя только тренировочную выборку, чтобы избежать утечки данных (Data Leakage). Параметры сохраняются в norm.json для использования при инференсе в Rust.

Файлы для изменения/создания:

python_lab/src/normalization.py (создать)
Инструкции для Gemini:

python_lab/src/normalization.py: Реализовать класс Normalizer с поддержкой LazyFrame и защитой от деления на ноль.

import polars as pl
import json
from pathlib import Path

class Normalizer:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.params = {}

    def fit(self, df: Union[pl.DataFrame, pl.LazyFrame]):
        """
        Рассчитывает параметры на Train выборке. 
        ВАЖНО: Вызывать только на данных ДО разделения или только на Train части.
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        
        # Эффективный расчет агрегатов через Polars
        means = df.select(feat_cols).mean()
        stds = df.select(feat_cols).std()

        for c in feat_cols:
            m = float(means.get_column(c)[0])
            s = float(stds.get_column(c)[0])
            
            # Защита от константных колонок и нулевого std
            if s == 0 or np.isnan(s):
                s = 1.0
            
            self.params[c] = {"mean": m, "std": s}
        
        return self.params

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.params, f, indent=4)
        print(f"Normalization params saved to {self.output_path}")

    def transform(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Применяет Z-score: (x - mean) / std и приводит к Float32"""
        exprs = [
            ((pl.col(c) - self.params[c]["mean"]) / self.params[c]["std"])
            .cast(pl.Float32)
            .alias(c)
            for c in self.params
        ]
        return df.with_columns(exprs)
Технические требования:

Валидация: Если std неопределен (NaN) или равен 0, принудительно устанавливать его в 1.0.
Типы: Обязательный cast(pl.Float32) в методе transform для полной совместимости с Rust-модулем ort и экономии памяти.
Data Leakage: В коде обучения (задача 027) добавить комментарий, что fit выполняется строго на train_df.
Почему это важно: Z-score нормализация критична для трансформеров (LiT), так как она центрирует данные вокруг нуля, что ускоряет сходимость градиентного спуска. Сохранение этих параметров в файл — единственный способ гарантировать, что во время реальной торговли Rust применит к стакану те же коэффициенты, которые видела модель при обучении.