# 022 - Python Lab Feature Engineering

Цель задачи: Реализовать модуль python_lab/src/features.py для преобразования сырых данных стакана в нормализованные признаки (features). Мы внедряем стандартный пайплайн для LOB-моделей: относительные цены и логарифмированные объемы. Главное требование — использование Float32 и векторизация через Polars для обеспечения идентичности будущему Rust-коду.

Файлы для изменения/создания:

python_lab/src/features.py (создать)
Инструкции для Gemini:

python_lab/src/features.py: Создать класс FeatureEngineer с использованием эффективных выражений Polars.

import polars as pl

class FeatureEngineer:
    def __init__(self, n_levels: int = 50):
        self.n_levels = n_levels

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Вход: DataFrame со схемой из задачи 012.
        Выход: DataFrame с колонками feat_... и метаданными.
        """
        # 1. Вычисляем mid_price (с защитой от 0)
        df = df.with_columns(
            mid_price=(pl.col("ask_p_0") + pl.col("bid_p_0")) / 2
        ).with_columns(
            mid_price=pl.when(pl.col("mid_price") == 0).then(1.0).otherwise(pl.col("mid_price"))
        )

        # 2. Обработка цен: (price - mid) / mid
        # Важно: если цена 0.0 (padding), заменяем её на mid, чтобы отклонение стало 0.0
        price_cols = [f"ask_p_{i}" for i in range(self.n_levels)] + \
                     [f"bid_p_{i}" for i in range(self.n_levels)]
        
        price_exprs = [
            (
                (pl.when(pl.col(c) == 0).then(pl.col("mid_price")).otherwise(pl.col(c)) - pl.col("mid_price")) 
                / pl.col("mid_price")
            ).cast(pl.Float32).alias(f"feat_{c}")
            for c in price_cols
        ]

        # 3. Обработка объемов: log(1 + volume)
        vol_cols = [f"ask_v_{i}" for i in range(self.n_levels)] + \
                   [f"bid_v_{i}" for i in range(self.n_levels)]
        
        vol_exprs = [
            (pl.col(c) + 1).log().cast(pl.Float32).alias(f"feat_{c}")
            for c in vol_cols
        ]

        # Выполняем все трансформации разом
        df = df.with_columns(price_exprs + vol_exprs)

        # 4. Выборка колонок: сохраняем timestamp и сырой mid_price для будущих лейблов
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        return df.select(["timestamp_ms", "mid_price"] + feat_cols)

if __name__ == "__main__":
    # Пример использования
    fe = FeatureEngineer(n_levels=50)
    # df = loader.load_data()
    # feat_df = fe.transform(df)
Технические требования:

Типы данных: Все признаки должны быть pl.Float32. Это критично для соответствия инференсу в Rust и экономии памяти при обучении.
Обработка Zero-Padding: Если в Rust-дампере уровень был заполнен нулем, замена 0.0 -> mid_price в расчете отклонения даст признак 0.0, что корректно интерпретируется нейросетью как отсутствие данных.
Векторизация: Использовать списковые включения (list comprehensions) для формирования списка выражений Polars, чтобы применить их одним вызовом with_columns.
Почему это важно: Относительная нормализация цен делает модель инвариантной к абсолютному уровню цены токена (будь то 0.01или0.01 или 0.01или60,000). Это позволяет в будущем использовать Transfer Learning между разными торговыми парами. Идентичность этого кода будущему коду в Rust гарантирует отсутствие "обучения на ошибках" (train-test skew).