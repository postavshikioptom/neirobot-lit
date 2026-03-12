import polars as pl

class FeatureEngineer:
    """
    Класс для преобразования сырых данных стакана (LOB) в нормализованные признаки.
    Реализует пайплайн: относительные цены и логарифмированные объемы.
    
    Согласно задаче 022:
    - Цены: feat_ask_p_{i} и feat_bid_p_{i} (100 признаков)
    - Объемы: feat_ask_v_{i} и feat_bid_v_{i} (100 признаков)
    """
    def __init__(self, n_levels: int = 50):
        self.n_levels = n_levels

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Преобразует DataFrame со схемой LOB в DataFrame с производными признаками.
        
        Согласно задаче 022, вычисляет:
        1. Normalized Price: (price - mid) / mid для каждого ask_p и bid_p
           - Если цена 0.0 (padding), заменяем на mid, чтобы отклонение = 0.0
        2. Log Volume: log(1 + volume) для каждого ask_v и bid_v
        
        Вход:
            df: DataFrame со схемой LOB (ask_p_i, ask_v_i, bid_p_i, bid_v_i для i=0..49)
        
        Выход:
            DataFrame с колонками timestamp_ms, mid_price и feat_ask_p_*, feat_bid_p_*, 
            feat_ask_v_*, feat_bid_v_* (всего 200 признаков)
        """
        # 1. Вычисляем mid_price (с защитой от 0)
        df = df.with_columns(
            mid_price=(pl.col("ask_p_0") + pl.col("bid_p_0")) / 2
        ).with_columns(
            mid_price=pl.when(pl.col("mid_price") == 0).then(1.0).otherwise(pl.col("mid_price"))
        )

        # 2. Подготовка списков столбцов по типам
        ask_p_cols = [f"ask_p_{i}" for i in range(self.n_levels)]
        ask_v_cols = [f"ask_v_{i}" for i in range(self.n_levels)]
        bid_p_cols = [f"bid_p_{i}" for i in range(self.n_levels)]
        bid_v_cols = [f"bid_v_{i}" for i in range(self.n_levels)]

        # 3. Обработка цен: (price - mid) / mid
        # Важно: если цена 0.0 (padding), заменяем её на mid, чтобы отклонение стало 0.0
        price_exprs = [
            (
                (pl.when(pl.col(c) == 0).then(pl.col("mid_price")).otherwise(pl.col(c)) - pl.col("mid_price")) 
                / pl.col("mid_price")
            ).cast(pl.Float32).alias(f"feat_{c}")
            for c in ask_p_cols + bid_p_cols
        ]

        # 4. Обработка объемов: log(1 + volume)
        vol_exprs = [
            (pl.col(c) + 1).log().cast(pl.Float32).alias(f"feat_{c}")
            for c in ask_v_cols + bid_v_cols
        ]

        # 5. Вычисляем OFI и VIB (Задача 053, 306)
        # OFI (Order Flow Imbalance) - Static Imbalance для лучшего уровня (depth 0)
        # Формула: (V_bid_0 - V_ask_0) / (V_bid_0 + V_ask_0 + epsilon)
        # Результат: скаляр в диапазоне [-1, 1] для каждого снапшота
        
        bv0 = pl.col("bid_v_0")
        av0 = pl.col("ask_v_0")
        
        # Вычисляем static imbalance с защитой от деления на ноль
        denom = bv0 + av0 + 1e-7
        ofi_expr = ((bv0 - av0) / denom).clip(-1.0, 1.0).cast(pl.Float32).alias("feat_ofi_100")
        
        # VIB (Trade Imbalance) - если есть колонки сделок (Задача 212)
        if "feat_trade_volume" in df.columns and "feat_trade_side" in df.columns:
            vib_expr = (pl.col("feat_trade_volume") * pl.col("feat_trade_side")).cast(pl.Float32).alias("feat_vib_100")
        else:
            vib_expr = pl.lit(0.0).cast(pl.Float32).alias("feat_vib_100")

        # 6. Выполняем все трансформации разом
        df = df.with_columns(price_exprs + vol_exprs + [ofi_expr, vib_expr])

        # 7. Формируем итоговый DF с СТРОГИМ ПОРЯДКОМ столбцов (Interleaved по стороне)
        # Это критически важно для Dataset.py (задача 304)
        ordered_feat_cols = []
        # Блок ASK (индексы 0-99)
        ordered_feat_cols.extend([f"feat_ask_p_{i}" for i in range(self.n_levels)]) # 0-49: только цены
        ordered_feat_cols.extend([f"feat_ask_v_{i}" for i in range(self.n_levels)]) # 50-99: только объемы

        # Блок BID (индексы 100-199)
        ordered_feat_cols.extend([f"feat_bid_p_{i}" for i in range(self.n_levels)]) # 100-149: только цены
        ordered_feat_cols.extend([f"feat_bid_v_{i}" for i in range(self.n_levels)]) # 150-199: только объемы

        # Все остальное (метаданные и дополнительные признаки)
        meta_cols = ["timestamp_ms", "mid_price", "last_update_id"]
        other_cols = [c for c in df.columns if c.startswith("feat_past_return_")]
        
        # Добавляем OFI и VIB в дополнительные признаки (Задача 306)
        extra_feats = ["feat_ofi_100", "feat_vib_100"]
        
        # Возвращаем DF, где ПРИЗНАКИ LOB ПЕРВЫМИ 200 КОЛОНКАМИ
        return df.select(
            [pl.col(c) for c in ordered_feat_cols] + 
            [pl.col(c) for c in meta_cols if c in df.columns] +
            [pl.col(c) for c in other_cols] +
            [pl.col(c) for c in extra_feats]
        )

if __name__ == "__main__":
    # Тестовый пример с искусственными данными
    import numpy as np
    
    # Создаем фейковый DataFrame согласно схеме 012
    data = {
        "timestamp_ms": [1000, 2000, 3000, 4000],
    }
    
    # Добавляем ask_p (50 уровней)
    for i in range(50):
        data[f"ask_p_{i}"] = [101.0 + i*0.1, 102.0 + i*0.1, 103.0 + i*0.1, 104.0 + i*0.1]
    
    # Добавляем ask_v (50 уровней)
    for i in range(50):
        data[f"ask_v_{i}"] = [10.0 + i, 20.0 + i, 30.0 + i, 40.0 + i]
    
    # Добавляем bid_p (50 уровней)
    for i in range(50):
        data[f"bid_p_{i}"] = [99.0 - i*0.1, 98.0 - i*0.1, 97.0 - i*0.1, 96.0 - i*0.1]
    
    # Добавляем bid_v (50 уровней)
    for i in range(50):
        data[f"bid_v_{i}"] = [15.0 + i, 25.0 + i, 35.0 + i, 45.0 + i]
        
    df = pl.DataFrame(data)
    fe = FeatureEngineer(n_levels=50)
    feat_df = fe.transform(df)
    
    print("Features DataFrame Shape:", feat_df.shape)
    print("Columns starts with feat_ask_p:", [c for c in feat_df.columns if c.startswith("feat_ask_p")][:5])
    print("Columns starts with feat_bid_p:", [c for c in feat_df.columns if c.startswith("feat_bid_p")][:5])
    print("Columns starts with feat_ask_v:", [c for c in feat_df.columns if c.startswith("feat_ask_v")][:5])
    print("Columns starts with feat_bid_v:", [c for c in feat_df.columns if c.startswith("feat_bid_v")][:5])
    print("First row mid_price:", feat_df["mid_price"][0])
    print("Total feat columns:", len([c for c in feat_df.columns if c.startswith("feat_")]))

