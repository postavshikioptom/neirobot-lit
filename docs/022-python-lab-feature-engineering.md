# 022 - Python Lab Feature Engineering

Цель задачи: Реализовать модуль python_lab/src/features.py для преобразования сырых данных стакана в нормализованные признаки (features). Мы внедряем стандартный пайплайн для LOB-моделей: относительные цены и логарифмированные объемы. Главное требование — использование Float32 и векторизация через Polars.

**ВАЖНО: Метод transform должен ДОБАВЛЯТЬ признаки в DataFrame, сохраняя исходные колонки (timestamp, raw prices/volumes), так как они потребуются для дальнейших расчетов (лейблинг, детекция режимов и т.д.).**

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
        Вход: DataFrame со схемой из задачи 012 (сырой стакан).
        Выход: DataFrame с ДОБАВЛЕННЫМИ колонками feat_... и оригинальными данными.
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

        # 4. Выполняем все трансформации и возвращаем полный DataFrame
        # ВАЖНО: Мы НЕ делаем select, чтобы сохранить все оригинальные колонки для задач 023 (labeling) и 155 (regimes)
        return df.with_columns(price_exprs + vol_exprs)

if __name__ == "__main__":
    # Пример использования
    fe = FeatureEngineer(n_levels=50)
    # df = loader.load_data()
    # feat_df = fe.transform(df)

Технические требования:

*   **Сохранение данных**: Исходные колонки (ask_p_*, bid_p_*, ask_v_*, bid_v_*) ДОЛЖНЫ оставаться в DataFrame после трансформации.
*   **Типы данных**: Все признаки должны быть pl.Float32.
*   **Обработка Zero-Padding**: 0.0 -> mid_price, отклонение 0.0.
*   **Векторизация**: Использовать expressions Polars через with_columns.

Почему это важно: Сохранение исходных колонок критично для дальнейших этапов пайплайна. Например, для расчета Order Flow Imbalance (задача 053/155) или для разметки данных (задача 023), где нужен исходный mid_price или уровни цен. Очистка колонок производится только на самом последнем этапе формирования тензора для обучения.