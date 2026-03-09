import polars as pl
from typing import Union, List

class Labeler:
    """
    Класс для формирования целевых переменных (labels) на основе будущих изменений цены.
    Реализует тернарную классификацию: Flat (0), Up (1), Down (2).
    
    Поддерживает Multi-Horizon Labeling (Задача 160):
    - Может генерировать метки для нескольких горизонтов одновременно
    - Автоматически маскирует недоступные горизонты значением -100
    """
    def __init__(self, horizon: Union[int, List[int]] = 100, threshold: float = 0.0005, dynamic_threshold: bool = True, window: int = 1000, K: float = 0.5):
        """
        horizon (K): Горизонт предсказания в количестве событий (строк).
                     Может быть int (один горизонт) или List[int] (несколько горизонтов).
                     Например: [10, 50, 100] для multi-horizon prediction.
        threshold: Базовый порог доходности (используется если dynamic_threshold=False).
        dynamic_threshold: Если True, порог вычисляется как rolling_std * K.
        window: Окно для расчета rolling_std.
        K: Коэффициент для динамического порога.
        """
        # Конвертируем в список для единообразия
        if isinstance(horizon, int):
            self.horizons = [horizon]
            self.single_horizon = True
        else:
            self.horizons = sorted(horizon)  # Сортируем для консистентности
            self.single_horizon = False
        
        self.threshold = threshold
        self.dynamic_threshold = dynamic_threshold
        self.window = window
        self.K = K
        self.max_horizon = max(self.horizons)

    def add_labels(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Добавляет в DataFrame колонки с метками для каждого горизонта.
        
        Для single horizon: добавляет колонку 'label'
        Для multi-horizon: добавляет колонки 'label_h{horizon}' для каждого горизонта
        
        Маскирование (Задача 160):
        - Если для сэмпла i значение i + horizon >= total_samples, метка = -100
        - Это позволяет использовать CrossEntropyLoss(ignore_index=-100)
        
        Вход:
            df: DataFrame или LazyFrame с колонкой 'mid_price'.
            
        Выход:
            DataFrame или LazyFrame с добавленными колонками меток.
            Строки БЕЗ будущего для max_horizon удаляются.
        """
        is_lazy = isinstance(df, pl.LazyFrame)
        
        # 1. Рассчитываем динамический порог если нужно
        if self.dynamic_threshold:
            # Используем std доходностей на окне
            df = df.with_columns(
                returns_std=pl.col("mid_price").pct_change().rolling_std(window_size=self.window).fill_null(strategy="backward")
            )
            # Порог = std * K. Ограничиваем снизу 0.0001 (0.01%)
            df = df.with_columns(
                dynamic_threshold=pl.col("returns_std").mul(self.K).clip(lower_bound=0.0001)
            )
            threshold_expr = pl.col("dynamic_threshold")
        else:
            threshold_expr = pl.lit(self.threshold)

        # Для каждого горизонта создаем метки
        label_columns = []
        
        for horizon in self.horizons:
            # 1. Получаем будущую цену через K шагов
            col_name = f"mid_future_h{horizon}"
            df = df.with_columns(
                **{col_name: pl.col("mid_price").shift(-horizon)}
            )
            
            # 2. Вычисляем доходность
            return_col = f"future_return_h{horizon}"
            df = df.with_columns(
                **{return_col: (pl.col(col_name) - pl.col("mid_price")) / pl.col("mid_price")}
            )
            
            # 3. Тернарная разметка с маскированием
            label_col = f"label_h{horizon}" if not self.single_horizon else "label"
            df = df.with_columns(
                **{label_col: pl.when(pl.col(return_col).is_null())
                    .then(pl.lit(-100))
                    .when(pl.col(return_col) > threshold_expr)
                    .then(pl.lit(1))
                    .when(pl.col(return_col) < -threshold_expr)
                    .then(pl.lit(2))
                    .otherwise(pl.lit(0))
                    .cast(pl.Int8)}
            )
            
            label_columns.append(label_col)
        
        # 4. Удаляем последние max_horizon строк (где все горизонты недоступны)
        # Для строк где хотя бы один горизонт доступен, оставляем с маскированием
        df = df.filter(pl.col(f"mid_future_h{self.max_horizon}").is_not_null())
        
        # Вывод статистики распределения классов для отладки (только для DataFrame)
        if not is_lazy:
            print(f"\n[{self.__class__.__name__}] Labels distribution:")
            print(f"  Horizons: {self.horizons}, Threshold: {self.threshold}")
            
            for label_col in label_columns:
                # Считаем распределение, исключая маскированные (-100)
                counts = df.filter(pl.col(label_col) != -100)[label_col].value_counts().sort(label_col)
                masked_count = df.filter(pl.col(label_col) == -100).height
                
                horizon_str = label_col.replace("label_h", "h") if "h" in label_col else "single"
                print(f"  {horizon_str}: {counts.to_dict(as_series=False)} | Masked: {masked_count}")
        
        return df

if __name__ == "__main__":
    # Тестовый запуск на демонстрационных данных
    data = {
        "mid_price": [100.0, 100.05, 100.1, 100.0, 99.9, 99.8, 100.0, 100.2, 100.3, 100.4, 100.5]
    }
    test_df = pl.DataFrame(data)
    
    print("=== Test 1: Single Horizon (backward compatibility) ===")
    labeler_single = Labeler(horizon=2, threshold=0.0005)
    res_single = labeler_single.add_labels(test_df)
    print("\nResulting DataFrame (single horizon):")
    print(res_single.select(["mid_price", "label"]))
    
    print("\n=== Test 2: Multi-Horizon [2, 5, 8] ===")
    labeler_multi = Labeler(horizon=[2, 5, 8], threshold=0.0005)
    res_multi = labeler_multi.add_labels(test_df)
    print("\nResulting DataFrame (multi-horizon):")
    print(res_multi.select(["mid_price", "label_h2", "label_h5", "label_h8"]))
