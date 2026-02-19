023 - Python Lab Labeling**

Цель задачи: Реализовать модуль python_lab/src/labels.py для формирования целевых переменных (labels). Мы используем тернарную классификацию (Up, Down, Flat) на основе изменения mid_price через горизонт в KKK событий. Это позволит модели предсказывать значимые движения цены, превышающие торговые издержки.

Файлы для изменения/создания:

python_lab/src/labels.py (создать)
Инструкции для Gemini:

python_lab/src/labels.py: Реализовать класс Labeler для расчета доходностей и квантования их в три класса.

import polars as pl

class Labeler:
    def __init__(self, horizon: int = 100, threshold: float = 0.0005):
        """
        horizon (K): через сколько событий смотреть будущую цену.
        threshold: порог доходности (0.0005 = 0.05%), должен учитывать комиссии.
        """
        self.horizon = horizon
        self.threshold = threshold

    def add_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Добавляет колонку 'label':
        1 (Up): return > threshold
        2 (Down): return < -threshold
        0 (Flat): иначе
        """
        # 1. Вычисляем будущую цену через K шагов
        # 2. Вычисляем доходность: (mid_future - mid_now) / mid_now
        df = df.with_columns(
            mid_future=pl.col("mid_price").shift(-self.horizon)
        ).with_columns(
            future_return=(pl.col("mid_future") - pl.col("mid_price")) / pl.col("mid_price")
        )

        # 3. Тернарная разметка (Ternary Classification)
        df = df.with_columns(
            label=pl.when(pl.col("future_return") > self.threshold).then(1)      # Up
                  .when(pl.col("future_return") < -self.threshold).then(2)     # Down
                  .when(pl.col("future_return").is_not_null()).otherwise(0)     # Flat (если есть данные)
                  .cast(pl.Int8)
        )

        # 4. Обязательно удаляем последние K строк, где будущее неизвестно (NaN)
        df = df.drop_nulls(subset=["mid_future"])
        
        # Выводим баланс классов для самопроверки
        counts = df["label"].value_counts()
        print(f"Labels distribution (K={self.horizon}, T={self.threshold}):\n{counts}")
        
        return df

if __name__ == "__main__":
    # Тестовый пример
    data = {"mid_price": [100.0, 100.05, 100.1, 100.0, 99.9, 99.8, 100.0]}
    test_df = pl.DataFrame(data)
    labeler = Labeler(horizon=2, threshold=0.0005)
    res = labeler.add_labels(test_df)
    print(res)
Технические требования:

Тип данных: Использовать Int8 для колонки label (0, 1, 2).
Порог (threshold): Значение по умолчанию установлено в 0.0005 (0.05%). Это база, которую мы будем подстраивать под волатильность конкретной монеты.
Векторизация: Использовать shift(-K) в Polars, что эквивалентно заглядыванию в будущее без циклов.
Валидация: Выводить value_counts() после разметки, чтобы видеть, не "захлебнулся" ли класс Flat или Up/Down.
Почему это важно: Тернарная разметка позволяет модели игнорировать рыночный шум. Если изменение цены меньше комиссии (например, 0.01% при комиссии 0.1%), такая сделка нам не нужна. Класс Flat (0) обучает модель "ждать" лучшего момента. Удаление последних KKK строк предотвращает "утечку данных из будущего" (data leakage) при обучении на краях файлов.