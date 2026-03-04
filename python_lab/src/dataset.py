import polars as pl
import json
import torch
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from typing import Union, List, Literal, Dict, Any
from torch.utils.data import Dataset
from onnxruntime.quantization.calibrate import CalibrationDataReader
import onnx

from .normalization import Normalizer

# ============================================================================
# Константы для аугментации LOB данных
# ============================================================================

# ============================================================================
# Функции расчета таргетов и признаков
# ============================================================================
def compute_target_vol(mid_prices, window=100):
    """
    Рассчитывает реализованную волатильность на окне вперед.
    
    mid_prices: массив средних цен.
    window: размер окна в тиках для расчета реализованной волатильности.
    """
    log_prices = np.log(mid_prices)
    log_returns = np.diff(log_prices)
    
    # Скользящее стандартное отклонение (std) лог-доходностей
    vol = pd.Series(log_returns).rolling(window).std().shift(-window)
    # Target: Log-Vol (добавляем epsilon для защиты от log(0))
    target_vol = np.log(vol + 1e-8)
    # Заполняем пропуски нулями (края последовательности)
    return target_vol.fillna(target_vol.mean() if not target_vol.isna().all() else 0).values


def compute_past_returns(mid_prices: np.ndarray, lags: List[int]) -> np.ndarray:
    """
    Рассчитывает Multi-scale Log-Returns (Past Returns) для различных временных горизонтов.
    
    Согласно плану 091:
    - Расчет: Rn = ln(mid_price_t) - ln(mid_price_t-n)
    - NaN Handling: Первые n значений заполняются 0.0 (нейтральный сигнал)
    - Форма выхода: (len(mid_prices), len(lags))
    
    Args:
        mid_prices: массив средних цен (N,)
        lags: список лагов для расчета [10, 50, 100]
    
    Returns:
        np.ndarray: матрица log-returns (N, len(lags))
    """
    n = len(mid_prices)
    n_lags = len(lags)
    past_returns = np.zeros((n, n_lags), dtype=np.float32)
    
    # Вычисляем логарифмы цен один раз для эффективности
    log_prices = np.log(mid_prices)
    
    for lag_idx, lag in enumerate(lags):
        # Первые lag значений заполняются 0.0
        for t in range(lag, n):
            # Rn = ln(price_t) - ln(price_t-n)
            past_returns[t, lag_idx] = log_prices[t] - log_prices[t - lag]
    
    return past_returns


# ============================================================================
# Функции расчета признаков для определения режимов рынка (HMM)
# ============================================================================

def compute_intensity(timestamps, window=1000):
    """
    Рассчитывает интенсивность событий (количество обновлений в окне).
    
    Args:
        timestamps: массив временных меток в миллисекундах
        window: размер окна в количестве событий
    
    Returns:
        np.ndarray: интенсивность для каждого момента времени
    """
    n = len(timestamps)
    intensity = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        # Интенсивность = количество событий в окне
        intensity[i] = i - start_idx + 1
    
    return intensity


def compute_volatility(mid_prices, window=1000):
    """
    Рассчитывает волатильность (log std mid_price) на скользящем окне.
    
    Args:
        mid_prices: массив средних цен
        window: размер окна в количестве событий
    
    Returns:
        np.ndarray: логарифм стандартного отклонения цен
    """
    # Стандартное отклонение цен (не логарифмированных!)
    vol = pd.Series(mid_prices).rolling(window, min_periods=1).std()
    # Логарифм волатильности для стабилизации масштаба
    log_vol = np.log(vol + 1e-8)
    return log_vol.fillna(log_vol.mean() if not log_vol.isna().all() else 0).values


def compute_spread_zscore(ask_prices, bid_prices, window=1000):
    """
    Рассчитывает Z-score спреда (нормализованный спред).
    
    Args:
        ask_prices: массив лучших цен ask (ask_p_0)
        bid_prices: массив лучших цен bid (bid_p_0)
        window: размер окна для нормализации
    
    Returns:
        np.ndarray: Z-score спреда
    """
    spread = ask_prices - bid_prices
    spread_series = pd.Series(spread)
    
    rolling_mean = spread_series.rolling(window, min_periods=1).mean()
    rolling_std = spread_series.rolling(window, min_periods=1).std()
    
    # Z-score = (spread - mean) / std
    zscore = (spread - rolling_mean) / (rolling_std + 1e-8)
    return zscore.fillna(0).values


def compute_ofi(ask_prices, ask_volumes, bid_prices, bid_volumes, window=1000):
    """
    Рассчитывает Order Flow Imbalance (OFI) - кумулятивный дисбаланс потока ордеров.
    
    OFI показывает агрессивность покупателей vs продавцов.
    Положительный OFI = больше покупок, отрицательный = больше продаж.
    
    Args:
        ask_prices: массив лучших цен ask
        ask_volumes: массив объемов на лучшем ask
        bid_prices: массив лучших цен bid
        bid_volumes: массив объемов на лучшем bid
        window: размер окна для кумулятивного расчета
    
    Returns:
        np.ndarray: кумулятивный OFI
    """
    n = len(ask_prices)
    ofi = np.zeros(n)
    
    for i in range(1, n):
        # Изменение объема на bid стороне
        if bid_prices[i] >= bid_prices[i-1]:
            delta_bid = bid_volumes[i] - bid_volumes[i-1] if bid_prices[i] == bid_prices[i-1] else bid_volumes[i]
        else:
            delta_bid = -bid_volumes[i-1]
        
        # Изменение объема на ask стороне
        if ask_prices[i] <= ask_prices[i-1]:
            delta_ask = ask_volumes[i] - ask_volumes[i-1] if ask_prices[i] == ask_prices[i-1] else ask_volumes[i]
        else:
            delta_ask = -ask_volumes[i-1]
        
        # OFI = delta_bid - delta_ask (положительный = агрессивные покупки)
        ofi[i] = delta_bid - delta_ask
    
    # Кумулятивная сумма на окне
    ofi_cumsum = pd.Series(ofi).rolling(window, min_periods=1).sum()
    return ofi_cumsum.fillna(0).values


def compute_regime_features(df: pl.DataFrame, window: int = 1000) -> np.ndarray:
    """
    Вычисляет признаки для определения режимов рынка из DataFrame.
    
    Args:
        df: DataFrame с колонками timestamp_ms, mid_price, ask_p_0, ask_v_0, bid_p_0, bid_v_0
        window: размер окна для расчета признаков
    
    Returns:
        np.ndarray: массив признаков формы (n_samples, 4)
                   где 4 признака: [intensity, volatility, spread_zscore, ofi]
    """
    # Извлекаем необходимые колонки
    timestamps = df["timestamp_ms"].to_numpy()
    mid_prices = df["mid_price"].to_numpy()
    ask_prices = df["ask_p_0"].to_numpy()
    ask_volumes = df["ask_v_0"].to_numpy()
    bid_prices = df["bid_p_0"].to_numpy()
    bid_volumes = df["bid_v_0"].to_numpy()
    
    # Вычисляем признаки
    intensity = compute_intensity(timestamps, window=window)
    volatility = compute_volatility(mid_prices, window=window)
    spread_zscore = compute_spread_zscore(ask_prices, bid_prices, window=window)
    ofi = compute_ofi(ask_prices, ask_volumes, bid_prices, bid_volumes, window=window)
    
    # Объединяем в матрицу признаков
    features = np.column_stack([intensity, volatility, spread_zscore, ofi])
    return features


def compute_trade_imbalance(
    df_snapshots: pl.DataFrame, 
    df_trades: pl.DataFrame, 
    windows: List[str] = ["1s", "5s", "15s", "60s"],
    agg_type: str = 'vol',
    noise_filter_pct: float = 0.05
) -> pl.DataFrame:
    """
    Вычисляет дисбаланс публичных сделок (Micro-Trades Imbalance) для заданных временных окон.
    Задача 236: Реализация признака на основе потока рыночных сделок.
    
    Args:
        df_snapshots: DataFrame со снапшотами стакана (timestamp_ms, ...)
        df_trades: DataFrame с публичными сделками (timestamp, price, size, side)
        windows: список временных окон для агрегации (например, ["1s", "5s", "15s", "60s"])
        agg_type: тип агрегации - 'vol' (объем) или 'count' (количество сделок)
        noise_filter_pct: процент для фильтрации шума (отсекаем сделки меньше этого процента от медианы)
    
    Returns:
        pl.DataFrame: исходный df_snapshots с добавленными колонками imbalance для каждого окна
    """
    if df_trades.is_empty():
        # Если нет сделок, возвращаем snapshots с нулевыми колонками imbalance
        for w in windows:
            col_name = f"feat_imb_{agg_type}_{w}"
            df_snapshots = df_snapshots.with_columns(pl.lit(0.0).alias(col_name))
        return df_snapshots
    
    # 1. Фильтрация шума: отсекаем сделки меньше noise_filter_pct от медианного размера
    median_size = df_trades["size"].median()
    threshold = median_size * noise_filter_pct
    df_trades = df_trades.filter(pl.col("size") > threshold)
    
    if df_trades.is_empty():
        # Если после фильтрации не осталось сделок
        for w in windows:
            col_name = f"feat_imb_{agg_type}_{w}"
            df_snapshots = df_snapshots.with_columns(pl.lit(0.0).alias(col_name))
        return df_snapshots
    
    # 2. Подготовка подписанного значения (signed_val) в зависимости от стороны сделки
    if agg_type == 'vol':
        df_trades = df_trades.with_columns(
            pl.when(pl.col("side") == "Buy")
            .then(pl.col("size"))
            .otherwise(-pl.col("size"))
            .alias("signed_val")
        )
    else:  # count
        df_trades = df_trades.with_columns(
            pl.when(pl.col("side") == "Buy")
            .then(pl.lit(1))
            .otherwise(pl.lit(-1))
            .alias("signed_val")
        )
    
    # Также создаем колонку с абсолютным значением для знаменателя
    df_trades = df_trades.with_columns(
        pl.col("signed_val").abs().alias("abs_val")
    )
    
    # 3. Преобразуем timestamp в datetime для rolling_sum_by
    # timestamp в trades - это миллисекунды (i64), преобразуем в datetime
    df_trades = df_trades.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime")
    )
    
    # 4. Для каждого окна вычисляем rolling imbalance
    for window in windows:
        # Используем rolling_sum_by для устранения lookahead bias
        # closed='right' означает, что окно включает текущую точку и смотрит назад
        df_agg = df_trades.with_columns([
            pl.col("signed_val").rolling_sum_by("datetime", window_size=window, closed="right").alias("sum_signed"),
            pl.col("abs_val").rolling_sum_by("datetime", window_size=window, closed="right").alias("sum_abs")
        ])
        
        # Вычисляем imbalance: sum(signed) / (sum(abs) + 1e-6)
        df_agg = df_agg.with_columns(
            (pl.col("sum_signed") / (pl.col("sum_abs") + 1e-6)).alias("imbalance")
        )
        
        # Преобразуем datetime обратно в timestamp_ms для join
        df_agg = df_agg.with_columns(
            pl.col("datetime").dt.epoch(time_unit="ms").alias("timestamp_ms")
        )
        
        # 5. Используем join_asof для сопоставления агрегированных imbalance со snapshots
        # Каждый snapshot получает imbalance из ближайшего предыдущего окна (backward join)
        col_name = f"feat_imb_{agg_type}_{window}"
        df_snapshots = df_snapshots.join_asof(
            df_agg.select(["timestamp_ms", "imbalance"]),
            on="timestamp_ms",
            strategy="backward"
        ).rename({"imbalance": col_name})
        
        # Заполняем пропуски нулями (для начала последовательности)
        df_snapshots = df_snapshots.with_columns(
            pl.col(col_name).fill_null(0.0)
        )
    
    return df_snapshots


import pyarrow.parquet as pq

def fast_parquet_reader(file_path, batch_size=100_000, columns=None):
    """
    Генератор батчей данных через PyArrow для минимизации RAM.
    Использует 50 уровней стакана согласно задаче 012.
    Задача 212: Включает колонки сделок (trade_price, trade_volume) для симуляции очереди.
    """
    parquet_file = pq.ParquetFile(file_path)
    
    if columns is None:
        # Если колонки не заданы, читаем все важные
        all_cols = parquet_file.schema.names
        
        target_cols = ["timestamp_ms", "mid_price"]
        # Задача 012: Порядок колонок согласно схеме - interleaved (p,v) для каждого уровня
        # Структура: ask_p_0, ask_v_0, ask_p_1, ask_v_1, ..., ask_p_49, ask_v_49, bid_p_0, bid_v_0, ..., bid_p_49, bid_v_49
        lob_cols = []
        for i in range(50):
            lob_cols.append(f"ask_p_{i}")
            lob_cols.append(f"ask_v_{i}")
        for i in range(50):
            lob_cols.append(f"bid_p_{i}")
            lob_cols.append(f"bid_v_{i}")
        # Задача 212: Добавляем колонки сделок для очереди лимитных ордеров
        trade_cols = ["trade_price", "trade_volume"]
        # Пытаемся найти feat_ колонки и label
        feat_cols = [c for c in all_cols if c.startswith("feat_") or c == "label"]
        
        # Пересечение с реальными колонками в файле
        columns = [c for c in (target_cols + lob_cols + trade_cols + feat_cols) if c in all_cols]

    # Итерируемся батчами через PyArrow (True Streaming)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        # Конвертируем PyArrow Batch -> Polars DataFrame (zero-copy где возможно)
        yield pl.from_arrow(batch)

def _compute_augmentation_indices(n_levels: int):
    """
    Вычисляет индексы колонок для аугментации стакана глубиной n_levels.
    Структура согласно задаче 012: [ask_p_0, ask_v_0, ask_p_1, ask_v_1, ..., ask_p_49, ask_v_49, bid_p_0, bid_v_0, ..., bid_p_49, bid_v_49]
    
    Returns:
        tuple: (PRICE_COLS, VOL_COLS, ASK_COLS, BID_COLS)
    """
    # Структура: 
    # [0, 1] = ask_p_0, ask_v_0
    # [2, 3] = ask_p_1, ask_v_1
    # ...
    # [98, 99] = ask_p_49, ask_v_49
    # [100, 101] = bid_p_0, bid_v_0
    # ...
    # [198, 199] = bid_p_49, bid_v_49
    
    # PRICE_COLS = все четные индексы (все цены ask_p и bid_p)
    PRICE_COLS = list(range(0, 200, 2))
    
    # VOL_COLS = все нечетные индексы (все объемы ask_v и bid_v)
    VOL_COLS = list(range(1, 200, 2))
    
    # ASK_COLS = ask_p и ask_v (индексы 0..99)
    ASK_COLS = list(range(0, 100))
    
    # BID_COLS = bid_p и bid_v (индексы 100..199)
    BID_COLS = list(range(100, 200))
    
    return PRICE_COLS, VOL_COLS, ASK_COLS, BID_COLS


def apply_symmetric_flip(features, label, price_cols, ask_cols, bid_cols):
    """
    Зеркальное отражение стакана.
    1. Меняем блоки Ask и Bid местами.
    2. Инвертируем знак относительных цен.
    3. Инвертируем метку Up (0) <-> Down (1).
    
    Args:
        features: torch.Tensor формы (seq_len, num_features)
        label: int - метка класса
        price_cols: list - индексы колонок с ценами
        ask_cols: list - индексы колонок ask
        bid_cols: list - индексы колонок bid
    
    Returns:
        tuple: (flipped_features, new_label)
    """
    flipped = features.clone()
    
    # Своп колонок Ask <-> Bid
    flipped[:, ask_cols] = features[:, bid_cols]
    flipped[:, bid_cols] = features[:, ask_cols]
    
    # Инверсия знака цен (относительно mid_price, который равен 0)
    flipped[:, price_cols] *= -1.0
    
    # Инверсия метки Up (1) <-> Down (2), Flat (0) остается без изменений
    new_label = label
    if label == 1:
        new_label = 2  # Up -> Down
    elif label == 2:
        new_label = 1  # Down -> Up
    
    return flipped, new_label


def apply_volume_jitter(features, jitter_range, vol_cols, generator):
    """
    Случайное изменение объемов в заданном диапазоне.
    
    Args:
        features: torch.Tensor формы (seq_len, num_features)
        jitter_range: float - максимальное относительное изменение (например, 0.1 для ±10%)
        vol_cols: list - индексы колонок с объемами
        generator: torch.Generator - генератор случайных чисел
    
    Returns:
        torch.Tensor: features с измененными объемами
    """
    # Генерируем случайный множитель в диапазоне [1-jitter_range, 1+jitter_range]
    multiplier = 1.0 + (torch.rand(1, generator=generator).item() * 2 - 1) * jitter_range
    features[:, vol_cols] *= multiplier
    return features

def validate_lob_sequence(sequence):
    """
    sequence: (Seq_Len, Features)
    Проверяет: цены > 0, объемы > 0, ask_0 > bid_0 (no cross).
    
    Структура согласно задаче 012: [ask_p_0, ask_v_0, ask_p_1, ask_v_1, ..., ask_p_49, ask_v_49, bid_p_0, bid_v_0, ..., bid_p_49, bid_v_49]
    Индексы:
    - ask_p_i: 2*i
    - ask_v_i: 2*i+1
    - bid_p_i: 100+2*i
    - bid_v_i: 100+2*i+1
    """
    for step in sequence:
        # Лучший ask и bid (индекс 0)
        ask_p0 = step[0]
        bid_p0 = step[100]
        
        # Спред схлопнулся или отрицательный
        # В нормализованном виде: ask_p0 > 0 > bid_p0
        if ask_p0 <= bid_p0: 
            return False 
            
        # Цены не могут быть <= -1.0 (в нормализованном виде (p-mid)/mid это p <= 0)
        # Проверяем все ask цены (индексы 0, 2, 4, ..., 98)
        for i in range(50):
            if step[i * 2] <= -1.0:
                return False
        
        # Проверяем все bid цены (индексы 100, 102, 104, ..., 198)
        for i in range(50):
            if step[100 + i * 2] >= 1.0:
                return False
            
        # Объемы логарифмированы log(1+v), должны быть >= 0
        # Проверяем все ask объемы (индексы 1, 3, 5, ..., 99)
        for i in range(50):
            if step[i * 2 + 1] < 0:
                return False
        
        # Проверяем все bid объемы (индексы 101, 103, 105, ..., 199)
        for i in range(50):
            if step[100 + i * 2 + 1] < 0:
                return False
        
    return True

def balance_dataset(features, labels, method='bgmm', ratio=0.5, sampling_strategy=None):
    """
    features: (N, Seq_Len, Feats) - numpy array
    labels: (N,) - numpy array
    ratio: целевая доля миноритарных классов относительно мажоритарного
    sampling_strategy: внешняя стратегия балансировки (для батчевой обработки)
    """
    from smote_variants import BGMM_SMOTE
    
    orig_shape = features.shape # (N, S, F)
    n_samples = orig_shape[0]
    
    # 1. Flatten для совместимости с библиотеками оверсэмплинга
    features_2d = features.reshape(n_samples, -1)
    
    # 2. Определение стратегии
    if sampling_strategy is None:
        counts = np.bincount(labels)
        # Обеспечиваем наличие всех 3 классов (0, 1, 2)
        if len(counts) < 3:
            full_counts = np.zeros(3, dtype=int)
            full_counts[:len(counts)] = counts
            counts = full_counts
            
        maj_class = np.argmax(counts)
        target_count = int(counts[maj_class] * ratio)
        
        # Стратегия: балансируем только Up (1) и Down (2) до target_count
        sampling_strategy = {1: max(counts[1], target_count), 2: max(counts[2], target_count)}
    
    if method == 'bgmm':
        sampler = BGMM_SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'smote':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    else:
        return features, labels

    # Исправлено: fit_resample вместо sample
    X_res, y_res = sampler.fit_resample(features_2d, labels)
    
    # 3. Reshape обратно в последовательности
    X_res = X_res.reshape(-1, orig_shape[1], orig_shape[2])
    
    # 4. Фильтрация физически невозможных стаканов
    mask = [validate_lob_sequence(seq) for seq in X_res]
    X_res_filt = X_res[mask]
    y_res_filt = y_res[mask]
    
    n_rejected = len(X_res) - len(X_res_filt)
    print(f"[Balance] Method: {method}, Rejected invalid samples: {n_rejected}")
    
    return X_res_filt, y_res_filt

class LOBDataLoader:
    """
    Загрузчик датасета Order Book для Python Lab.
    Обеспечивает валидацию схемы данных и эффективную загрузку через Polars.
    """
    def __init__(self, data_path: str, symbol: str, schema_path: str = "../../docs/data_schema.json"):
        self.data_path = Path(data_path)
        self.symbol = symbol
        # Путь к схеме относительно места запуска скрипта
        self.schema_path = Path(__file__).parent.parent.parent / "docs" / "data_schema.json" if schema_path == "../../docs/data_schema.json" else Path(schema_path)
        self.expected_columns = self._load_schema()

    def _load_schema(self) -> List[str]:
        """Загружает ожидаемый список колонок из JSON схемы."""
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
            return schema["columns"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found at {self.schema_path.absolute()}")

    def load_data(self, lazy: bool = False) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Загружает данные из Parquet.
        
        Args:
            lazy: Если True, возвращает LazyFrame для отложенных вычислений.
                  Если False, загружает данные в память (collect).
        
        Returns:
            pl.DataFrame или pl.LazyFrame в зависимости от параметра lazy.
        """
        # Фильтруем файлы по символу (формат: SYMBOL_TIMESTAMP.parquet)
        pattern = f"{self.symbol}_*.parquet"
        files = list(self.data_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found for {self.symbol} in {self.data_path}")
        
        # Используем scan_parquet для ленивой загрузки и параллелизма
        lf = pl.scan_parquet(str(self.data_path / pattern))
        
        # 1. Валидация схемы (сравнение названий колонок)
        actual_columns = lf.collect_schema().names()
        if actual_columns[:len(self.expected_columns)] != self.expected_columns:
            missing = set(self.expected_columns) - set(actual_columns)
            extra = set(actual_columns) - set(self.expected_columns)
            raise ValueError(
                f"Schema mismatch for {self.symbol}!\n"
                f"Missing in files: {missing}\n"
                f"Extra in files: {extra}"
            )

        # 2. Сортировка по времени (критично для последовательных моделей)
        lf = lf.sort("timestamp_ms")

        if lazy:
            return lf
        
        df = lf.collect()
        print(f"[{self.symbol}] Loaded {len(df)} rows. Memory: {df.estimated_size('mb'):.2f} MB")
        return df
    def load_trades(self, lazy: bool = False) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Загружает все Parquet файлы с публичными сделками из директории raw/ для заданного символа.
        Задача 236: Загрузка trades.parquet файлов.

        Args:
            lazy: если True, возвращает LazyFrame для отложенного выполнения

        Returns:
            DataFrame или LazyFrame с данными trades (timestamp, price, size, side)
        """
        raw_dir = self.data_path
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

        # Находим все Parquet файлы с trades
        trades_files = sorted(raw_dir.glob("trades_*.parquet"))

        if not trades_files:
            print(f"Warning: No trades files found for {self.symbol} in {raw_dir}")
            # Возвращаем пустой DataFrame с правильной схемой
            empty_schema = {
                "timestamp": pl.Int64,
                "price": pl.Float64,
                "size": pl.Float64,
                "side": pl.Utf8
            }
            if lazy:
                return pl.LazyFrame(schema=empty_schema)
            else:
                return pl.DataFrame(schema=empty_schema)

        print(f"Found {len(trades_files)} trades files")

        if lazy:
            # Ленивая загрузка через scan_parquet
            df_trades = pl.scan_parquet(trades_files)
        else:
            # Загрузка в память
            df_trades = pl.read_parquet(trades_files)

        return df_trades

class LOBDataset(Dataset):
    """
    Dataset для PyTorch с использованием скользящего окна.
    Использует Polars для эффективного формирования последовательностей.
    Поддерживает 3 базовых канала (Price, Volume, Imbalance) + N каналов Past Returns.
    
    Поддерживает 3 режима загрузки данных:
    1. memory: Загрузка всех данных в RAM (по умолчанию)
    2. streaming: Lazy-загрузка через Polars streaming engine
    3. memmap: Memory-mapped binary файлы для быстрого случайного доступа
    
    Поддерживает временное взвешивание примеров (time-weighted loss):
    - half_life_hours: период полураспада для экспоненциального затухания весов
    - min_weight: минимальный вес для старых примеров
    - class_weights: веса классов для балансировки (перемножаются с временными весами)
    """
    def __init__(
        self, 
        df: Union[pl.DataFrame, pl.LazyFrame, str], 
        seq_len: int = 100, 
        n_past_returns: int = 3,
        past_returns_lags: Union[List[int], None] = None,  # Задача 091: Лаги для Past Returns
        vol_window: int = 100,
        data_mode: Literal["memory", "streaming", "memmap"] = "memory",
        cache_dir: Union[str, Path, None] = None,
        half_life_hours: float = 24.0,
        min_weight: float = 0.1,
        class_weights: Union[torch.Tensor, np.ndarray, None] = None,
        is_train: bool = False,
        augment_prob: float = 0.5,
        use_symmetric_flip: bool = False,
        volume_jitter_range: float = 0.1,
        aug_seed: int = 42,
        regime_detector = None,  # RegimeDetector для определения режимов рынка
        regime_window: int = 1000,  # Размер окна для расчета признаков режима
        exclude_features: Union[List[str], None] = None,  # Задача 239: Список признаков для исключения (ablation studies)
        scaler_type: str = "zscore",  # Задача 240: Тип масштабирования
        winsor_limits: tuple[float, float] = (0.01, 0.99)  # Задача 240: Пороги винзоризации
    ):
        self.seq_len = seq_len
        self.n_levels = 50
        self.n_past_returns = n_past_returns
        self.past_returns_lags = past_returns_lags if past_returns_lags is not None else [10, 50, 100]  # Задача 091: Лаги по умолчанию
        self.vol_window = vol_window
        self.data_mode = data_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.half_life_hours = half_life_hours
        self.min_weight = min_weight
        self.regime_detector = regime_detector
        self.regime_window = regime_window
        self.exclude_features = exclude_features  # Задача 239: Сохраняем список исключаемых признаков
        self.scaler_type = scaler_type  # Задача 240: Тип масштабирования
        self.winsor_limits = winsor_limits  # Задача 240: Пороги винзоризации
        
        # Параметры аугментации
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.use_symmetric_flip = use_symmetric_flip
        self.volume_jitter_range = volume_jitter_range
        self.generator = torch.Generator().manual_seed(aug_seed)
        
        # Вычисляем индексы для аугментации
        self.price_cols, self.vol_cols, self.ask_cols, self.bid_cols = _compute_augmentation_indices(self.n_levels)
        
        # Конвертируем class_weights в numpy для расчетов
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.cpu().numpy()
            else:
                self.class_weights = np.array(class_weights, dtype=np.float32)
        else:
            self.class_weights = None
        
        # Инициализация в зависимости от режима
        if data_mode == "memory":
            self._init_memory_mode(df)
        elif data_mode == "streaming":
            self._init_streaming_mode(df)
        elif data_mode == "memmap":
            self._init_memmap_mode(df)
        else:
            raise ValueError(f"Unknown data_mode: {data_mode}. Use 'memory', 'streaming', or 'memmap'")

    def _calculate_time_weights(self, timestamps: np.ndarray, labels: np.ndarray) -> torch.Tensor:
        """
        Расчет временных весов с экспоненциальным затуханием.
        
        Args:
            timestamps: массив временных меток в миллисекундах
            labels: массив меток классов
            
        Returns:
            torch.Tensor с весами для каждого примера
        """
        # 1. Расчет временных весов
        max_ts = timestamps.max()
        half_life_ms = self.half_life_hours * 3600 * 1000
        decay_lambda = np.log(2) / half_life_ms
        
        deltas = max_ts - timestamps
        time_weights = np.exp(-decay_lambda * deltas)
        time_weights = np.clip(time_weights, self.min_weight, 1.0)
        
        # 2. Интеграция с весами классов (если заданы)
        if self.class_weights is not None:
            # class_weights — массив весов [w_flat, w_up, w_down]
            sample_class_weights = self.class_weights[labels]
            final_weights = time_weights * sample_class_weights
        else:
            final_weights = time_weights
        
        # 3. Нормализация (среднее = 1.0 для стабильности градиентов)
        final_weights = final_weights / final_weights.mean()
        
        return torch.tensor(final_weights, dtype=torch.float32)

    def _init_memory_mode(self, df: pl.DataFrame):
        """Стандартная загрузка всех данных в RAM"""
        if not isinstance(df, pl.DataFrame):
            raise TypeError("For 'memory' mode, df must be a pl.DataFrame")
        
        # 1. Выделяем признаки и метки
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        
        # Задача 239: Фильтруем исключаемые признаки для ablation studies
        if self.exclude_features is not None:
            feat_cols = [c for c in feat_cols if c not in self.exclude_features]
            print(f"[Ablation] Excluded {len(self.exclude_features)} features. Remaining: {len(feat_cols)}")
        
        # Задача 240: Применяем Winsorization и Robust Scaling через единый интерфейс Normalizer
        if self.scaler_type in ("robust", "winsor_robust"):
            print(f"[Scaler] Applying {self.scaler_type} scaling with winsor_limits={self.winsor_limits}")
            
            # Используем Normalizer как единый интерфейс
            norm = Normalizer("") 
            norm.scaler_type = self.scaler_type
            norm.winsor_limits = self.winsor_limits
            
            # Обучаем на текущем срезе признаков
            df_feat = df.select(feat_cols)
            norm.fit(df_feat, winsor_limits=self.winsor_limits)
            
            # Трансформируем (винзоризация + робастное масштабирование внутри)
            df_norm = norm.transform(df_feat)
            
            self.robust_params = norm.params  # Сохраняем для экспорта
            
            # Заменяем признаки в исходном DataFrame
            df = df.drop(feat_cols).hstack(df_norm)
            
            first_feat = feat_cols[0] if feat_cols else None
            if first_feat:
                p = self.robust_params[first_feat]
                print(f"[Scaler] Applied {self.scaler_type} scaling. {first_feat}: median={p['median']:.4f}, iqr={p['iqr']:.4f}")
        else:
            self.robust_params = None
        
        # Определяем колонки меток (single или multi-horizon)
        if "label" in df.columns:
            # Single horizon (обратная совместимость)
            label_cols = ["label"]
            self.is_multi_horizon = False
        else:
            # Multi-horizon: ищем все колонки label_h*
            label_cols = sorted([c for c in df.columns if c.startswith("label_h")])
            if not label_cols:
                raise ValueError("No label columns found in DataFrame. Expected 'label' or 'label_h*'")
            self.is_multi_horizon = True
            self.num_horizons = len(label_cols)
        
        # 2. Формируем последовательности через Polars shift + concat_list
        select_cols = [
            pl.concat_list(feat_cols).alias("features"),
            *[pl.col(lc) for lc in label_cols],
            pl.col("timestamp_ms")
        ]
        sequence_df = df.select(select_cols)

        # ИСПРАВЛЕНИЕ: Zero-Copy реализация
        # Вместо создания всех последовательностей в памяти, сохраняем исходные данные
        # и делаем срезы "на лету" в __getitem__
        features_series = sequence_df.select(pl.col("features")).to_series()
        
        # Конвертируем в numpy array (N, n_features) - исходные данные без скользящего окна
        self.features = np.stack([np.array(row, dtype=np.float32) for row in features_series.to_list()], axis=0)
        
        # Метки должны соответствовать концу окна (idx + seq_len)
        if self.is_multi_horizon:
            # Multi-horizon: формируем массив (n_samples, num_horizons)
            labels_list = []
            for label_col in label_cols:
                labels_list.append(sequence_df.select(pl.col(label_col)).to_series().to_numpy())
            self.labels = np.stack(labels_list, axis=1)  # (n_samples, num_horizons)
        else:
            # Single horizon: (n_samples,)
            self.labels = sequence_df.select(pl.col("label")).to_series().to_numpy()
        
        # Временные метки для расчета весов
        self.timestamps = sequence_df.select(pl.col("timestamp_ms")).to_series().to_numpy()
        
        # 4. Расчет волатильности (MTL)
        if "mid_price" in df.columns:
            mid_prices = df["mid_price"].to_numpy()
            vols_full = compute_target_vol(mid_prices, window=self.vol_window)
            # Берем волатильность для конца каждого окна (idx + seq_len)
            self.vols = vols_full[self.seq_len - 1:]
        else:
            # Для multi-horizon используем первую размерность labels
            if self.is_multi_horizon:
                self.vols = np.zeros(self.labels.shape[0], dtype=np.float32)
            else:
                self.vols = np.zeros_like(self.labels, dtype=np.float32)

        # 4.5. Расчет Past Returns (Задача 091)
        if self.n_past_returns > 0 and "mid_price" in df.columns:
            print(f"[Past Returns] Computing log-returns for lags: {self.past_returns_lags}")
            mid_prices = df["mid_price"].to_numpy()
            past_returns = compute_past_returns(mid_prices, self.past_returns_lags)  # (N, n_lags)
            
            # Расширяем self.features с past returns
            # Было: (N, 200), Станет: (N, 200 + n_lags)
            self.features = np.hstack([self.features, past_returns])
            print(f"[Past Returns] Features shape: {self.features.shape}")
        else:
            if self.n_past_returns > 0:
                print("[Past Returns] Warning: n_past_returns > 0 but 'mid_price' column not found. Skipping past returns computation.")

        # Расчет временных весов (используем первый горизонт для расчета весов)
        labels_for_weights = self.labels[:, 0] if self.is_multi_horizon else self.labels
        self.sample_weights = self._calculate_time_weights(self.timestamps, labels_for_weights)
        
        # 5. Расчет режимов рынка (если включено)
        if self.regime_detector is not None and self.regime_detector.is_fitted:
            print("[Regime] Computing market regime features...")
            # Вычисляем признаки режима для всех данных
            regime_features = compute_regime_features(df, window=self.regime_window)
            # Предсказываем режимы для каждого момента времени
            all_regime_ids = self.regime_detector.predict_states(regime_features)
            # Берем regime_id для конца каждого окна (соответствует меткам)
            self.regime_ids = all_regime_ids[self.seq_len - 1:]
            print(f"[Regime] Computed regime IDs. Distribution: {np.bincount(self.regime_ids)}")
        else:
            # Если regime_detector не передан, используем dummy значение 0
            # Размер должен быть len(features) - seq_len + 1
            self.regime_ids = np.zeros(len(self.features) - self.seq_len + 1, dtype=np.int64)

    def _init_streaming_mode(self, df: Union[pl.DataFrame, pl.LazyFrame, str]):
        """Lazy-загрузка через Polars streaming с индексной картой"""
        # Если передан путь к файлу, создаем LazyFrame
        if isinstance(df, str):
            df = pl.scan_parquet(df, low_memory=True)
        elif isinstance(df, pl.DataFrame):
            df = df.lazy()
        
        if not isinstance(df, pl.LazyFrame):
            raise TypeError("For 'streaming' mode, df must be LazyFrame, DataFrame, or file path")
        
        # Получаем список колонок признаков
        schema = df.collect_schema()
        feat_cols = [c for c in schema.names() if c.startswith("feat_")]
        
        # Задача 239: Фильтруем исключаемые признаки для ablation studies
        if self.exclude_features is not None:
            feat_cols = [c for c in feat_cols if c not in self.exclude_features]
            print(f"[Ablation] Excluded {len(self.exclude_features)} features. Remaining: {len(feat_cols)}")
        
        # Сохраняем LazyFrame для потоковой обработки
        self.lazy_df = df.select([*feat_cols, "label", "timestamp_ms"])
        
        # Создаем индексную карту: считаем количество строк
        # Используем streaming для подсчета без загрузки всех данных
        total_rows = self.lazy_df.select(pl.len()).collect(streaming=True).item()
        self.total_samples = total_rows - self.seq_len + 1
        
        if self.total_samples <= 0:
            raise ValueError(f"Dataset too small: {total_rows} rows, need at least {self.seq_len}")
        
        # Сохраняем путь к файлу для thread safety в DataLoader
        if isinstance(df, str):
            self.file_path = df
        else:
            # Для LazyFrame/DataFrame пытаемся извлечь путь
            self.file_path = None
        
        # Создаем индексную карту (массив физических смещений строк в Parquet файле)
        # Используем PyArrow для получения информации о row groups
        self.row_offsets = self._build_row_offsets(self.file_path, total_rows)
        
        # Для streaming режима мы будем читать данные батчами при обращении
        self.feat_cols = feat_cols
        self.n_features = len(feat_cols)
        
        # Кэш для последнего загруженного батча (оптимизация последовательного доступа)
        self._cache_batch = None
        self._cache_labels = None
        self._cache_vols = None
        self._cache_start_idx = -1
        self._cache_end_idx = -1
        # ИСПРАВЛЕНИЕ: Увеличиваем размер батча для уменьшения количества обращений к диску
        self._batch_size = 50000  # Увеличено с 10000 до 50000
        
        # Для streaming режима загружаем mid_price для расчета волатильности
        if "mid_price" in schema.names():
            mid_prices = df.select("mid_price").collect(streaming=True).to_series().to_numpy()
            self.vols = compute_target_vol(mid_prices, window=self.vol_window)[self.seq_len - 1:]
        else:
            self.vols = np.zeros(self.total_samples, dtype=np.float32)

        # Для streaming режима загружаем timestamps для расчета весов
        # Загружаем только timestamps (легковесная операция)
        timestamps_df = self.lazy_df.select(pl.col("timestamp_ms")).slice(self.seq_len - 1, self.total_samples).collect(streaming=True)
        self.timestamps = timestamps_df.to_series().to_numpy()
        
        # Загружаем labels для расчета весов
        labels_df = self.lazy_df.select(pl.col("label")).slice(self.seq_len - 1, self.total_samples).collect(streaming=True)
        labels_array = labels_df.to_series().to_numpy()
        
        # Расчет временных весов
        self.sample_weights = self._calculate_time_weights(self.timestamps, labels_array)
        
        # Инициализация regime_ids (для streaming режима используем dummy значения)
        # Regime detection требует полных данных в памяти
        self.regime_ids = np.zeros(self.total_samples, dtype=np.int64)

    def _build_row_offsets(self, file_path: Union[str, None], total_rows: int) -> np.ndarray:
        """
        Создает индексную карту (массив физических смещений строк в Parquet файле).
        Использует PyArrow для получения информации о row groups.
        
        Args:
            file_path: путь к Parquet файлу
            total_rows: общее количество строк в файле
        
        Returns:
            np.ndarray: массив cumulative row offsets для бинарного поиска
        """
        import pyarrow.parquet as pq
        
        if file_path and Path(file_path).exists():
            try:
                pf = pq.ParquetFile(file_path)
                num_row_groups = pf.num_row_groups
                
                # Создаем массив cumulative row counts
                row_counts = np.zeros(num_row_groups, dtype=np.int64)
                for i in range(num_row_groups):
                    row_counts[i] = pf.row_group(i).num_rows
                
                # Преобразуем в cumulative offsets (смещения начала каждой группы)
                row_offsets = np.cumsum(row_counts)
                print(f"[IndexMap] Created row offsets from {num_row_groups} row groups")
                return row_offsets
            except Exception as e:
                print(f"[IndexMap] Warning: Could not read Parquet metadata: {e}")
        
        # Fallback: создаем равномерные смещения на основе общего количества строк
        # Это менее точно, но работает когда файл недоступен
        num_groups = min(100, total_rows)  # Ограничиваем количество групп
        group_size = total_rows // num_groups
        row_offsets = np.arange(group_size, total_rows + group_size, group_size, dtype=np.int64)[:num_groups]
        print(f"[IndexMap] Created fallback row offsets ({num_groups} groups)")
        return row_offsets

    def _init_memmap_mode(self, df: Union[pl.DataFrame, str]):
        """Memory-mapped binary файлы для быстрого случайного доступа"""
        if self.cache_dir is None:
            raise ValueError("cache_dir must be specified for 'memmap' mode")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Пути к memmap файлам
        features_path = self.cache_dir / "features.npy"
        labels_path = self.cache_dir / "labels.npy"
        vols_path = self.cache_dir / "vols.npy"
        timestamps_path = self.cache_dir / "timestamps.npy"
        weights_path = self.cache_dir / "weights.npy"
        metadata_path = self.cache_dir / "metadata.json"
        
        # Проверяем, существуют ли уже файлы
        if features_path.exists() and labels_path.exists() and metadata_path.exists() and timestamps_path.exists() and weights_path.exists() and vols_path.exists():
            # Загружаем существующие memmap файлы
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            shape = tuple(metadata['features_shape'])
            self.features_seq = np.memmap(features_path, dtype='float32', mode='r', shape=shape)
            self.labels = np.memmap(labels_path, dtype='int64', mode='r', shape=(metadata['n_samples'],))
            self.vols = np.memmap(vols_path, dtype='float32', mode='r', shape=(metadata['n_samples'],))
            self.timestamps = np.memmap(timestamps_path, dtype='int64', mode='r', shape=(metadata['n_samples'],))
            self.sample_weights = np.memmap(weights_path, dtype='float32', mode='r', shape=(metadata['n_samples'],))
            
            print(f"[Memmap] Loaded existing cache from {self.cache_dir}")
        else:
            # Создаем новые memmap файлы
            print(f"[Memmap] Creating cache in {self.cache_dir}...")
            
            # Загружаем данные в память для конвертации
            if isinstance(df, str):
                df = pl.read_parquet(df)
            
            feat_cols = [c for c in df.columns if c.startswith("feat_")]
            
            # Задача 239: Фильтруем исключаемые признаки для ablation studies
            if self.exclude_features is not None:
                feat_cols = [c for c in feat_cols if c not in self.exclude_features]
                print(f"[Ablation] Excluded {len(self.exclude_features)} features. Remaining: {len(feat_cols)}")
            
            # Формируем последовательности (аналогично memory mode)
            sequence_df = df.select([
                pl.concat_list(feat_cols).alias("features"),
                pl.col("label"),
                pl.col("timestamp_ms")
            ])
            
            exprs = [pl.col("features").shift(i) for i in range(self.seq_len - 1, -1, -1)]
            
            # ИСПРАВЛЕНИЕ: Chunk-by-chunk запись для экономии памяти
            # Сначала считаем размер и создаем memmap файлы
            total_rows = len(sequence_df) - self.seq_len + 1
            n_features = len(feat_cols)
            
            # Создаем пустые memmap файлы нужного размера
            features_memmap = np.memmap(
                features_path, 
                dtype='float32', 
                mode='w+', 
                shape=(total_rows, self.seq_len, n_features)
            )
            
            labels_memmap = np.memmap(
                labels_path,
                dtype='int64',
                mode='w+',
                shape=(total_rows,)
            )
            
            vols_memmap = np.memmap(
                vols_path,
                dtype='float32',
                mode='w+',
                shape=(total_rows,)
            )
            
            timestamps_memmap = np.memmap(
                timestamps_path,
                dtype='int64',
                mode='w+',
                shape=(total_rows,)
            )
            
            # Рассчитываем волатильность заранее
            if "mid_price" in df.columns:
                mid_prices = df["mid_price"].to_numpy()
                vols_array = compute_target_vol(mid_prices, window=self.vol_window)[self.seq_len - 1:]
            else:
                vols_array = np.zeros(total_rows, dtype=np.float32)

            # Записываем данные чанками для экономии памяти
            chunk_size = 10000  # Обрабатываем по 10k строк за раз
            
            for chunk_start in range(0, total_rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_rows)
                
                # Читаем чанк с учетом seq_len
                chunk_df = sequence_df.slice(chunk_start, chunk_end - chunk_start + self.seq_len - 1)
                
                # Формируем последовательности для чанка
                chunk_features_list = (
                    chunk_df.select(
                        pl.concat_list(exprs).alias("features_seq")
                    )
                    .slice(self.seq_len - 1)
                    .to_series()
                )
                
                chunk_labels = chunk_df.select(pl.col("label")).slice(self.seq_len - 1).to_series().to_numpy()
                chunk_timestamps = chunk_df.select(pl.col("timestamp_ms")).slice(self.seq_len - 1).to_series().to_numpy()
                
                # Конвертируем в numpy и записываем в memmap
                chunk_features = np.stack([np.array(row, dtype=np.float32) for row in chunk_features_list.to_list()], axis=0)
                
                features_memmap[chunk_start:chunk_end] = chunk_features
                labels_memmap[chunk_start:chunk_end] = chunk_labels
                vols_memmap[chunk_start:chunk_end] = vols_array[chunk_start:chunk_end]
                timestamps_memmap[chunk_start:chunk_end] = chunk_timestamps
                
                print(f"[Memmap] Processed {chunk_end}/{total_rows} samples...")
            
            # Flush данных на диск
            features_memmap.flush()
            labels_memmap.flush()
            vols_memmap.flush()
            timestamps_memmap.flush()
            
            # Расчет временных весов
            weights_array = self._calculate_time_weights(timestamps_memmap[:], labels_memmap[:])
            
            # Создаем memmap для весов
            weights_memmap = np.memmap(
                weights_path,
                dtype='float32',
                mode='w+',
                shape=(total_rows,)
            )
            weights_memmap[:] = weights_array
            weights_memmap.flush()
            
            # Сохраняем метаданные
            metadata = {
                'features_shape': list(features_memmap.shape),
                'n_samples': total_rows,
                'seq_len': self.seq_len,
                'n_past_returns': self.n_past_returns
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Переоткрываем в режиме чтения
            self.features_seq = np.memmap(features_path, dtype='float32', mode='r', shape=features_memmap.shape)
            self.labels = np.memmap(labels_path, dtype='int64', mode='r', shape=labels_memmap.shape)
            self.vols = np.memmap(vols_path, dtype='float32', mode='r', shape=vols_memmap.shape)
            self.timestamps = np.memmap(timestamps_path, dtype='int64', mode='r', shape=timestamps_memmap.shape)
            self.sample_weights = np.memmap(weights_path, dtype='float32', mode='r', shape=weights_memmap.shape)
            
            print(f"[Memmap] Cache created successfully")
        
        # Инициализация regime_ids (для memmap режима используем dummy значения)
        # Regime detection требует полных данных в памяти
        self.regime_ids = np.zeros_like(self.labels, dtype=np.int64)

    def __len__(self):
        if self.data_mode == "streaming":
            return self.total_samples
        else:
            # Для memory и memmap: количество окон = len(features) - seq_len + 1
            return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        if self.data_mode == "streaming":
            return self._getitem_streaming(idx)
        else:
            return self._getitem_standard(idx)

    def _getitem_standard(self, idx):
        """Стандартный доступ для memory и memmap режимов"""
        # Zero-copy срез: берем окно прямо из оригинального массива
        # x_raw: (seq_len, 200 + n_past_returns) где 200 = 50*4 (ask_p, ask_v, bid_p, bid_v)
        # Структура согласно задаче 012: [ask_p_0..49, ask_v_0..49, bid_p_0..49, bid_v_0..49]
        x_raw = self.features[idx : idx + self.seq_len]
        
        # Разделяем на LOB признаки и past returns
        lob_features_flat = x_raw[:, :200]  # (seq_len, 200)
        
        # Преобразуем в torch tensor для аугментации
        x = torch.from_numpy(lob_features_flat.copy())
        y = self.labels[idx + self.seq_len - 1]  # Метка для конца окна
        v = self.vols[idx]
        w = self.sample_weights[idx]
        
        # Применяем аугментацию (только для тренировочного режима)
        if self.is_train and torch.rand(1, generator=self.generator).item() < self.augment_prob:
            x_aug, y_aug = x.clone(), y
            
            # 1. Применяем Symmetric Flip
            if self.use_symmetric_flip and torch.rand(1, generator=self.generator).item() < 0.5:
                x_aug, y_aug = apply_symmetric_flip(x_aug, y_aug, self.price_cols, self.ask_cols, self.bid_cols)
            
            # 2. Применяем Volume Jitter
            if self.volume_jitter_range > 0:
                x_aug = apply_volume_jitter(x_aug, self.volume_jitter_range, self.vol_cols, self.generator)
            
            # 3. Проверка консистентности: Best Bid (отрицательный) < Best Ask (положительный)
            # В нормализованных данных (p-mid)/mid: Best Bid < 0 < Best Ask
            # Индексы: ask_p_0 = 0, bid_p_0 = 100
            if x_aug[0, 100] < x_aug[0, 0]:  # bid_p_0 < ask_p_0
                x, y = x_aug, y_aug
        
        # Реализуем расчет 3-канального тензора согласно плану 053
        # Структура x: [ask_p_0..49, ask_v_0..49, bid_p_0..49, bid_v_0..49] (seq_len, 200)
        ask_p = x[:, 0:50]      # (seq_len, 50)
        ask_v = x[:, 50:100]    # (seq_len, 50)
        bid_p = x[:, 100:150]   # (seq_len, 50)
        bid_v = x[:, 150:200]   # (seq_len, 50)
        
        # Канал 0: Normalized Price (среднее отклонение)
        # Используем уже нормализованные цены (p-mid)/mid
        price_ch = (ask_p + bid_p) / 2.0  # (seq_len, 50)
        
        # Канал 1: Log Volume
        vol_ch = ask_v + bid_v  # (seq_len, 50)
        
        # Канал 2: Static Level Imbalance
        # Формула: (Vbid - Vask) / (Vbid + Vask + eps)
        imb_ch = (bid_v - ask_v) / (bid_v + ask_v + 1e-7)  # (seq_len, 50)
        
        # Собираем 3-канальный тензор: (seq_len, 3, 50)
        x_reshaped = torch.stack([price_ch, vol_ch, imb_ch], dim=1)
        
        # Добавляем past returns если есть
        if self.n_past_returns > 0:
            past_returns = torch.from_numpy(x_raw[:, 200:200+self.n_past_returns].copy())
            # Broadcast на 50 уровней: (seq_len, n_past_returns, 50)
            past_returns_broadcast = past_returns.unsqueeze(-1).repeat(1, 1, self.n_levels)
            # Объединяем: (seq_len, 3+n_past_returns, 50)
            x_final = torch.cat([x_reshaped, past_returns_broadcast], dim=1)
        else:
            x_final = x_reshaped
        
        # Получаем regime_id
        regime_id = torch.tensor(self.regime_ids[idx]).long()
        
        return x_final, torch.tensor(y).long(), torch.tensor(v).float(), w, regime_id

    def _getitem_streaming(self, idx):
        """Потоковый доступ с бинарным поиском по индексной карте"""
        # Используем бинарный поиск для мгновенного нахождения нужной строки
        # Это обеспечивает O(log n) вместо O(n) перебора
        if hasattr(self, 'row_offsets') and self.row_offsets is not None:
            # Находим row group через бинарный поиск
            row_group_idx = np.searchsorted(self.row_offsets, idx, side='right')
            # Корректировка: row_offsets содержит cumulative counts, поэтому нужна проверка
            if row_group_idx > 0:
                row_start = self.row_offsets[row_group_idx - 1]
            else:
                row_start = 0
            # Смещение внутри row group
            offset_in_group = idx - row_start
        else:
            # Fallback если row_offsets не доступен
            row_group_idx = 0
            offset_in_group = idx
        
        # Проверяем, есть ли нужные данные в кэше
        if self._cache_batch is not None and self._cache_start_idx <= idx < self._cache_end_idx:
            cache_offset = idx - self._cache_start_idx
            x_raw = self._cache_batch[cache_offset]
            y = self._cache_labels[cache_offset]
            v = self._cache_vols[cache_offset]
        else:
            # Загружаем новый батч с учетом позиции в row group
            start_row = idx
            end_row = min(idx + self._batch_size, self.total_samples + self.seq_len - 1)
            
            # Читаем батч через streaming
            batch_df = (
                self.lazy_df
                .slice(start_row, end_row - start_row + self.seq_len - 1)
                .collect(streaming=True)
            )
            
            # Формируем последовательности для батча
            feat_cols = self.feat_cols
            
            # Определяем колонки меток (single или multi-horizon)
            if self.is_multi_horizon:
                label_cols = [f"label_h{h}" for h in range(self.num_horizons)]
            else:
                label_cols = ["label"]
            
            sequence_df = batch_df.select([
                pl.concat_list(feat_cols).alias("features"),
                *[pl.col(lc) for lc in label_cols]
            ])
            
            exprs = [pl.col("features").shift(i) for i in range(self.seq_len - 1, -1, -1)]
            
            features_list = (
                sequence_df.select(
                    pl.concat_list(exprs).alias("features_seq")
                )
                .slice(self.seq_len - 1)
                .to_series()
                .to_list()
            )
            
            # Метки для multi-horizon или single
            if self.is_multi_horizon:
                labels_list = []
                for label_col in label_cols:
                    labels_list.append(sequence_df.select(pl.col(label_col)).slice(self.seq_len - 1).to_series().to_numpy())
                labels = np.stack(labels_list, axis=1)  # (n_samples, num_horizons)
            else:
                labels = sequence_df.select(pl.col("label")).slice(self.seq_len - 1).to_series().to_numpy()
            
            # Волатильность берется из пре-рассчитанного массива
            vols = self.vols[start_row : start_row + len(labels)]
            
            # ИСПРАВЛЕНИЕ: Используем np.stack вместо np.array для эффективности
            self._cache_batch = np.stack([np.array(row, dtype=np.float32) for row in features_list], axis=0)
            self._cache_labels = labels
            self._cache_vols = vols
            self._cache_start_idx = start_row
            self._cache_end_idx = start_row + len(self._cache_batch)
            
            # Получаем нужный элемент
            cache_offset = idx - self._cache_start_idx
            x_raw = self._cache_batch[cache_offset]
            y = self._cache_labels[cache_offset]
            v = self._cache_vols[cache_offset]
        
        # Преобразуем в нужный формат
        # x_raw имеет форму (seq_len, n_features) где n_features = 200 + n_past_returns
        lob_features_flat = x_raw[:, :200]  # (seq_len, 200)
        
        # Преобразуем в torch tensor для аугментации
        x = torch.from_numpy(lob_features_flat.copy())
        
        # Применяем аугментацию (только для тренировочного режима)
        if self.is_train and torch.rand(1, generator=self.generator).item() < self.augment_prob:
            x_aug, y_aug = x.clone(), y
            
            # 1. Применяем Symmetric Flip
            if self.use_symmetric_flip and torch.rand(1, generator=self.generator).item() < 0.5:
                x_aug, y_aug = apply_symmetric_flip(x_aug, y_aug, self.price_cols, self.ask_cols, self.bid_cols)
            
            # 2. Применяем Volume Jitter
            if self.volume_jitter_range > 0:
                x_aug = apply_volume_jitter(x_aug, self.volume_jitter_range, self.vol_cols, self.generator)
            
            # 3. Проверка консистентности: Best Bid (отрицательный) < Best Ask (положительный)
            # Индексы: ask_p_0 = 0, bid_p_0 = 100
            if x_aug[0, 100] < x_aug[0, 0]:  # bid_p_0 < ask_p_0
                x, y = x_aug, y_aug
        
        # Реализуем расчет 3-канального тензора согласно плану 053
        # Структура x: [ask_p_0..49, ask_v_0..49, bid_p_0..49, bid_v_0..49] (seq_len, 200)
        ask_p = x[:, 0:50]      # (seq_len, 50)
        ask_v = x[:, 50:100]    # (seq_len, 50)
        bid_p = x[:, 100:150]   # (seq_len, 50)
        bid_v = x[:, 150:200]   # (seq_len, 50)
        
        # Канал 0: Normalized Price (среднее отклонение)
        # Используем уже нормализованные цены (p-mid)/mid
        price_ch = (ask_p + bid_p) / 2.0  # (seq_len, 50)
        
        # Канал 1: Log Volume
        vol_ch = ask_v + bid_v  # (seq_len, 50)
        
        # Канал 2: Static Level Imbalance
        # Формула: (Vbid - Vask) / (Vbid + Vask + eps)
        imb_ch = (bid_v - ask_v) / (bid_v + ask_v + 1e-7)  # (seq_len, 50)
        
        # Собираем 3-канальный тензор: (seq_len, 3, 50)
        x_reshaped = torch.stack([price_ch, vol_ch, imb_ch], dim=1)
        
        # Добавляем past returns если есть
        if self.n_past_returns > 0:
            past_returns = torch.from_numpy(x_raw[:, 200:200+self.n_past_returns].copy())
            # Broadcast на 50 уровней: (seq_len, n_past_returns, 50)
            past_returns_broadcast = past_returns.unsqueeze(-1).repeat(1, 1, self.n_levels)
            # Объединяем: (seq_len, 3+n_past_returns, 50)
            x_final = torch.cat([x_reshaped, past_returns_broadcast], dim=1)
        else:
            x_final = x_reshaped
        
        # Получаем вес для этого примера
        w = self.sample_weights[idx]
        
        # Получаем regime_id (если доступен)
        if self.regime_ids is not None:
            regime_id = torch.tensor(self.regime_ids[idx]).long()
        else:
            regime_id = torch.tensor(0).long()
        
        return x_final, torch.tensor(y).long(), torch.tensor(v).float(), w, regime_id

    def get_class_distribution(self) -> np.ndarray:
        """
        Возвращает количество примеров для каждого класса (Up, Down, Flat).
        """
        if self.data_mode == "streaming":
            # Для streaming режима считаем через Polars
            label_counts = (
                self.lazy_df
                .select(pl.col("label"))
                .slice(self.seq_len - 1, self.total_samples)
                .group_by("label")
                .agg(pl.len().alias("count"))
                .collect(streaming=True)
            )
            
            full_counts = np.zeros(3, dtype=np.int64)
            for row in label_counts.iter_rows():
                label, count = row
                if 0 <= label < 3:
                    full_counts[int(label)] = count
            
            return full_counts
        else:
            classes, counts = np.unique(self.labels, return_counts=True)
            
            full_counts = np.zeros(3, dtype=np.int64)
            for cls, count in zip(classes, counts):
                if 0 <= cls < 3:
                    full_counts[int(cls)] = count
            
            return full_counts
    
    def get_timestamps(self) -> np.ndarray:
        """
        Возвращает массив временных меток (timestamp_ms) для всех сэмплов в датасете.
        
        Используется для Purged K-Fold кросс-валидации для проверки сортировки
        и корректного разбиения данных по времени.
        
        Returns:
            np.ndarray: массив временных меток в миллисекундах
        """
        return self.timestamps


# Алиас для обратной совместимости и использования в tune.py
LOBPyTorchDataset = LOBDataset


def get_val_loader(
    data_path: str,
    symbol: str,
    seq_len: int = 100,
    n_past_returns: int = 0,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 4,
    data_mode: str = "memory",
    val_split: float = 0.8
):
    """
    Создает DataLoader для валидационной выборки.
    
    Args:
        data_path: путь к директории с parquet файлами
        symbol: торговый символ
        seq_len: длина последовательности
        n_past_returns: количество past returns каналов
        batch_size: размер батча
        shuffle: перемешивать ли данные (для feature importance должно быть False)
        num_workers: количество воркеров для загрузки
        data_mode: режим загрузки данных ('memory', 'streaming', 'memmap')
        val_split: доля данных для обучения (остальное - валидация)
    
    Returns:
        DataLoader: загрузчик валидационных данных
    """
    from torch.utils.data import DataLoader
    import polars as pl
    
    # Загружаем данные
    data_path = Path(data_path)
    pattern = f"{symbol}_*.parquet"
    files = list(data_path.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No parquet files found for {symbol} in {data_path}")
    
    df = pl.read_parquet(data_path / pattern)
    
    # Создаем валидационную выборку (последние 20% данных)
    val_start = int(len(df) * val_split)
    val_df = df.slice(val_start)
    
    # Создаем датасет
    val_ds = LOBPyTorchDataset(
        val_df,
        seq_len=seq_len,
        n_past_returns=n_past_returns,
        data_mode=data_mode,
        is_train=False  # Отключаем аугментацию для валидации
    )
    
    # Создаем DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return val_loader

if __name__ == "__main__":
    # Тестовый запуск
    import sys
    
    # Путь к данным по умолчанию для теста
    test_symbol = "BTCUSDT"
    test_path = Path(__file__).parent.parent.parent / "bots" / test_symbol / "data" / "raw"
    
    print(f"Testing LOBDataLoader for {test_symbol}...")
    try:
        loader = LOBDataLoader(str(test_path), test_symbol)
        df = loader.load_data(lazy=False)
        print("First 5 rows:")
        print(df.select(["timestamp_ms", "ask_p_0", "bid_p_0"]).head())
    except Exception as e:
        print(f"Note: Test run skipped or failed (likely no data yet): {e}")


class LOBCalibrationDataReader(CalibrationDataReader):
    """
    Калибровочный загрузчик данных для статического квантования ONNX модели.
    
    Используется для PTQ (Post-Training Quantization) - загружает репрезентативный
    набор данных из валидационной выборки для калибровки диапазонов активаций.
    
    Args:
        onnx_model_path: путь к FP32 ONNX модели (для извлечения имени входного тензора)
        data_path: путь к директории с parquet файлами
        symbol: торговый символ
        seq_len: длина последовательности
        n_past_returns: количество past returns каналов
        n_samples: количество снапшотов для калибровки (500-1000 рекомендуется)
        val_split: доля данных для обучения (остальное - валидация)
    """
    def __init__(
        self,
        onnx_model_path: str,
        data_path: str,
        symbol: str,
        seq_len: int = 100,
        n_past_returns: int = 0,
        n_samples: int = 1000,
        val_split: float = 0.8
    ):
        self.onnx_model_path = onnx_model_path
        self.data_path = Path(data_path)
        self.symbol = symbol
        self.seq_len = seq_len
        self.n_past_returns = n_past_returns
        self.n_samples = n_samples
        self.val_split = val_split
        
        # Динамически извлекаем имя входного тензора из ONNX модели
        self.input_name = self._get_input_name()
        
        # Загружаем калибровочные данные
        self.calibration_data = self._load_calibration_data()
        
        # Итератор для get_next()
        self.enum_data = None
    
    def _get_input_name(self) -> str:
        """
        Динамически извлекает имя входного тензора из ONNX модели.
        Это делает скрипт универсальным для любых изменений в lit_model.py.
        """
        model = onnx.load(self.onnx_model_path)
        input_name = model.graph.input[0].name
        print(f"[Calibration] Extracted input name from ONNX model: {input_name}")
        return input_name
    
    def _load_calibration_data(self) -> List[np.ndarray]:
        """
        Загружает калибровочные данные из валидационной выборки.
        Возвращает список numpy массивов формы (seq_len, channels, n_levels).
        """
        print(f"[Calibration] Loading {self.n_samples} samples from validation set...")
        
        # Загружаем данные
        pattern = f"{self.symbol}_*.parquet"
        files = list(self.data_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found for {self.symbol} in {self.data_path}")
        
        df = pl.read_parquet(self.data_path / pattern)
        
        # Создаем валидационную выборку (последние 20% данных)
        val_start = int(len(df) * self.val_split)
        val_df = df.slice(val_start)
        
        # Создаем датасет
        val_ds = LOBPyTorchDataset(
            val_df,
            seq_len=self.seq_len,
            n_past_returns=self.n_past_returns,
            data_mode="memory",
            is_train=False  # Отключаем аугментацию для калибровки
        )
        
        # Берем первые n_samples примеров
        n_samples = min(self.n_samples, len(val_ds))
        calibration_data = []
        
        for i in range(n_samples):
            x, _, _, _, _ = val_ds[i]  # x: (seq_len, channels, n_levels)
            calibration_data.append(x.numpy())
        
        print(f"[Calibration] Loaded {len(calibration_data)} samples. Shape: {calibration_data[0].shape}")
        return calibration_data
    
    def get_next(self):
        """
        Возвращает следующий батч данных для калибровки.
        Формат: {'input_name': batch_tensor.numpy()}
        """
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: data} for data in self.calibration_data])
        return next(self.enum_data, None)
    
    def rewind(self):
        """Сбрасывает итератор для повторного прохода по данным."""
        self.enum_data = None


# ============================================================================
# Задача 213: Поддержка мульти-инструментальности
# ============================================================================

def load_multi_symbol_data(
    symbols: List[str],
    data_path: str = "bots",
    lazy: bool = True
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Загружает и объединяет данные Order Book для нескольких торговых пар.
    
    Функция обеспечивает:
    - Детерминированное слияние потоков данных
    - Глобальную сортировку по timestamp для предотвращения look-ahead bias
    - Ленивую загрузку для экономии памяти
    - Задача 212: Объединение снимков стакана с данными о сделках (trades)
    
    Args:
        symbols: Список символов (например, ['BTCUSDT', 'ETHUSDT'])
        data_path: Базовый путь к данным (по умолчанию 'bots')
        lazy: Если True, возвращает LazyFrame; если False, загружает в память
    
    Returns:
        pl.DataFrame или pl.LazyFrame с объединенными данными и колонкой 'symbol'
    
    Raises:
        FileNotFoundError: Если не найдены Parquet файлы для какого-либо символа
        ValueError: Если список символов пуст
    """
    if not symbols:
        raise ValueError("Список символов не может быть пустым")
    
    data_path = Path(data_path)
    scans = []
    
    # Загружаем Parquet файлы для каждого символа и добавляем колонку symbol
    # Задача 213: Используем sorted() для детерминированного порядка обработки
    for symbol in sorted(symbols):
        pattern = f"{symbol}_*.parquet"
        files = list(data_path.glob(f"{symbol}/data/raw/{pattern}"))
        
        if not files:
            raise FileNotFoundError(
                f"Не найдены Parquet файлы для {symbol} в {data_path}/{symbol}/data/raw/"
            )
        
        # Используем scan_parquet для ленивой загрузки снимков стакана
        lf = pl.scan_parquet(data_path / symbol / "data" / "raw" / pattern)
        
        # Добавляем колонку symbol для идентификации источника данных
        lf = lf.with_columns(pl.lit(symbol).alias("symbol"))

        # Задача 212: Загружаем файлы сделок (trades_*.parquet) и объединяем со снимками
        trades_pattern = "trades_*.parquet"
        trades_files = list(data_path.glob(f"{symbol}/data/raw/{trades_pattern}"))

        if trades_files:
            # Загружаем все файлы сделок для символа
            trades_lf = pl.scan_parquet(data_path / symbol / "data" / "raw" / trades_pattern)

            # Нормализуем timestamp: в Rust trades пишутся с колонкой "timestamp" (не timestamp_ms)
            schema = trades_lf.collect_schema()
            if "timestamp" in schema.names() and "timestamp_ms" not in schema.names():
                trades_lf = trades_lf.rename({"timestamp": "timestamp_ms"})

            # Берём только нужные колонки сделок
            trades_lf = trades_lf.select([
                pl.col("timestamp_ms"),
                pl.col("price").alias("trade_price"),
                pl.col("size").alias("trade_volume"),
            ])

            # join_asof: для каждого снимка стакана берём последнюю сделку <= timestamp_ms
            # Сортировка required для join_asof
            lf = lf.sort("timestamp_ms")
            trades_lf = trades_lf.sort("timestamp_ms")

            lf = lf.join_asof(
                trades_lf,
                on="timestamp_ms",
                strategy="backward"
            )
        else:
            # Если файлов сделок нет — заполняем нулями, чтобы схема была консистентна
            print(f"[{symbol}] Предупреждение: файлы trades_*.parquet не найдены. "
                  f"trade_price и trade_volume будут null.")
            lf = lf.with_columns([
                pl.lit(None).cast(pl.Float64).alias("trade_price"),
                pl.lit(None).cast(pl.Float64).alias("trade_volume"),
            ])
        
        scans.append(lf)
    
    # Объединяем все потоки данных
    merged = pl.concat(scans)
    
    # Глобальная сортировка по timestamp и symbol для детерминированности
    # Если временные метки совпадают, порядок символов должен быть фиксированным
    # и предотвращения look-ahead bias
    merged = merged.sort(["timestamp_ms", "symbol"])
    
    if lazy:
        return merged
    
    # Если требуется загрузить в память
    df = merged.collect()
    total_rows = len(df)
    memory_mb = df.estimated_size('mb')
    print(f"[Multi-Symbol] Загружено {total_rows} строк. Память: {memory_mb:.2f} MB")
    print(f"[Multi-Symbol] Символы: {', '.join(symbols)}")
    
    return df



def load_symbol_config(symbol: str, config_path: str = "bots") -> Dict[str, Any]:
    """
    Загружает конфигурацию для конкретного символа из TOML файла.
    
    Функция обеспечивает:
    - Поддержку Python 3.11+ (tomllib) и более старых версий (tomli)
    - Загрузку индивидуальных конфигураций для каждого символа
    - Валидацию наличия файла конфигурации
    
    Args:
        symbol: Символ торговой пары (например, 'BTCUSDT')
        config_path: Базовый путь к конфигурациям (по умолчанию 'bots')
    
    Returns:
        Словарь с конфигурацией символа
    
    Raises:
        FileNotFoundError: Если файл конфигурации не найден
        ValueError: Если файл конфигурации некорректен
    """
    # Попытаемся использовать встроенный tomllib (Python 3.11+)
    try:
        import tomllib
    except ImportError:
        # Для Python < 3.11 используем tomli
        try:
            import tomli as tomllib
        except ImportError:
            raise ImportError(
                "Требуется tomllib (Python 3.11+) или tomli. "
                "Установите: pip install tomli"
            )
    
    config_file = Path(config_path) / symbol / "config.toml"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {config_file.absolute()}"
        )
    
    try:
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
        
        print(f"[{symbol}] Конфигурация загружена из {config_file}")
        return config
    
    except Exception as e:
        raise ValueError(
            f"Ошибка при загрузке конфигурации для {symbol}: {str(e)}"
        )


def load_multi_symbol_configs(
    symbols: List[str],
    config_path: str = "bots"
) -> Dict[str, Dict[str, Any]]:
    """
    Загружает конфигурации для нескольких символов.
    
    Args:
        symbols: Список символов
        config_path: Базовый путь к конфигурациям
    
    Returns:
        Словарь {symbol: config_dict}
    """
    configs = {}
    for symbol in symbols:
        configs[symbol] = load_symbol_config(symbol, config_path)
    
    return configs


if __name__ == "__main__":
    try:
        # Тестовый запуск загрузчика данных
        loader = LOBDataLoader("bots/BTCUSDT/data/raw", "BTCUSDT")
        df = loader.load_data(lazy=False)
        print(df.select(["timestamp_ms", "ask_p_0", "bid_p_0"]).head())
    except Exception as e:
        print(f"Error loading dataset: {e}")
