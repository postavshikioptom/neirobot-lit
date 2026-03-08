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
        df_trades: DataFrame с публичными сделками (timestamp_ms, price, size, side)
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
    
    # Задача 306: Гарантируем имя timestamp_ms
    if "timestamp" in df_trades.columns and "timestamp_ms" not in df_trades.columns:
        df_trades = df_trades.rename({"timestamp": "timestamp_ms"})
    
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
    
    # 3. Преобразуем timestamp_ms в datetime для rolling_sum_by
    # timestamp_ms в trades - это миллисекунды (i64), преобразуем в datetime
    df_trades = df_trades.with_columns(
        pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms").alias("datetime")
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
        # Пытаемся найти feat_ колонки и label - сортируем для детерминизма
        feat_and_label_cols = sorted([c for c in all_cols if c.startswith("feat_") or c == "label" or c.startswith("label_h")])

        # Пересечение с реальными колонками в файле
        columns = [c for c in (target_cols + lob_cols + trade_cols + feat_and_label_cols) if c in all_cols]

    # Итерируемся батчами через PyArrow (True Streaming)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        # Конвертируем PyArrow Batch -> Polars DataFrame (zero-copy где возможно)
        yield pl.from_arrow(batch)

def _compute_augmentation_indices(n_levels: int):
    """
    Вычисляет индексы колонок для аугментации стакана глубиной n_levels.
    Новый порядок согласно задаче 304: 
    [ask_p_0..49, ask_v_0..49, bid_p_0..49, bid_v_0..49]
    
    Returns:
        tuple: (PRICE_COLS, VOL_COLS, ASK_COLS, BID_COLS)
    """
    # ASK_COLS = блоки ask_p и ask_v (индексы 0..99)
    ASK_COLS = list(range(0, 100))
    
    # BID_COLS = блоки bid_p и bid_v (индексы 100..199)
    BID_COLS = list(range(100, 200))
    
    # PRICE_COLS = ask_p (0..49) и bid_p (100..149)
    PRICE_COLS = list(range(0, 50)) + list(range(100, 150))
    
    # VOL_COLS = ask_v (50..99) и bid_v (150..199)
    VOL_COLS = list(range(50, 100)) + list(range(150, 200))
    
    return PRICE_COLS, VOL_COLS, ASK_COLS, BID_COLS


def apply_symmetric_flip(features, label, price_cols, ask_cols, bid_cols):
    """
    Зеркальное отражение стакана.
    1. Меняем блоки Ask и Bid местами.
    2. Инвертируем знак относительных цен.
    3. Инвертируем метку Up (0) <-> Down (1).
    
    Args:
        features: torch.Tensor формы (seq_len, num_features)
        label: int или array - метка класса
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
    if isinstance(label, (int, np.integer)):
        new_label = label
        if label == 1:
            new_label = 2  # Up -> Down
        elif label == 2:
            new_label = 1  # Down -> Up
    else:
        # Multi-horizon labels
        new_label = label.copy()
        up_mask = (label == 1)
        down_mask = (label == 2)
        new_label[up_mask] = 2
        new_label[down_mask] = 1
    
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
    """
    for step in sequence:
        # Лучший ask и bid (индекс 0 в соответствующих блоках)
        # Блок Ask: Price 0-49, Volume 50-99
        # Блок Bid: Price 100-149, Volume 150-199
        ask_p0 = step[0]
        bid_p0 = step[100]
        
        # Спред схлопнулся или отрицательный
        # В нормализованном виде: ask_p0 > 0 > bid_p0
        if ask_p0 <= bid_p0: 
            return False 
            
        # Цены не могут быть <= -1.0 (в нормализованном виде (p-mid)/mid это p <= 0)
        # Проверка цен Ask (0-49)
        for i in range(50):
            if step[i] <= -1.0:
                return False
        
        # Проверка цен Bid (100-149)
        for i in range(100, 150):
            if step[i] >= 1.0:
                return False
            
        # Объемы логарифмированы log(1+v), должны быть >= 0
        # Проверка объемов Ask (50-99)
        for i in range(50, 100):
            if step[i] < 0:
                return False
        
        # Проверка объемов Bid (150-199)
        for i in range(150, 200):
            if step[i] < 0:
                return False
        
    return True

def balance_dataset(features, labels, method='bgmm', ratio=0.5, sampling_strategy=None):
    """
    features: (N, Seq_Len, Feats) - numpy array
    labels: (N,) - numpy array
    """
    from smote_variants import BGMM_SMOTE
    
    orig_shape = features.shape # (N, S, F)
    n_samples = orig_shape[0]
    
    features_2d = features.reshape(n_samples, -1)
    
    if sampling_strategy is None:
        counts = np.bincount(labels)
        if len(counts) < 3:
            full_counts = np.zeros(3, dtype=int)
            full_counts[:len(counts)] = counts
            counts = full_counts
            
        maj_class = np.argmax(counts)
        target_count = int(counts[maj_class] * ratio)
        sampling_strategy = {1: max(counts[1], target_count), 2: max(counts[2], target_count)}
    
    if method == 'bgmm':
        sampler = BGMM_SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'smote':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    else:
        return features, labels

    X_res, y_res = sampler.fit_resample(features_2d, labels)
    X_res = X_res.reshape(-1, orig_shape[1], orig_shape[2])
    
    mask = [validate_lob_sequence(seq) for seq in X_res]
    X_res_filt = X_res[mask]
    y_res_filt = y_res[mask]
    
    return X_res_filt, y_res_filt

class LOBDataLoader:
    def __init__(self, data_path: str, symbol: str, schema_path: str = "../../docs/data_schema.json"):
        self.data_path = Path(data_path)
        self.symbol = symbol
        self.schema_path = Path(__file__).parent.parent.parent / "docs" / "data_schema.json" if schema_path == "../../docs/data_schema.json" else Path(schema_path)
        self.expected_columns = self._load_schema()

    def _load_schema(self) -> List[str]:
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
            return schema["columns"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found at {self.schema_path.absolute()}")

    def load_data(self, lazy: bool = False) -> Union[pl.DataFrame, pl.LazyFrame]:
        pattern = f"{self.symbol}_*.parquet"
        files = list(self.data_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found for {self.symbol} in {self.data_path}")
        
        lf = pl.scan_parquet(str(self.data_path / pattern))
        actual_columns = lf.collect_schema().names()
        if actual_columns[:len(self.expected_columns)] != self.expected_columns:
            missing = set(self.expected_columns) - set(actual_columns)
            extra = set(actual_columns) - set(self.expected_columns)
            raise ValueError(f"Schema mismatch for {self.symbol}!\nMissing: {missing}\nExtra: {extra}")

        lf = lf.sort("timestamp_ms")
        return lf if lazy else lf.collect()

    def load_trades(self, lazy: bool = False) -> Union[pl.LazyFrame, pl.DataFrame]:
        trades_files = sorted(self.data_path.glob("trades_*.parquet"))
        # Задача 306: Используем только timestamp_ms
        empty_schema = {"timestamp_ms": pl.Int64, "price": pl.Float64, "size": pl.Float64, "side": pl.Utf8}
        
        if not trades_files:
            return pl.LazyFrame(schema=empty_schema) if lazy else pl.DataFrame(schema=empty_schema)
        
        lf = pl.scan_parquet(trades_files)
        # Если в файле колонка называется timestamp, переименовываем в timestamp_ms
        actual_cols = lf.collect_schema().names()
        if "timestamp" in actual_cols and "timestamp_ms" not in actual_cols:
            lf = lf.rename({"timestamp": "timestamp_ms"})
            
        return lf if lazy else lf.collect()

class LOBDataset(Dataset):
    """
    Dataset для PyTorch с поддержкой multi-horizon, streaming и memmap режимов.
    """
    def __init__(
        self, 
        df: Union[pl.DataFrame, pl.LazyFrame, str], 
        seq_len: int = 100, 
        n_past_returns: int = 3,
        past_returns_lags: Union[List[int], None] = None,
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
        regime_detector = None,
        regime_window: int = 1000,
        exclude_features: Union[List[str], None] = None,
        scaler_type: str = "zscore",
        winsor_limits: tuple[float, float] = (0.01, 0.99)
    ):
        self.seq_len = seq_len
        self.n_levels = 50
        self.n_past_returns = n_past_returns
        self.past_returns_lags = past_returns_lags if past_returns_lags is not None else [10, 50, 100]
        self.vol_window = vol_window
        self.data_mode = data_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.half_life_hours = half_life_hours
        self.min_weight = min_weight
        self.regime_detector = regime_detector
        self.regime_window = regime_window
        self.exclude_features = exclude_features
        self.scaler_type = scaler_type
        self.winsor_limits = winsor_limits
        
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.use_symmetric_flip = use_symmetric_flip
        self.volume_jitter_range = volume_jitter_range
        self.generator = torch.Generator().manual_seed(aug_seed)
        
        self.price_cols, self.vol_cols, self.ask_cols, self.bid_cols = _compute_augmentation_indices(self.n_levels)
        
        if class_weights is not None:
            self.class_weights = class_weights.cpu().numpy() if isinstance(class_weights, torch.Tensor) else np.array(class_weights, dtype=np.float32)
        else:
            self.class_weights = None

        # 0. Horizon Detection (Unified for all modes) - Задача 094-2
        if isinstance(df, pl.DataFrame):
            all_cols = df.columns
        elif isinstance(df, pl.LazyFrame):
            all_cols = df.collect_schema().names()
        elif isinstance(df, str):
            all_cols = pl.scan_parquet(df).collect_schema().names()
        else:
            all_cols = []

        if any(c.startswith("label_h") for c in all_cols):
            self.label_cols = sorted([c for c in all_cols if c.startswith("label_h")])
            self.is_multi_horizon = True
            self.num_horizons = len(self.label_cols)
        else:
            self.label_cols = ["label"]
            self.is_multi_horizon = False
            self.num_horizons = 1

        # Mode-specific initialization
        if data_mode == "memory":
            self._init_memory_mode(df)
        elif data_mode == "streaming":
            self._init_streaming_mode(df)
        elif data_mode == "memmap":
            self._init_memmap_mode(df)
        else:
            raise ValueError(f"Unknown data_mode: {data_mode}")

    def _calculate_time_weights(self, timestamps: np.ndarray, labels: np.ndarray) -> torch.Tensor:
        max_ts = timestamps.max()
        half_life_ms = self.half_life_hours * 3600 * 1000
        decay_lambda = np.log(2) / half_life_ms
        
        deltas = max_ts - timestamps
        time_weights = np.exp(-decay_lambda * deltas)
        time_weights = np.clip(time_weights, self.min_weight, 1.0)
        
        if self.class_weights is not None:
            # Используем метку для первого горизонта при расчете весов
            labels_idx = labels[:, 0] if labels.ndim > 1 else labels
            sample_class_weights = self.class_weights[labels_idx]
            final_weights = time_weights * sample_class_weights
        else:
            final_weights = time_weights
        
        final_weights = final_weights / (final_weights.mean() + 1e-8)
        return torch.tensor(final_weights, dtype=torch.float32)

    def _setup_feature_indices(self, df_cols: List[str] = None):
        """
        Задача 306.2.2: Централизованная настройка индексов признаков для устойчивости к изменениям структуры.
        Результат: В массиве x_raw первым элементом всегда будет ask_p_0 (индекс 0).
        """
        # Если передан список колонок DF, используем его для поиска
        # Иначе используем self.feat_cols (который соответствует numpy массиву)
        lookup_list = df_cols if df_cols is not None else self.feat_cols
        
        # Строго заданный порядок признаков LOB (Задача 304)
        lob_column_names = [f"feat_ask_p_{i}" for i in range(self.n_levels)] + \
                           [f"feat_ask_v_{i}" for i in range(self.n_levels)] + \
                           [f"feat_bid_p_{i}" for i in range(self.n_levels)] + \
                           [f"feat_bid_v_{i}" for i in range(self.n_levels)]
        
        # Задача 306.2.2: Создаем маску индексов для LOB
        self.lob_indices = []
        for c in lob_column_names:
            try:
                self.lob_indices.append(lookup_list.index(c))
            except ValueError:
                pass # Пропускаем если колонки нет
        
        if not self.lob_indices:
            # Fallback на первые 200 если ничего не нашли
            self.lob_indices = list(range(0, 200))
        
        # Поиск индексов по именам
        def get_idx(name):
            try: return lookup_list.index(name)
            except (ValueError, AttributeError): return -1

        self.trade_vol_idx = get_idx("feat_trade_volume")
        self.trade_side_idx = get_idx("feat_trade_side")
        self.ofi_idx = get_idx("feat_ofi_100")
        self.vib_idx = get_idx("feat_vib_100")
        
        # Индексы для Past Returns
        self.past_ret_indices = []
        for lag in self.past_returns_lags:
            name = f"feat_past_return_{lag}"
            idx = get_idx(name)
            if idx >= 0:
                self.past_ret_indices.append(idx)
        
        # Если past_returns_lags не в feat_cols (они могут добавляться позже через hstack),
        # то они будут находиться после всех feat_cols
        if not self.past_ret_indices and self.n_past_returns > 0:
            start_idx = len(self.feat_cols) if hasattr(self, 'feat_cols') else 200
            self.past_ret_indices = list(range(start_idx, start_idx + self.n_past_returns))

    def _init_memory_mode(self, df: pl.DataFrame):
        if not isinstance(df, pl.DataFrame):
            raise TypeError("For 'memory' mode, df must be a pl.DataFrame")
        
        # Строго заданный порядок признаков LOB (Задача 304)
        feat_cols = [f"feat_ask_p_{i}" for i in range(self.n_levels)] + \
                    [f"feat_ask_v_{i}" for i in range(self.n_levels)] + \
                    [f"feat_bid_p_{i}" for i in range(self.n_levels)] + \
                    [f"feat_bid_v_{i}" for i in range(self.n_levels)]

        # Добавляем остальные признаки в СТРОГОМ отсортированном порядке для детерминизма
        all_feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
        extra_feats = [c for c in all_feat_cols if c not in feat_cols]
        feat_cols.extend(extra_feats)

        if self.exclude_features:
            feat_cols = [c for c in feat_cols if c not in self.exclude_features]
        
        self.feat_cols = feat_cols
        
        # Задача 306.2.2: Инициализируем именованные индексы
        self._setup_feature_indices()

        if self.scaler_type in ("robust", "winsor_robust"):
            norm = Normalizer(output_path="norm_params.json")
            norm.scaler_type = self.scaler_type
            norm.winsor_limits = self.winsor_limits
            df_feat = df.select(feat_cols)
            norm.fit(df_feat, winsor_limits=self.winsor_limits)
            df_norm = norm.transform(df_feat)
            df = df.drop(feat_cols).hstack(df_norm)
            self.robust_params = norm.params
        else:
            self.robust_params = None
        
        # Задача 306.2.1: Гарантируем наличие timestamp_ms для внутренних нужд, но исключаем из признаков
        select_cols = [pl.concat_list(feat_cols).alias("features"), *self.label_cols]
        if "timestamp_ms" in df.columns:
            select_cols.append("timestamp_ms")
        elif "timestamp" in df.columns:
            df = df.rename({"timestamp": "timestamp_ms"})
            select_cols.append("timestamp_ms")
            
        if "mid_price" in df.columns:
            select_cols.append(pl.col("mid_price"))
            
        sequence_df = df.select(select_cols)
        features_series = sequence_df.select(pl.col("features")).to_series()
        self.features = np.stack([np.array(row, dtype=np.float32) for row in features_series.to_list()], axis=0)
        
        if self.is_multi_horizon:
            labels_list = [sequence_df.select(pl.col(lc)).to_series().to_numpy() for lc in self.label_cols]
            self.labels = np.stack(labels_list, axis=1)
        else:
            self.labels = sequence_df.select(pl.col("label")).to_series().to_numpy()
        
        if "timestamp_ms" in sequence_df.columns:
            self.timestamps = sequence_df.select(pl.col("timestamp_ms")).to_series().to_numpy()
        else:
            # Если таймстампов нет (удалены в load_multi_symbol_data), используем индексы
            self.timestamps = np.arange(len(sequence_df), dtype=np.int64)
        
        if "mid_price" in df.columns:
            mid_prices = df["mid_price"].to_numpy()
            self.vols = compute_target_vol(mid_prices, window=self.vol_window)[self.seq_len - 1:]
        else:
            self.vols = np.zeros(len(self.labels) - self.seq_len + 1, dtype=np.float32)

        if self.n_past_returns > 0 and "mid_price" in df.columns:
            past_returns = compute_past_returns(df["mid_price"].to_numpy(), self.past_returns_lags)
            self.features = np.hstack([self.features, past_returns])
            # Обновляем feat_cols, чтобы индексы в _process_sample были корректными
            for lag in self.past_returns_lags:
                self.feat_cols.append(f"feat_past_return_{lag}")

        weight_labels = self.labels[self.seq_len-1:]
        self.sample_weights = self._calculate_time_weights(self.timestamps[self.seq_len-1:], weight_labels)
        
        self.regime_ids = np.zeros(len(self.features) - self.seq_len + 1, dtype=np.int64)
        if self.regime_detector and self.regime_detector.is_fitted:
            regime_features = compute_regime_features(df, window=self.regime_window)
            self.regime_ids = self.regime_detector.predict_states(regime_features)[self.seq_len - 1:]

    def _init_streaming_mode(self, df: Union[pl.DataFrame, pl.LazyFrame, str]):
        if isinstance(df, str):
            df = pl.scan_parquet(df, low_memory=True)
        elif isinstance(df, pl.DataFrame):
            df = df.lazy()
        
        schema = df.collect_schema()
        
        # Строго заданный порядок признаков LOB (Задача 304)
        feat_cols = [f"feat_ask_p_{i}" for i in range(self.n_levels)] + \
                    [f"feat_ask_v_{i}" for i in range(self.n_levels)] + \
                    [f"feat_bid_p_{i}" for i in range(self.n_levels)] + \
                    [f"feat_bid_v_{i}" for i in range(self.n_levels)]

        # Добавляем остальные признаки в СТРОГОМ отсортированном порядке для детерминизма
        all_feat_cols = sorted([c for c in schema.names() if c.startswith("feat_")])
        extra_feats = [c for c in all_feat_cols if c not in feat_cols]
        feat_cols.extend(extra_feats)

        if self.exclude_features:
            feat_cols = [c for c in feat_cols if c not in self.exclude_features]

        # Защита от NaN в пайплайне Polars (Задача 094-2)
        # Задача 306.2.1: Гарантируем наличие timestamp_ms
        select_cols = [*feat_cols, *self.label_cols, "mid_price"]
        if "timestamp_ms" in schema.names():
            select_cols.append("timestamp_ms")
        elif "timestamp" in schema.names():
            df = df.rename({"timestamp": "timestamp_ms"})
            select_cols.append("timestamp_ms")
            
        self.lazy_df = df.select(select_cols).fill_null(0.0)
        
        total_rows = self.lazy_df.select(pl.len()).collect(streaming=True).item()
        self.total_samples = total_rows - self.seq_len + 1
        
        self.file_path = df if isinstance(df, str) else None
        self.row_offsets = self._build_row_offsets(self.file_path, total_rows)
        self.feat_cols = feat_cols
        # Задача 306.2.2: Централизованная настройка индексов
        self._setup_feature_indices()

        
        self._cache_batch = None
        self._batch_size = 50000 
        
        mid_prices = self.lazy_df.select("mid_price").collect(streaming=True).to_series().to_numpy()
        self.vols = compute_target_vol(mid_prices, window=self.vol_window)[self.seq_len - 1:]
        
        self.timestamps = self.lazy_df.select("timestamp_ms").slice(self.seq_len - 1).collect(streaming=True).to_series().to_numpy()
        
        # Загружаем метки для весов (первый горизонт)
        weight_col = self.label_cols[0]
        labels_for_weights = self.lazy_df.select(weight_col).slice(self.seq_len - 1).collect(streaming=True).to_series().to_numpy()
        self.sample_weights = self._calculate_time_weights(self.timestamps, labels_for_weights)
        self.regime_ids = np.zeros(self.total_samples, dtype=np.int64)

    def _build_row_offsets(self, file_path: Union[str, None], total_rows: int) -> np.ndarray:
        if file_path and Path(file_path).exists():
            try:
                pf = pq.ParquetFile(file_path)
                row_counts = [pf.row_group(i).num_rows for i in range(pf.num_row_groups)]
                return np.cumsum(row_counts)
            except: pass
        num_groups = min(100, total_rows)
        group_size = total_rows // num_groups
        return np.arange(group_size, total_rows + group_size, group_size, dtype=np.int64)[:num_groups]

    def _init_memmap_mode(self, df: Union[pl.DataFrame, str]):
        if not self.cache_dir: raise ValueError("cache_dir required")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        meta_path = self.cache_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f: meta = json.load(f)
            # Проверка консистентности (Задача 094-2)
            if meta.get('label_cols') == self.label_cols and meta.get('seq_len') == self.seq_len:
                self.features_seq = np.memmap(self.cache_dir / "features.npy", dtype='float32', mode='r', shape=tuple(meta['features_shape']))
                self.labels = np.memmap(self.cache_dir / "labels.npy", dtype='int64', mode='r', shape=tuple(meta['labels_shape']))
                self.vols = np.memmap(self.cache_dir / "vols.npy", dtype='float32', mode='r', shape=(meta['n_samples'],))
                self.timestamps = np.memmap(self.cache_dir / "timestamps.npy", dtype='int64', mode='r', shape=(meta['n_samples'],))
                self.sample_weights = torch.from_numpy(np.memmap(self.cache_dir / "weights.npy", dtype='float32', mode='r', shape=(meta['n_samples'],)).copy())
                self.regime_ids = np.zeros(meta['n_samples'], dtype=np.int64)
                return

        if isinstance(df, str): df = pl.read_parquet(df)
        elif isinstance(df, pl.LazyFrame): df = df.collect()
        
        # Строго заданный порядок признаков LOB (Задача 304)
        feat_cols = [f"feat_ask_p_{i}" for i in range(self.n_levels)] + \
                    [f"feat_ask_v_{i}" for i in range(self.n_levels)] + \
                    [f"feat_bid_p_{i}" for i in range(self.n_levels)] + \
                    [f"feat_bid_v_{i}" for i in range(self.n_levels)]

        # Добавляем остальные признаки в СТРОГОМ отсортированном порядке для детерминизма
        all_feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
        extra_feats = [c for c in all_feat_cols if c not in feat_cols]
        feat_cols.extend(extra_feats)

        if self.exclude_features: feat_cols = [c for c in feat_cols if c not in self.exclude_features]
        
        self.feat_cols = feat_cols
        # Задача 306.2.2: Централизованная настройка индексов
        self._setup_feature_indices()
        
        total_rows = len(df) - self.seq_len + 1
        n_feats = len(feat_cols)
        
        f_map = np.memmap(self.cache_dir / "features.npy", dtype='float32', mode='w+', shape=(total_rows, self.seq_len, n_feats))
        l_shape = (total_rows, self.num_horizons) if self.is_multi_horizon else (total_rows,)
        l_map = np.memmap(self.cache_dir / "labels.npy", dtype='int64', mode='w+', shape=l_shape)
        v_map = np.memmap(self.cache_dir / "vols.npy", dtype='float32', mode='w+', shape=(total_rows,))
        t_map = np.memmap(self.cache_dir / "timestamps.npy", dtype='int64', mode='w+', shape=(total_rows,))
        
        mid_prices = df["mid_price"].to_numpy()
        vols_arr = compute_target_vol(mid_prices, window=self.vol_window)[self.seq_len - 1:]
        
        has_ts = "timestamp_ms" in df.columns
        if not has_ts and "timestamp" in df.columns:
            df = df.rename({"timestamp": "timestamp_ms"})
            has_ts = True

        # Построчная запись (чанками)
        for i in range(total_rows):
            f_map[i] = df.select(feat_cols).slice(i, self.seq_len).to_numpy()
            if self.is_multi_horizon:
                l_map[i] = [df[lc][i + self.seq_len - 1] for lc in self.label_cols]
            else:
                l_map[i] = df["label"][i + self.seq_len - 1]
            v_map[i] = vols_arr[i]
            if has_ts:
                t_map[i] = df["timestamp_ms"][i + self.seq_len - 1]
            else:
                t_map[i] = i + self.seq_len - 1

        w_arr = self._calculate_time_weights(t_map[:], l_map[:]).numpy()
        w_map = np.memmap(self.cache_dir / "weights.npy", dtype='float32', mode='w+', shape=(total_rows,))
        w_map[:] = w_arr
        
        with open(meta_path, 'w') as f:
            json.dump({'n_samples': total_rows, 'features_shape': [total_rows, self.seq_len, n_feats], 
                      'labels_shape': list(l_shape), 'label_cols': self.label_cols, 'seq_len': self.seq_len}, f)
        
        f_map.flush(); l_map.flush(); v_map.flush(); t_map.flush(); w_map.flush()
        self.features_seq, self.labels, self.vols, self.timestamps = f_map, l_map, v_map, t_map
        self.sample_weights = torch.from_numpy(w_map[:])
        self.regime_ids = np.zeros(total_rows, dtype=np.int64)

    def __len__(self):
        return self.total_samples if self.data_mode == "streaming" else len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        if self.data_mode == "streaming":
            return self._getitem_streaming(idx)
        
        # Memory/Memmap access
        x_raw = self.features[idx : idx + self.seq_len] if self.data_mode == "memory" else self.features_seq[idx]
        y = self.labels[idx + self.seq_len - 1] if self.data_mode == "memory" else self.labels[idx]
        v = self.vols[idx]
        w = self.sample_weights[idx]
        regime_id = torch.tensor(self.regime_ids[idx]).long()
        
        return self._process_sample(x_raw, y, v, w, regime_id)

    def _getitem_streaming(self, idx):
        if self._cache_batch is None or not (self._cache_start_idx <= idx < self._cache_end_idx):
            start = idx
            batch_df = self.lazy_df.slice(start, self._batch_size + self.seq_len - 1).collect(streaming=True)
            
            # Формирование последовательностей
            feat_data = batch_df.select(self.feat_cols).to_numpy()
            self._cache_batch = np.stack([feat_data[i:i+self.seq_len] for i in range(len(batch_df) - self.seq_len + 1)], axis=0)
            
            if self.is_multi_horizon:
                l_lists = [batch_df.select(lc).slice(self.seq_len-1).to_series().to_numpy() for lc in self.label_cols]
                self._cache_labels = np.stack(l_lists, axis=1)
            else:
                self._cache_labels = batch_df.select("label").slice(self.seq_len-1).to_series().to_numpy()
            
            self._cache_vols = self.vols[start : start + len(self._cache_labels)]
            self._cache_start_idx, self._cache_end_idx = start, start + len(self._cache_labels)

        off = idx - self._cache_start_idx
        return self._process_sample(self._cache_batch[off], self._cache_labels[off], self._cache_vols[off], self.sample_weights[idx], torch.tensor(0).long())

    def _process_sample(self, x_raw, y, v, w, regime_id):
        # NaN protection (Задача 094-2)
        x_raw = np.nan_to_num(x_raw, nan=0.0)
        
        # 3-channel LOB (Задача 304: Строго по блокам)
        # Задача 306.2.2: Используем именованный слайс lob_indices
        lob_flat = x_raw[:, self.lob_indices]
        x = torch.from_numpy(lob_flat.copy())
        
        if self.is_train and torch.rand(1, generator=self.generator).item() < self.augment_prob:
            if self.use_symmetric_flip and torch.rand(1, generator=self.generator).item() < 0.5:
                x, y = apply_symmetric_flip(x, y, self.price_cols, self.ask_cols, self.bid_cols)
            if self.volume_jitter_range > 0:
                x = apply_volume_jitter(x, self.volume_jitter_range, self.vol_cols, self.generator)

        # Подготовка 6 каналов (Задача 306)
        ask_p, ask_v = x[:, 0:50], x[:, 50:100]
        bid_p, bid_v = x[:, 100:150], x[:, 150:200]
        
        # ch[0-2]: Базовые LOB каналы
        price_ch = (ask_p + bid_p) / 2.0
        vol_ch = ask_v + bid_v
        # Задача 306.3.1: Безопасный расчет Imbalance с abs() в знаменателе
        denom = torch.abs(bid_v) + torch.abs(ask_v) + 1e-6
        imb_ch = torch.clamp((bid_v - ask_v) / denom, min=-5.0, max=5.0)
        
        # ch[3]: OFI (Order Flow Imbalance) - берем уже нормализованный из FeatureEngineer
        if self.ofi_idx >= 0:
            ofi = torch.from_numpy(x_raw[:, self.ofi_idx].copy()).float()
        else:
            # Fallback для совместимости
            ap0, av0, bp0, bv0 = ask_p[:, 0], ask_v[:, 0], bid_p[:, 0], bid_v[:, 0]
            bp_prev = torch.cat([bp0[:1], bp0[:-1]])
            bv_prev = torch.cat([bv0[:1], bv0[:-1]])
            ap_prev = torch.cat([ap0[:1], ap0[:-1]])
            av_prev = torch.cat([av0[:1], av0[:-1]])
            delta_bid = torch.where(bp0 > bp_prev, bv0, torch.where(bp0 < bp_prev, -bv_prev, bv0 - bv_prev))
            delta_ask = torch.where(ap0 < ap_prev, av0, torch.where(ap0 > ap_prev, -av_prev, av0 - av_prev))
            ofi = (delta_bid - delta_ask).cumsum(dim=0)
        ofi_ch = ofi.unsqueeze(-1).repeat(1, 50)
        
        # ch[4]: Trade Imbalance (VIB) - берем уже нормализованный из FeatureEngineer
        if self.vib_idx >= 0:
            vib = torch.from_numpy(x_raw[:, self.vib_idx].copy()).float()
        else:
            # Fallback
            if self.trade_vol_idx >= 0 and self.trade_side_idx >= 0:
                tr_v = torch.from_numpy(x_raw[:, self.trade_vol_idx].copy()).float()
                tr_s = torch.from_numpy(x_raw[:, self.trade_side_idx].copy()).float()
                vib = (tr_v * tr_s).cumsum(dim=0)
            else:
                vib = torch.zeros(x.shape[0], device=x.device)
        vib_ch = vib.unsqueeze(-1).repeat(1, 50)
        
        # ch[5]: Past Returns (100 тиков)
        # Задача 306.2.2: Используем заранее найденные индексы
        if self.past_ret_indices:
            # Пытаемся взять лаг 100 (обычно последний в списке, если lags=[10, 50, 100])
            # Для простоты берем первый доступный из найденных, если 100 нет
            pr_idx = self.past_ret_indices[-1] # Предполагаем, что самый длинный лаг последний
            # Задача 306.3.2: Убрано умножение на 100.0, так как RobustScaler уже нормализовал данные
            pr = torch.from_numpy(x_raw[:, pr_idx].copy()).float()
        else:
            pr = torch.zeros(x.shape[0], device=x.device)
        pr_ch = pr.unsqueeze(-1).repeat(1, 50)
        
        # Собираем итоговый тензор (Seq, 6, 50)
        x_final = torch.stack([price_ch, vol_ch, imb_ch, ofi_ch, vib_ch, pr_ch], dim=1)
        
        # Задача 306.3.3: Глобальный предохранитель - ограничиваем весь тензор
        x_final = torch.clamp(x_final, min=-12.0, max=12.0)
        
        return x_final, torch.tensor(y).long(), torch.tensor(v).float(), w, regime_id

    def get_class_distribution(self) -> np.ndarray:
        if self.data_mode == "streaming":
            weight_col = self.label_cols[0]
            counts = self.lazy_df.select(weight_col).slice(self.seq_len-1).group_by(weight_col).agg(pl.len()).collect()
            dist = np.zeros(3, dtype=np.int64)
            for r in counts.iter_rows(): 
                if 0 <= r[0] < 3: dist[int(r[0])] = r[1]
            return dist
        labels = self.labels[:, 0] if self.is_multi_horizon else self.labels
        classes, counts = np.unique(labels, return_counts=True)
        dist = np.zeros(3, dtype=np.int64)
        for c, count in zip(classes, counts): 
            if 0 <= c < 3: dist[int(c)] = count
        return dist
    
    def get_timestamps(self) -> np.ndarray:
        return self.timestamps

LOBPyTorchDataset = LOBDataset

def get_val_loader(data_path, symbol, seq_len=100, n_past_returns=0, batch_size=256, shuffle=False, num_workers=4, data_mode="memory", val_split=0.8):
    from torch.utils.data import DataLoader
    df = pl.read_parquet(Path(data_path) / f"{symbol}_*.parquet")
    val_df = df.slice(int(len(df) * val_split))
    val_ds = LOBDataset(val_df, seq_len=seq_len, n_past_returns=n_past_returns, data_mode=data_mode, is_train=False)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())

class LOBCalibrationDataReader(CalibrationDataReader):
    def __init__(self, onnx_model_path, data_path, symbol, seq_len=100, n_past_returns=0, n_samples=1000, val_split=0.8):
        self.onnx_model_path, self.input_name = onnx_model_path, onnx.load(onnx_model_path).graph.input[0].name
        df = pl.read_parquet(Path(data_path) / f"{symbol}_*.parquet")
        val_ds = LOBDataset(df.slice(int(len(df) * val_split)), seq_len=seq_len, n_past_returns=n_past_returns, data_mode="memory", is_train=False)
        self.calibration_data = [val_ds[i][0].numpy() for i in range(min(n_samples, len(val_ds)))]
        self.enum_data = None
    
    def get_next(self):
        if self.enum_data is None: self.enum_data = iter([{self.input_name: d} for d in self.calibration_data])
        return next(self.enum_data, None)
    
    def rewind(self): self.enum_data = None

def load_multi_symbol_data(symbols, data_path="bots", lazy=True) -> Union[pl.DataFrame, pl.LazyFrame]:
    scans = []
    for symbol in sorted(symbols):
        path = Path(data_path) / symbol / "data" / "raw"
        lf = pl.scan_parquet(path / f"{symbol}_*.parquet").with_columns(pl.lit(symbol).alias("symbol"))
        trades_files = list(path.glob("trades_*.parquet"))
        if trades_files:
            tr_lf = pl.scan_parquet(trades_files)
            # Приводим к единому имени timestamp_ms
            if "timestamp" in tr_lf.collect_schema().names():
                tr_lf = tr_lf.rename({"timestamp": "timestamp_ms"})
            
            tr_lf = tr_lf.select([
                "timestamp_ms", 
                pl.col("price").alias("feat_trade_price"), 
                pl.col("size").alias("feat_trade_volume"),
                pl.when(pl.col("side") == "Buy").then(1.0)
                .when(pl.col("side") == "Sell").then(-1.0)
                .otherwise(0.0).alias("feat_trade_side")
            ])
            lf = lf.sort("timestamp_ms").join_asof(tr_lf.sort("timestamp_ms"), on="timestamp_ms", strategy="backward")
        else:
            lf = lf.with_columns([
                pl.lit(None).cast(pl.Float64).alias("feat_trade_price"), 
                pl.lit(None).cast(pl.Float64).alias("feat_trade_volume"),
                pl.lit(0.0).alias("feat_trade_side")
            ])
        scans.append(lf)
    merged = pl.concat(scans).sort(["timestamp_ms", "symbol"])
    
    # Задача 306.2.1: Удаляем служебные колонки ПЕРЕД конвертацией в numpy/memmap
    # Используем правильные имена согласно уточнению пользователя
    meta_cols = ["timestamp_ms", "last_update_id", "symbol"]
    merged = merged.drop([c for c in meta_cols if c in merged.columns])
    
    return merged if lazy else merged.collect()

def load_symbol_config(symbol, config_path="bots") -> Dict[str, Any]:
    try: import tomllib
    except: import tomli as tomllib
    with open(Path(config_path) / symbol / "config.toml", "rb") as f: return tomllib.load(f)

def load_multi_symbol_configs(symbols, config_path="bots") -> Dict[str, Dict[str, Any]]:
    return {s: load_symbol_config(s, config_path) for s in symbols}

if __name__ == "__main__":
    try:
        loader = LOBDataLoader("bots/BTCUSDT/data/raw", "BTCUSDT")
        print(loader.load_data().select(["timestamp_ms", "ask_p_0", "bid_p_0"]).head())
    except: pass
