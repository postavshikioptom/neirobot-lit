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


def compute_depth_imbalance_globally(bid_v: np.ndarray, ask_v: np.ndarray) -> np.ndarray:
    """
    Векторизованный расчет Volume Imbalance (VIB) по всей глубине стакана.
    
    Формула: (sum(bid_v) - sum(ask_v)) / (sum(bid_v) + sum(ask_v) + epsilon)
    
    Args:
        bid_v: матрица объемов bid (N, 50), где N - количество сэмплов
        ask_v: матрица объемов ask (N, 50)
    
    Returns:
        np.ndarray: массив VIB значений (N,)
    """
    # Суммируем по всем 50 уровням для каждого сэмпла
    sum_bid = np.sum(bid_v, axis=1)  # (N,)
    sum_ask = np.sum(ask_v, axis=1)  # (N,)
    
    # Вычисляем дисбаланс с защитой от деления на ноль
    vib = (sum_bid - sum_ask) / (sum_bid + sum_ask + 1e-8)
    
    return vib.astype(np.float32)


def compute_ofi_from_lob(bid_p: np.ndarray, ask_p: np.ndarray,
                         bid_v: np.ndarray, ask_v: np.ndarray,
                         update_ids: np.ndarray, depth: int = 3) -> np.ndarray:
    """
    Векторизованный расчёт OFI (Order Flow Imbalance) по Cont-Kukanov-Stoikov.

    OFI = per-tick (non-cumulative) order flow imbalance.
    Возвращает дельты (сырой OFI за каждый таймстеп), НЕ кумулятивную сумму.

    Правильная CKS логика (учитывает изменения цен):
    - buyer-initiated: bid_price увеличился ИЛИ (bid_price не изменился И bid_volume увеличился)
    - seller-initiated: ask_price уменьшился ИЛИ (ask_price не изменился И ask_volume уменьшился)

    Формула:
    OFI_delta[t] = sum_{i=0}^{depth-1} [ bid_vol_diff[t,i] * I{buyer-initiated at level i}
                                          - ask_vol_diff[t,i] * I{seller-initiated at level i} ]
    где diff = текущее значение - предыдущее значение

    Args:
        bid_p, ask_p: (N, 50) best bid/ask prices (relative)
        bid_v, ask_v: (N, 50) raw volumes per level (NOT log1p)
        update_ids: (N,) last_update_id values
        depth: число уровней для расчета (обычно 3)
    Returns:
        (N,) OFI values, per-tick deltas (non-cumulative), dtype=np.float32
    """
    n = len(update_ids)

    # Определяем точки обновления стакана
    id_diff = np.diff(update_ids, prepend=update_ids[0])
    is_update = id_diff > 0  # bool mask

    # Рассчитываем diff цен и объемов только для указанной глубины
    # Используем prepend=0 чтобы получить diff[t] = value[t] - value[t-1], с diff[0]=0
    bid_p_diff = np.diff(bid_p[:, :depth], axis=0, prepend=bid_p[:, :depth][:1])
    ask_p_diff = np.diff(ask_p[:, :depth], axis=0, prepend=ask_p[:, :depth][:1])
    bid_v_diff = np.diff(bid_v[:, :depth], axis=0, prepend=bid_v[:, :depth][:1])
    ask_v_diff = np.diff(ask_v[:, :depth], axis=0, prepend=ask_v[:, :depth][:1])

    # Создаем masks согласно CKS логике
    buyer_mask = (bid_p_diff > 0) | ((bid_p_diff == 0) & (bid_v_diff > 0))
    seller_mask = (ask_p_diff < 0) | ((ask_p_diff == 0) & (ask_v_diff > 0))

    # Суммируем contribution по уровням для каждого таймстемпа
    buy_contrib = np.where(buyer_mask, bid_v_diff, 0).sum(axis=1)
    sell_contrib = np.where(seller_mask, ask_v_diff, 0).sum(axis=1)

    # OFI = buy_contrib - sell_contrib
    ofi_deltas = buy_contrib - sell_contrib

    # Применяем маску обновлений (только при реальном обновлении стакана)
    ofi = np.where(is_update, ofi_deltas, 0.0).astype(np.float32)

    return ofi


def compute_ofi_from_lob_cache(bid_p: np.ndarray, ask_p: np.ndarray,
                                bid_v: np.ndarray, ask_v: np.ndarray,
                                update_ids: np.ndarray) -> np.ndarray:
    """
    Глобальная версия compute_ofi_from_lob для precompute при инициализации.
    Принимает LOG1P объёмы и преобразует их: exp(x) - 1
    Принимает relative prices напрямую (feat_ask_p_i, feat_bid_p_i)
    """
    # Восстанавливаем сырые объёмы из log1p
    bid_v_raw = np.exp(np.clip(bid_v, None, 20.0)) - 1.0
    ask_v_raw = np.exp(np.clip(ask_v, None, 20.0)) - 1.0

    return compute_ofi_from_lob(bid_p, ask_p, bid_v_raw, ask_v_raw, update_ids)


def compute_past_returns_globally(mid_prices: np.ndarray, lags: List[int] = [10, 50, 100]) -> np.ndarray:
    """
    Глобальный расчет Past Returns для всех сэмплов сразу.
    
    Использует безопасное логарифмирование и векторизованные операции.
    
    Args:
        mid_prices: массив средних цен (N,)
        lags: список лагов для расчета [10, 50, 100]
    
    Returns:
        np.ndarray: матрица доходностей (N, len(lags))
    """
    n = len(mid_prices)
    n_lags = len(lags)
    
    # Безопасное логарифмирование с защитой от нулей и отрицательных значений
    log_p = np.log(np.maximum(mid_prices, 1e-10))
    
    # Инициализируем результат
    returns = np.zeros((n, n_lags), dtype=np.float32)
    
    # Векторизованный расчет для каждого лага
    for lag_idx, lag in enumerate(lags):
        if lag < n:
            # returns[lag:] = log_p[lag:] - log_p[:-lag]
            returns[lag:, lag_idx] = log_p[lag:] - log_p[:-lag]
    
    return returns


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
    ОТКЛЮЧЕНО: Динамический OFI больше не используется.
    
    Вместо этого используется compute_static_imbalance() для расчета
    статического per-level imbalance согласно задаче 053.
    
    Старая реализация вычисляла кумулятивный дисбаланс потока ордеров,
    который мог расти экспоненциально и достигать экстремальных значений.
    """
    raise NotImplementedError("Dynamic OFI is deprecated. Use compute_static_imbalance() instead.")


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
    
    # Задача 053: Вычисляем static imbalance для лучшего уровня (depth 0)
    # Формула: (V_bid_0 - V_ask_0) / (V_bid_0 + V_ask_0 + epsilon)
    # Результат: скаляр для каждого снапшота в диапазоне [-1, 1]
    denom = bid_volumes + ask_volumes + 1e-7
    ofi = (bid_volumes - ask_volumes) / denom
    ofi = np.clip(ofi, -1.0, 1.0).astype(np.float32)
    
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
        # Пытаемся найти feat_ колонки и label
        feat_cols = [c for c in all_cols if c.startswith("feat_") or c == "label" or c.startswith("label_h")]
        
        # Пересечение с реальными колонками в файле
        columns = [c for c in (target_cols + lob_cols + trade_cols + feat_cols) if c in all_cols]

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
        normalizer: Union[Normalizer, None] = None,
        regime_window: int = 1000,
        exclude_features: Union[List[str], None] = None,
        scaler_type: str = "robust",  # Задача 313: Используем только RobustScaler (синхронизация с train.py)
        winsor_limits: tuple[float, float] = (0.01, 0.99),
        scale_multiplier: float = 1.0
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
        self.normalizer = normalizer
        self.regime_window = regime_window
        self.exclude_features = exclude_features
        self.scaler_type = scaler_type
        self.winsor_limits = winsor_limits
        self.scale_multiplier = scale_multiplier
        
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.use_symmetric_flip = use_symmetric_flip
        self.volume_jitter_range = volume_jitter_range
        self.generator = torch.Generator().manual_seed(aug_seed)

        # Задача 320.5: Диагностика saturation каналов
        self.channel_names = ["MicropriceDev", "Vol", "Imb", "OFI", "VIB", "Ret_10", "Ret_50", "Ret_100", "Spread", "DeltaImb", "DeltaSpread"]
        self._clip_diag_prints = 0
        self.max_clip_diag_prints = 2  # Ограничиваем логирование первыми 2 сэмплами

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
        # ПРИМЕЧАНИЕ: Режимы "streaming" и "memmap" отключены для упрощения кода.
        # Теперь используется только режим "memory" для загрузки данных в оперативную память.
        if data_mode == "memory":
            self._init_memory_mode(df)
        # elif data_mode == "streaming":
        #     # ОТКЛЮЧЕНО: Streaming режим больше не поддерживается
        #     self._init_streaming_mode(df)
        # elif data_mode == "memmap":
        #     # ОТКЛЮЧЕНО: Memmap режим больше не поддерживается
        #     self._init_memmap_mode(df)
        else:
            raise ValueError(f"Unknown data_mode: {data_mode}. Only 'memory' mode is supported.")

        # Задача 307.6: Аудит целостности признаков и логирование первого сэмпла
        channel_names = ["MicropriceDev", "Vol", "Imb", "OFI", "VIB", "Ret_10", "Ret_50", "Ret_100", "Spread", "DeltaImb", "DeltaSpread"]
        n_channels = len(channel_names)
        expected_cols = [f"feat_{i}" for i in range(n_channels * self.n_levels)]
        actual_cols = self.feat_cols if hasattr(self, 'feat_cols') else []
        missing = [c for c in expected_cols if c not in actual_cols]

        if not missing:
            print(f"[{self.__class__.__name__}] Feature Map Verified: {n_channels} channels, {self.n_levels} levels. Total features: {n_channels * self.n_levels}. Status: OK")

        if len(self) > 0:
            sample_x = self[0][0]
            print(f"[{self.__class__.__name__}] First Sample Statistics (z-score normalized, clipped [-5,5]):")
            for i in range(sample_x.shape[1]):
                chan = sample_x[:, i, :]
                print(f"  Channel {i} ({channel_names[i]}): min={chan.min():.4f}, max={chan.max():.4f}, mean={chan.mean():.4f}")
        
        # Информационный лог о загрузке
        n_samples = len(self)
        print(f"[{self.__class__.__name__}] Loaded {n_samples} samples. Data mode: {data_mode}")

        # Логирование распределения классов (диагностика дисбаланса)
        if hasattr(self, 'labels') and len(self.labels) > 0:
            labels_for_stats = self.labels[:, 0] if self.labels.ndim > 1 else self.labels
            unique, counts = np.unique(labels_for_stats, return_counts=True)
            total = len(labels_for_stats)
            print(f"[{self.__class__.__name__}] Label distribution:")
            for cls, cnt in zip(unique, counts):
                print(f"  Class {cls}: {cnt} samples ({cnt/total:.1%})")
            # Предупреждение о доминации Flat
            flat_count = counts[unique == 0].sum() if 0 in unique else 0
            if flat_count / total > 0.9:
                print(f"[WARN] Flat class dominates: {flat_count/total:.1%}. Consider adjusting threshold or oversampling.")

    def get_timestamps(self) -> np.ndarray:
        """Возвращает массив timestamps для всех сэмплов датасета."""
        return self.timestamps

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
        
        # Задача 306.4.2: Инициализация масок индексов для каждого типа колонок LOB
        # Это позволяет обращаться к колонкам по именам, а не по жестким слайсам
        self.ask_p_indices = []
        self.ask_v_indices = []
        self.bid_p_indices = []
        self.bid_v_indices = []
        
        for i in range(self.n_levels):
            try:
                self.ask_p_indices.append(lookup_list.index(f"feat_ask_p_{i}"))
            except ValueError:
                pass
            try:
                self.ask_v_indices.append(lookup_list.index(f"feat_ask_v_{i}"))
            except ValueError:
                pass
            try:
                self.bid_p_indices.append(lookup_list.index(f"feat_bid_p_{i}"))
            except ValueError:
                pass
            try:
                self.bid_v_indices.append(lookup_list.index(f"feat_bid_v_{i}"))
            except ValueError:
                pass
        
        # Поиск индексов по именам
        def get_idx(name):
            try: return lookup_list.index(name)
            except (ValueError, AttributeError): return -1

        self.trade_vol_idx = get_idx("feat_trade_volume")
        self.trade_side_idx = get_idx("feat_trade_side")
        self.ofi_idx = get_idx("feat_ofi_100")
        self.vib_idx = get_idx("feat_vib_100")
        # Задача 318.1: Индекс для last_update_id (расчёт OFI)
        self.update_id_idx = get_idx("feat_update_id")
        
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
        
        # 1. РАСЧЕТ ИЗ СЫРЫХ ДАННЫХ (ДО НОРМАЛИЗАЦИИ)
        # Задача 308-4: Расчет индикаторов до нормализации основных данных
        print(f"[{self.__class__.__name__}] Computing features from RAW data...")
        
        bid_v_cols = [f"feat_bid_v_{i}" for i in range(self.n_levels)]
        ask_v_cols = [f"feat_ask_v_{i}" for i in range(self.n_levels)]
        price_col = "mid_price" if "mid_price" in df.columns else df.columns[0]
        
        # 1. Извлекаем логарифмированные объемы и восстанавливаем сырые данные
        # Задача 315-3 fix: сначала восстанавливаем каждый уровень, потом суммируем
        # БЫЛО (неверно): exp(sum(log(1+v))) = product(1+v) - 1
        # СТАЛО (верно):  sum(exp(v_i) - 1) = сумма сырых объемов по уровням
        bid_v_matrix = df.select(bid_v_cols).to_numpy().astype(np.float64)  # (N, 50)
        ask_v_matrix = df.select(ask_v_cols).to_numpy().astype(np.float64)  # (N, 50)
        # Clamp to prevent float64 overflow (consistent with _calculate_6_channels_raw)
        raw_bid_sum = (np.exp(np.clip(bid_v_matrix, None, 20.0)) - 1.0).sum(axis=1)  # (N,)
        raw_ask_sum = (np.exp(np.clip(ask_v_matrix, None, 20.0)) - 1.0).sum(axis=1)  # (N,)
        raw_prices = df[price_col].to_numpy().astype(np.float64)

        # 2. Расчет VIB из СЫРЫХ объемов (Задача 315)
        denom = raw_bid_sum + raw_ask_sum
        # Используем маску, чтобы не делить на ноль, и считаем разницу
        v_diff = raw_bid_sum - raw_ask_sum
        # Use same epsilon and logic as _calculate_6_channels_raw
        self.vib_cache = (v_diff / (denom + 1e-8)).clip(-1.0, 1.0).astype(np.float32)

        # 3. Расчет PastRet (3 лага: 10, 50, 100)
        log_p = np.log(np.maximum(raw_prices, 1e-9))
        self.past_ret_cache = {}  # словарь: lag -> массив
        for lag in [10, 50, 100]:
            ret = np.zeros(len(log_p), dtype=np.float32)
            if len(log_p) > lag:
                ret[lag:] = (log_p[lag:] - log_p[:-lag]).astype(np.float32)
            self.past_ret_cache[lag] = ret

        # 4. Расчёт OFI из сырых данных (Задача 318.2)
        if hasattr(self, 'update_id_raw') and self.update_id_raw is not None:
            bid_p_matrix = df.select([f"feat_bid_p_{i}" for i in range(self.n_levels)]).to_numpy().astype(np.float64)
            ask_p_matrix = df.select([f"feat_ask_p_{i}" for i in range(self.n_levels)]).to_numpy().astype(np.float64)
            self.ofi_cache = compute_ofi_from_lob_cache(
                bid_p_matrix, ask_p_matrix, bid_v_matrix, ask_v_matrix,
                self.update_id_raw
            )
            print(f"[DEBUG] ofi raw (per-tick, non-cumulative): min={self.ofi_cache.min():.6f}, max={self.ofi_cache.max():.6f}, mean={self.ofi_cache.mean():.6f}")
        else:
            self.ofi_cache = None
            print("[DEBUG] update_id_raw not available, OFI will use fallback")
        
        # Шаг 1: Диагностика сырых признаков (индикаторов)
        print(f"[DEBUG] past_ret raw (lag=10): min={self.past_ret_cache[10].min():.6f}, max={self.past_ret_cache[10].max():.6f}")
        print(f"[DEBUG] vib raw: min={self.vib_cache.min():.6f}, max={self.vib_cache.max():.6f}")

        # Задача 318.1: Извлекаем feat_update_id ДО удаления мета-колонок
        if "feat_update_id" in df.columns:
            self.update_id_raw = df["feat_update_id"].to_numpy().astype(np.int64)
            print(f"[DEBUG] update_id_raw: shape={self.update_id_raw.shape}, dtype={self.update_id_raw.dtype}")
        else:
            self.update_id_raw = None
            print("[DEBUG] feat_update_id not found, OFI will use fallback (diff(imbalance))")
        
        self.has_computed_features = True

        # 3. ПОДГОТОВКА И НОРМАЛИЗАЦИЯ ОСНОВНЫХ ДАННЫХ
        # Строго заданный порядок признаков LOB (Задача 304)
        feat_cols = [f"feat_ask_p_{i}" for i in range(self.n_levels)] + \
                    [f"feat_ask_v_{i}" for i in range(self.n_levels)] + \
                    [f"feat_bid_p_{i}" for i in range(self.n_levels)] + \
                    [f"feat_bid_v_{i}" for i in range(self.n_levels)]
        
        # Добавляем остальные признаки, которые начинаются на feat_, но не входят в LOB
        all_feat_cols = [c for c in df.columns if c.startswith("feat_")]
        extra_feats = [c for c in all_feat_cols if c not in feat_cols]
        feat_cols.extend(extra_feats)

        if self.exclude_features:
            feat_cols = [c for c in feat_cols if c not in self.exclude_features]
        
        self.feat_cols = feat_cols
        
        # Задача 306.2.2: Инициализируем именованные индексы
        self._setup_feature_indices()

        # Задача 311: Сохраняем СЫРЫЕ данные
        df_feat = df.select(feat_cols)
        self.x_raw = df_feat.to_numpy().astype(np.float32)

        # Задача 311: Обучаем нормализатор на каналах (а не на сырых признаках)
        if self.normalizer is not None and self.is_train:
            print(f"[{self.__class__.__name__}] Training normalizer on CHANNELS...")
            channels_data = self._compute_channels_for_normalization(df_feat)
            self.normalizer.fit(channels_data, feature_names=channels_data.columns)

        self.robust_params = None
        
        # Задача 306.2.1: Гарантируем наличие timestamp_ms для внутренних нужд, но исключаем из признаков
        # (Мы больше не создаем self.features как np.stack, так как используем self.x_raw)
        
        # Нам все еще нужны labels и timestamps
        select_cols = self.label_cols[:]
        
        # Добавляем future_return для расширенной аналитики (Задача 313.4)
        future_ret_cols = [c for c in df.columns if c.startswith("future_return_h")]
        if not future_ret_cols and "future_return" in df.columns:
            future_ret_cols = ["future_return"]
        
        for fr_col in future_ret_cols:
            if fr_col not in select_cols:
                select_cols.append(fr_col)
        self.future_ret_cols = future_ret_cols
            
        if "timestamp_ms" in df.columns:
            select_cols.append("timestamp_ms")
        elif "timestamp" in df.columns:
            select_cols.append("timestamp") # Will rename later
            
        if "mid_price" in df.columns:
            select_cols.append("mid_price")
            
        aux_df = df.select(select_cols)
        if "timestamp" in aux_df.columns and "timestamp_ms" not in aux_df.columns:
            aux_df = aux_df.rename({"timestamp": "timestamp_ms"})

        if self.is_multi_horizon:
            labels_list = [aux_df.select(pl.col(lc)).to_series().to_numpy() for lc in self.label_cols]
            self.labels = np.stack(labels_list, axis=1)
            
            # Сохраняем future_returns для аналитики
            if self.future_ret_cols:
                f_ret_list = [aux_df.select(pl.col(fc)).to_series().to_numpy() for fc in self.future_ret_cols]
                self.future_returns = np.stack(f_ret_list, axis=1)
            else:
                self.future_returns = np.zeros_like(self.labels, dtype=np.float32)
        else:
            self.labels = aux_df.select(pl.col("label")).to_series().to_numpy()
            
            # Сохраняем future_returns для аналитики
            if self.future_ret_cols:
                self.future_returns = aux_df.select(pl.col(self.future_ret_cols[0])).to_series().to_numpy()
            else:
                self.future_returns = np.zeros_like(self.labels, dtype=np.float32)
        
        if "timestamp_ms" in aux_df.columns:
            self.timestamps = aux_df.select(pl.col("timestamp_ms")).to_series().to_numpy()
        else:
            self.timestamps = np.arange(len(aux_df), dtype=np.int64)
        
        # Расчет волатильности для взвешивания или адаптивного порога
        if "mid_price" in aux_df.columns:
            self.mid_prices = aux_df["mid_price"].to_numpy()
            # Задача 319: защита от NaN/Inf в vols
            vols = compute_target_vol(self.mid_prices, window=self.vol_window)[self.seq_len - 1:]
            vols = np.asarray(vols, dtype=np.float32)
            vols = np.nan_to_num(vols, nan=0.0, posinf=0.0, neginf=0.0)
            self.vols = vols
        else:
            self.mid_prices = np.zeros(len(self.labels), dtype=np.float32)
            self.vols = np.zeros(len(self.labels) - self.seq_len + 1, dtype=np.float32)

        weight_labels = self.labels[self.seq_len-1:]
        self.sample_weights = self._calculate_time_weights(self.timestamps[self.seq_len-1:], weight_labels)
        
        self.regime_ids = np.zeros(len(self.x_raw) - self.seq_len + 1, dtype=np.int64)
        if self.regime_detector and self.regime_detector.is_fitted:
            regime_features = compute_regime_features(df, window=self.regime_window)
            self.regime_ids = self.regime_detector.predict_states(regime_features)[self.seq_len - 1:]

    # ОТКЛЮЧЕНО: Streaming режим больше не поддерживается для упрощения кода
    # def _init_streaming_mode(self, df: Union[pl.DataFrame, pl.LazyFrame, str]):
    #     if isinstance(df, str):
    #         df = pl.scan_parquet(df, low_memory=True)
    #     elif isinstance(df, pl.DataFrame):
    #         df = df.lazy()
    #     
    #     schema = df.collect_schema()
    #     
    #     # Строго заданный порядок признаков LOB (Задача 304)
    #     feat_cols = [f"feat_ask_p_{i}" for i in range(self.n_levels)] + \
    #                 [f"feat_ask_v_{i}" for i in range(self.n_levels)] + \
    #                 [f"feat_bid_p_{i}" for i in range(self.n_levels)] + \
    #                 [f"feat_bid_v_{i}" for i in range(self.n_levels)]
    #     
    #     # Добавляем остальные признаки
    #     all_feat_cols = [c for c in schema.names() if c.startswith("feat_")]
    #     extra_feats = [c for c in all_feat_cols if c not in feat_cols]
    #     feat_cols.extend(extra_feats)
    #
    #     if self.exclude_features:
    #         feat_cols = [c for c in feat_cols if c not in self.exclude_features]
    #     
    #     # Защита от NaN в пайплайне Polars (Задача 094-2)
    #     # Задача 306.2.1: Гарантируем наличие timestamp_ms
    #     select_cols = [*feat_cols, *self.label_cols, "mid_price"]
    #     if "timestamp_ms" in schema.names():
    #         select_cols.append("timestamp_ms")
    #     elif "timestamp" in schema.names():
    #         df = df.rename({"timestamp": "timestamp_ms"})
    #         select_cols.append("timestamp_ms")
    #         
    #     self.lazy_df = df.select(select_cols).fill_null(0.0)
    #     
    #     total_rows = self.lazy_df.select(pl.len()).collect(streaming=True).item()
    #     self.total_samples = total_rows - self.seq_len + 1
    #     
    #     self.file_path = df if isinstance(df, str) else None
    #     self.row_offsets = self._build_row_offsets(self.file_path, total_rows)
    #     self.feat_cols = feat_cols
    #     # Задача 306.2.2: Централизованная настройка индексов
    #     self._setup_feature_indices()
    #
    #     
    #     self._cache_batch = None
    #     self._batch_size = 50000 
    #     
    #     mid_prices = self.lazy_df.select("mid_price").collect(streaming=True).to_series().to_numpy()
    #     self.vols = compute_target_vol(mid_prices, window=self.vol_window)[self.seq_len - 1:]
    #     
    #     self.timestamps = self.lazy_df.select("timestamp_ms").slice(self.seq_len - 1).collect(streaming=True).to_series().to_numpy()
    #     
    #     # Загружаем метки для весов (первый горизонт)
    #     weight_col = self.label_cols[0]
    #     labels_for_weights = self.lazy_df.select(weight_col).slice(self.seq_len - 1).collect(streaming=True).to_series().to_numpy()
    #     self.sample_weights = self._calculate_time_weights(self.timestamps, labels_for_weights)
    #     self.regime_ids = np.zeros(self.total_samples, dtype=np.int64)
    #     
    #     # Задача 308: Инициализация кэшей для streaming режима (пока пустые, будут заполняться в _getitem_streaming)
    #     self.vib_cache = None
    #     self.past_ret_cache = None

    # ОТКЛЮЧЕНО: Используется только для streaming режима
    # def _build_row_offsets(self, file_path: Union[str, None], total_rows: int) -> np.ndarray:
    #     if file_path and Path(file_path).exists():
    #         try:
    #             pf = pq.ParquetFile(file_path)
    #             row_counts = [pf.row_group(i).num_rows for i in range(pf.num_row_groups)]
    #             return np.cumsum(row_counts)
    #         except: pass
    #     num_groups = min(100, total_rows)
    #     group_size = total_rows // num_groups
    #     return np.arange(group_size, total_rows + group_size, group_size, dtype=np.int64)[:num_groups]

    # ОТКЛЮЧЕНО: Memmap режим больше не поддерживается для упрощения кода
    # def _init_memmap_mode(self, df: Union[pl.DataFrame, str]):
    #     if not self.cache_dir: raise ValueError("cache_dir required")
    #     self.cache_dir.mkdir(parents=True, exist_ok=True)
    #     
    #     meta_path = self.cache_dir / "metadata.json"
    #     if meta_path.exists():
    #         with open(meta_path, 'r') as f: meta = json.load(f)
    #         # Проверка консистентности (Задача 094-2)
    #         if meta.get('label_cols') == self.label_cols and meta.get('seq_len') == self.seq_len:
    #             self.features_seq = np.memmap(self.cache_dir / "features.npy", dtype='float32', mode='r', shape=tuple(meta['features_shape']))
    #             self.labels = np.memmap(self.cache_dir / "labels.npy", dtype='int64', mode='r', shape=tuple(meta['labels_shape']))
    #             self.vols = np.memmap(self.cache_dir / "vols.npy", dtype='float32', mode='r', shape=(meta['n_samples'],))
    #             self.timestamps = np.memmap(self.cache_dir / "timestamps.npy", dtype='int64', mode='r', shape=(meta['n_samples'],))
    #             self.sample_weights = torch.from_numpy(np.memmap(self.cache_dir / "weights.npy", dtype='float32', mode='r', shape=(meta['n_samples'],)).copy())
    #             self.regime_ids = np.zeros(meta['n_samples'], dtype=np.int64)
    #             return
    #
    #     if isinstance(df, str): df = pl.read_parquet(df)
    #     elif isinstance(df, pl.LazyFrame): df = df.collect()
    #     
    #     # Строго заданный порядок признаков LOB (Задача 304)
    #     feat_cols = [f"feat_ask_p_{i}" for i in range(self.n_levels)] + \
    #                 [f"feat_ask_v_{i}" for i in range(self.n_levels)] + \
    #                 [f"feat_bid_p_{i}" for i in range(self.n_levels)] + \
    #                 [f"feat_bid_v_{i}" for i in range(self.n_levels)]
    #     
    #     # Добавляем остальные признаки
    #     all_feat_cols = [c for c in df.columns if c.startswith("feat_")]
    #     extra_feats = [c for c in all_feat_cols if c not in feat_cols]
    #     feat_cols.extend(extra_feats)
    #
    #     if self.exclude_features: feat_cols = [c for c in feat_cols if c not in self.exclude_features]
    #     
    #     self.feat_cols = feat_cols
    #     # Задача 306.2.2: Централизованная настройка индексов
    #     self._setup_feature_indices()
    #     
    #     total_rows = len(df) - self.seq_len + 1
    #     n_feats = len(feat_cols)
    #     
    #     f_map = np.memmap(self.cache_dir / "features.npy", dtype='float32', mode='w+', shape=(total_rows, self.seq_len, n_feats))
    #     l_shape = (total_rows, self.num_horizons) if self.is_multi_horizon else (total_rows,)
    #     l_map = np.memmap(self.cache_dir / "labels.npy", dtype='int64', mode='w+', shape=l_shape)
    #     v_map = np.memmap(self.cache_dir / "vols.npy", dtype='float32', mode='w+', shape=(total_rows,))
    #     t_map = np.memmap(self.cache_dir / "timestamps.npy", dtype='int64', mode='w+', shape=(total_rows,))
    #     
    #     mid_prices = df["mid_price"].to_numpy()
    #     vols_arr = compute_target_vol(mid_prices, window=self.vol_window)[self.seq_len - 1:]
    #     
    #     has_ts = "timestamp_ms" in df.columns
    #     if not has_ts and "timestamp" in df.columns:
    #         df = df.rename({"timestamp": "timestamp_ms"})
    #         has_ts = True
    #
    #     # Построчная запись (чанками)
    #     for i in range(total_rows):
    #         f_map[i] = df.select(feat_cols).slice(i, self.seq_len).to_numpy()
    #         if self.is_multi_horizon:
    #             l_map[i] = [df[lc][i + self.seq_len - 1] for lc in self.label_cols]
    #         else:
    #             l_map[i] = df["label"][i + self.seq_len - 1]
    #         v_map[i] = vols_arr[i]
    #         if has_ts:
    #             t_map[i] = df["timestamp_ms"][i + self.seq_len - 1]
    #         else:
    #             t_map[i] = i + self.seq_len - 1
    #
    #     w_arr = self._calculate_time_weights(t_map[:], l_map[:]).numpy()
    #     w_map = np.memmap(self.cache_dir / "weights.npy", dtype='float32', mode='w+', shape=(total_rows,))
    #     w_map[:] = w_arr
    #     
    #     with open(meta_path, 'w') as f:
    #         json.dump({'n_samples': total_rows, 'features_shape': [total_rows, self.seq_len, n_feats], 
    #                   'labels_shape': list(l_shape), 'label_cols': self.label_cols, 'seq_len': self.seq_len}, f)
    #     
    #     f_map.flush(); l_map.flush(); v_map.flush(); t_map.flush(); w_map.flush()
    #     self.features_seq, self.labels, self.vols, self.timestamps = f_map, l_map, v_map, t_map
    #     self.sample_weights = torch.from_numpy(w_map[:])
    #     self.regime_ids = np.zeros(total_rows, dtype=np.int64)
    #     
    #     # Задача 308: Инициализация кэшей для memmap режима
    #     vib_path = self.cache_dir / "vib_cache.npy"
    #     past_ret_path = self.cache_dir / "past_ret_cache.npy"
    #     
    #     if vib_path.exists() and past_ret_path.exists():
    #         # Загружаем существующие кэши
    #         self.vib_cache = np.memmap(vib_path, dtype='float32', mode='r', shape=(total_rows + self.seq_len - 1,))
    #         self.past_ret_cache = np.memmap(past_ret_path, dtype='float32', mode='r', shape=(total_rows + self.seq_len - 1, len(self.past_returns_lags)))
    #         print(f"[{self.__class__.__name__}] Loaded VIB/PastRet caches from memmap files")
    #     else:
    #         # Вычисляем и сохраняем кэши
    #         print(f"[{self.__class__.__name__}] Memmap cache for VIB/PastRet not found. Computing...")
    #         
    #         bid_v_raw = df.select([f"feat_bid_v_{i}" for i in range(self.n_levels)]).to_numpy()
    #         ask_v_raw = df.select([f"feat_ask_v_{i}" for i in range(self.n_levels)]).to_numpy()
    #         mid_prices = df["mid_price"].to_numpy()
    #         
    #         vib_raw = compute_depth_imbalance_globally(bid_v_raw, ask_v_raw)
    #         past_ret_raw = compute_past_returns_globally(mid_prices, lags=self.past_returns_lags)
    #         
    #         # Нормализация
    #         temp_df = pl.DataFrame({
    #             "feat_vib": vib_raw.astype(np.float32),
    #             "feat_past_ret": past_ret_raw[:, -1].astype(np.float32)
    #         })
    #         
    #         if temp_df.height > 0 and self.scaler_type in ("robust", "winsor_robust"):
    #             norm_cache = Normalizer(output_path=None)
    #             norm_cache.scaler_type = self.scaler_type
    #             norm_cache.winsor_limits = self.winsor_limits
    #             norm_cache.fit(temp_df)
    #             temp_norm = norm_cache.transform(temp_df)
    #             
    #             # Создаем memmap файлы и записываем данные
    #             self.vib_cache = np.memmap(vib_path, dtype='float32', mode='w+', shape=vib_raw.shape)
    #             self.vib_cache[:] = temp_norm["feat_vib"].to_numpy().astype(np.float32)
    #             
    #             self.past_ret_cache = np.memmap(past_ret_path, dtype='float32', mode='w+', shape=vib_raw.shape)
    #             self.past_ret_cache[:] = temp_norm["feat_past_ret"].to_numpy().astype(np.float32)
    #         else:
    #             self.vib_cache = np.memmap(vib_path, dtype='float32', mode='w+', shape=vib_raw.shape)
    #             self.vib_cache[:] = vib_raw.astype(np.float32)
    #             self.past_ret_cache = np.memmap(past_ret_path, dtype='float32', mode='w+', shape=vib_raw.shape)
    #             self.past_ret_cache[:] = past_ret_raw[:, -1].astype(np.float32)
    #         
    #         self.vib_cache.flush()
    #         self.past_ret_cache.flush()
    #         print(f"[{self.__class__.__name__}] VIB/PastRet caches created and saved to memmap files")



    def __len__(self):
        max_lag = max(self.past_returns_lags) if self.past_returns_lags else 100
        base_len = self.total_samples if self.data_mode == "streaming" else len(self.x_raw) - self.seq_len + 1
        return max(0, base_len - max_lag)

    def __getitem__(self, idx):
        # Задача 310.2.1: Добавляем смещение (offset) равное максимальному лагу, 
        # чтобы первый же сэмпл содержал валидную историю PastReturns
        max_lag = max(self.past_returns_lags) if self.past_returns_lags else 100
        idx = idx + max_lag

        if self.data_mode == "streaming":
            return self._getitem_streaming(idx)
        
        # Memory/Memmap access
        x_raw = self.x_raw[idx : idx + self.seq_len] if self.data_mode == "memory" else self.features_seq[idx]
        y = self.labels[idx + self.seq_len - 1] if self.data_mode == "memory" else self.labels[idx]
        v = self.vols[idx]
        w = self.sample_weights[idx]
        regime_id = torch.tensor(self.regime_ids[idx]).long()
        
        # Извлекаем дополнительные данные для расширенной аналитики (Задача 313.4)
        ts = self.timestamps[idx + self.seq_len - 1]
        mid = self.mid_prices[idx + self.seq_len - 1]
        
        if hasattr(self, 'future_returns') and self.future_returns is not None:
            f_ret = self.future_returns[idx + self.seq_len - 1]
        else:
            f_ret = 0.0
        
        return self._process_sample(x_raw, y, v, w, regime_id, ts, mid, idx=idx, f_ret=f_ret)

    # ОТКЛЮЧЕНО: Используется только для streaming режима
    # def _getitem_streaming(self, idx):
    #     if self._cache_batch is None or not (self._cache_start_idx <= idx < self._cache_end_idx):
    #         start = idx
    #         batch_df = self.lazy_df.slice(start, self._batch_size + self.seq_len - 1).collect(streaming=True)
    #         
    #         # Задача 308-4: Расчет индикаторов до нормализации в streaming моде
    #         bid_cols = [f"feat_bid_v_{i}" for i in range(self.n_levels)]
    #         ask_cols = [f"feat_ask_v_{i}" for i in range(self.n_levels)]
    #         
    #         raw_bid_sum = batch_df.select(pl.sum_horizontal(bid_cols)).to_numpy().flatten()
    #         raw_ask_sum = batch_df.select(pl.sum_horizontal(ask_cols)).to_numpy().flatten()
    #         denom = raw_bid_sum + raw_ask_sum
    #         vib_raw = np.where(denom > 1e-9, (raw_bid_sum - raw_ask_sum) / (denom + 1e-9), 0.0).astype(np.float32)
    #         
    #         mid_prices = batch_df["mid_price"].to_numpy()
    #         log_prices = np.log(np.maximum(mid_prices, 1e-9))
    #         past_ret_raw = np.zeros(len(mid_prices), dtype=np.float32)
    #         if len(log_prices) > self.seq_len: # Используем seq_len как лаг в streaming моде
    #             past_ret_raw[self.seq_len:] = log_prices[self.seq_len:] - log_prices[:-self.seq_len]
    #         
    #         # Нормализация индикаторов
    #         temp_df_ind = pl.DataFrame({"feat_vib_val": vib_raw, "feat_past_ret_val": past_ret_raw})
    #         norm_temp = Normalizer(output_path=None, scale_multiplier=self.scale_multiplier)
    #         norm_temp.scaler_type = self.scaler_type
    #         norm_temp.winsor_limits = self.winsor_limits
    #         norm_temp.fit(temp_df_ind)
    #         transformed_ind = norm_temp.transform(temp_df_ind)
    #         
    #         self.vib_cache = transformed_ind["feat_vib_val"].to_numpy().astype(np.float32)
    #         self.past_ret_cache = transformed_ind["feat_past_ret_val"].to_numpy().astype(np.float32)
    #         
    #         # Формирование последовательностей
    #         feat_data = batch_df.select(self.feat_cols).to_numpy()
    #         self._cache_batch = np.stack([feat_data[i:i+self.seq_len] for i in range(len(batch_df) - self.seq_len + 1)], axis=0)
    #         
    #         if self.is_multi_horizon:
    #             l_lists = [batch_df.select(lc).slice(self.seq_len-1).to_series().to_numpy() for lc in self.label_cols]
    #             self._cache_labels = np.stack(l_lists, axis=1)
    #         else:
    #             self._cache_labels = batch_df.select("label").slice(self.seq_len-1).to_series().to_numpy()
    #         
    #         self._cache_vols = self.vols[start : start + len(self._cache_labels)]
    #         self._cache_start_idx, self._cache_end_idx = start, start + len(self._cache_labels)
    #
    #     off = idx - self._cache_start_idx
    #     x_raw_current = self._cache_batch[off]
    #     
    #     # Update current cache indices
    #     self._current_cache_off = off
    #
    #     return self._process_sample(
    #         x_raw_current, 
    #         self._cache_labels[off], 
    #         self._cache_vols[off], 
    #         self.sample_weights[idx], 
    #         torch.tensor(self.regime_ids[idx]).long(), 
    #         idx=off # Using offset because caches in streaming are batch-local
    #     )

    def normalize_channel(self, channel_data: torch.Tensor, channel_idx: int) -> torch.Tensor:
        """
        Нормализует канал используя статистики из normalizer.
        Векторизованная версия (Задача 312.2.1).
        """
        if self.normalizer is None:
            return channel_data
        
        n_levels = channel_data.shape[1]
        start_feat_idx = channel_idx * n_levels
        
        # Векторизованное извлечение параметров для всех уровней сразу
        if self.normalizer.scaler_type == "zscore":
            means = []
            stds = []
            for level in range(n_levels):
                feat_idx = start_feat_idx + level
                param_key = f"feat_{feat_idx}"
                params = self.normalizer.params.get(param_key, {})
                means.append(params.get("mean", 0.0))
                stds.append(params.get("std", 1.0))
            
            mean_tensor = torch.tensor(means, device=channel_data.device, dtype=channel_data.dtype)
            std_tensor = torch.tensor(stds, device=channel_data.device, dtype=channel_data.dtype)
            return (channel_data - mean_tensor) / (std_tensor + 1e-8)
        
        elif self.normalizer.scaler_type == "robust":
            medians = []
            iqrs = []
            for level in range(n_levels):
                feat_idx = start_feat_idx + level
                param_key = f"feat_{feat_idx}"
                params = self.normalizer.params.get(param_key, {})
                medians.append(params.get("median", 0.0))
                iqrs.append(params.get("iqr", 1.0))
            
            median_tensor = torch.tensor(medians, device=channel_data.device, dtype=channel_data.dtype)
            iqr_tensor = torch.tensor(iqrs, device=channel_data.device, dtype=channel_data.dtype)
            return (channel_data - median_tensor) / (iqr_tensor + 1e-8)
        
        return channel_data

    def _calculate_6_channels_raw(self, ask_p, ask_v, bid_p, bid_v, vib_raw=None, pr_raw=None, ofi_precomputed=None):
        """
        Единый метод формирования 11 каналов LOB (Задача 319).
        Каналы: MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread,
                DeltaImb, DeltaSpread.
        Вход: torch.Tensors (seq_len, 50).

        Каналы:
        ch[0]: Microprice Deviation — предсказательное отклонение microprice от mid
        ch[1]: Volume — среднее логарифмов объемов
        ch[2]: Static Imbalance — дисбаланс объемов [-1, 1]
        ch[3]: OFI — Order Flow Imbalance (Cont-Kukanov-Stoikov, per-tick)
        ch[4]: VIB — Volume Imbalance (суммарный дисбаланс по всем уровням)
        ch[5]: Ret_10 — short-term momentum (лог-возврат за 10 тиков)
        ch[6]: Ret_50 — medium-term momentum (лог-возврат за 50 тиков)
        ch[7]: Ret_100 — long-term trend (лог-возврат за 100 тиков)
        ch[8]: Spread — нормализованный спред (ask_0 - bid_0) / mid
        ch[9]: DeltaImb — скорость изменения Imbalance (first derivative)
        ch[10]: DeltaSpread — скорость изменения спреда
        """
        eps = 1e-8

        # Восстанавливаем сырые объемы для дисбалансов
        # Clamp to prevent float32 overflow (exp(20) ≈ 485M is safe)
        ask_v_raw = torch.exp(torch.clamp(ask_v, max=20.0)) - 1.0
        bid_v_raw = torch.exp(torch.clamp(bid_v, max=20.0)) - 1.0

        # Сырые цены лучшего уровня (относительные: price/mid - 1)
        ask_p_0 = ask_p[:, 0]  # (seq_len,)
        bid_p_0 = bid_p[:, 0]  # (seq_len,)
        ask_v_0 = ask_v_raw[:, 0]  # (seq_len,)
        bid_v_0 = bid_v_raw[:, 0]  # (seq_len,)

        # ch[0]: Microprice Deviation (Задача 318.3: делим на spread/2 для диапазона [-1, 1])
        microprice = (bid_p_0 * ask_v_0 + ask_p_0 * bid_v_0) / (ask_v_0 + bid_v_0 + eps)
        spread_width = ask_p_0 - bid_p_0  # ширина спреда в относительных ценах
        microprice_dev = microprice / (spread_width / 2.0 + eps)
        price_ch_raw = microprice_dev.unsqueeze(-1).expand(-1, 50)

        # ch[1]: Volume — среднее логарифмов
        vol_ch_raw = (ask_v + bid_v) / 2.0

        # ch[2]: Static Imbalance [-1, 1]
        denom = bid_v_raw + ask_v_raw + eps
        imb_ch_raw = (bid_v_raw - ask_v_raw) / denom

        # ch[3]: OFI — настоящий Cont-Kukanov-Stoikov Order Flow Imbalance (Задача 318.2)
        if ofi_precomputed is not None:
            ofi_raw = ofi_precomputed.unsqueeze(-1).expand(-1, 50)
        else:
            ofi_raw = torch.diff(imb_ch_raw, dim=0, prepend=imb_ch_raw[:1])

        # ch[4]: VIB (Volume Imbalance)
        if vib_raw is None:
            bv_sum = bid_v_raw.sum(dim=1)
            av_sum = ask_v_raw.sum(dim=1)
            vib_val = (bv_sum - av_sum) / (bv_sum + av_sum + eps)
            vib_ch_raw = vib_val.unsqueeze(-1).repeat(1, 50)
        else:
            vib_ch_raw = vib_raw.unsqueeze(-1).repeat(1, 50) if vib_raw.ndim == 1 else vib_raw

        # ch[5]: Ret_10 (short-term momentum)
        if pr_raw is None:
            seq_len = ask_p.shape[0]
            mid_approx = ask_p[:, 0] + 1.0  # восстановленный mid (относительный)

            if seq_len >= 10:
                ret_10 = torch.log(torch.clamp(mid_approx[-1], min=eps)) - \
                         torch.log(torch.clamp(mid_approx[-10], min=eps))
            else:
                ret_10 = torch.tensor(0.0, device=ask_p.device)
            ret_10_ch_raw = ret_10.unsqueeze(0).unsqueeze(-1).expand(seq_len, 50)
        else:
            ret_10_ch_raw = pr_raw[:, 0].unsqueeze(-1).expand(-1, 50) if pr_raw.ndim > 1 else pr_raw.unsqueeze(-1).expand(-1, 50)

        # ch[6]: Ret_50 (medium-term momentum)
        if pr_raw is None:
            if seq_len >= 50:
                ret_50 = torch.log(torch.clamp(mid_approx[-1], min=eps)) - \
                         torch.log(torch.clamp(mid_approx[-50], min=eps))
            else:
                ret_50 = torch.tensor(0.0, device=ask_p.device)
            ret_50_ch_raw = ret_50.unsqueeze(0).unsqueeze(-1).expand(seq_len, 50)
        else:
            ret_50_ch_raw = pr_raw[:, 1].unsqueeze(-1).expand(-1, 50)

        # ch[7]: Ret_100 (long-term trend)
        if pr_raw is None:
            if seq_len >= 100:
                ret_100 = torch.log(torch.clamp(mid_approx[-1], min=eps)) - \
                          torch.log(torch.clamp(mid_approx[-100], min=eps))
            else:
                ret_100 = torch.tensor(0.0, device=ask_p.device)
            ret_100_ch_raw = ret_100.unsqueeze(0).unsqueeze(-1).expand(seq_len, 50)
        else:
            ret_100_ch_raw = pr_raw[:, 2].unsqueeze(-1).expand(-1, 50)

        # ch[8]: Spread — нормализованный спред
        spread = (ask_p_0 - bid_p_0).unsqueeze(-1).expand(-1, 50)

        # ===== НОВЫЕ TEMPORAL DERIVATIVE КАНАЛЫ (Задача 318.4) =====

        # ch[9]: DeltaImb — скорость изменения Imbalance (first derivative)
        delta_imb = torch.diff(imb_ch_raw[:, 0], dim=0, prepend=torch.zeros(1, device=imb_ch_raw.device))
        delta_imb_ch = delta_imb.unsqueeze(-1).expand(-1, 50)  # (seq_len, 50)

        # ch[10]: DeltaSpread — скорость изменения спреда
        spread_1d = ask_p_0 - bid_p_0  # (seq_len,)
        delta_spread = torch.diff(spread_1d, dim=0, prepend=torch.zeros(1, device=spread_1d.device))
        delta_spread_ch = delta_spread.unsqueeze(-1).expand(-1, 50)

        return price_ch_raw, vol_ch_raw, imb_ch_raw, ofi_raw, vib_ch_raw, \
               ret_10_ch_raw, ret_50_ch_raw, ret_100_ch_raw, spread, \
               delta_imb_ch, delta_spread_ch

    def _compute_channels_for_normalization(self, data: Union[pl.DataFrame, List[int], np.ndarray]) -> pl.DataFrame:
        """
        Вычисляет каналы из сырых данных для обучения нормализатора.
        data может быть DataFrame, списком индексов или NumPy массивом.
        """
        if isinstance(data, list):
            # Если переданы индексы, берем данные из self.x_raw
            if not hasattr(self, 'x_raw') or self.x_raw is None:
                if isinstance(self.features, np.ndarray):
                    x_data = self.features[data]
                else:
                    raise ValueError("Indices provided for _compute_channels_for_normalization, but x_raw is missing.")
            else:
                x_data = self.x_raw[data]
            
            ask_p = torch.from_numpy(x_data[:, self.ask_p_indices]).float()
            ask_v = torch.from_numpy(x_data[:, self.ask_v_indices]).float()
            bid_p = torch.from_numpy(x_data[:, self.bid_p_indices]).float()
            bid_v = torch.from_numpy(x_data[:, self.bid_v_indices]).float()
            
            vib_raw = None
            if hasattr(self, 'vib_cache') and self.vib_cache is not None:
                vib_raw = torch.from_numpy(self.vib_cache[data]).float()

            pr_raw = None
            if hasattr(self, 'past_ret_cache') and self.past_ret_cache is not None:
                r10 = self.past_ret_cache[10][data]
                r50 = self.past_ret_cache[50][data]
                r100 = self.past_ret_cache[100][data]
                pr_raw = np.stack([r10, r50, r100], axis=1)  # (N, 3)
                pr_raw = torch.from_numpy(pr_raw).float()

            # Задача 318.2: OFI из precomputed cache
            ofi_precomp = None
            if hasattr(self, 'ofi_cache') and self.ofi_cache is not None:
                ofi_precomp = torch.from_numpy(self.ofi_cache[data]).float()
        else:
            # Для DataFrame (streaming/temp_ds)
            df_raw = data
            ask_p = torch.from_numpy(df_raw.select([f"feat_ask_p_{i}" for i in range(50)]).to_numpy()).float()
            bid_p = torch.from_numpy(df_raw.select([f"feat_bid_p_{i}" for i in range(50)]).to_numpy()).float()
            ask_v = torch.from_numpy(df_raw.select([f"feat_ask_v_{i}" for i in range(50)]).to_numpy()).float()
            bid_v = torch.from_numpy(df_raw.select([f"feat_bid_v_{i}" for i in range(50)]).to_numpy()).float()
            vib_raw, pr_raw = None, None
            ofi_precomp = None

        # Используем единый метод расчета (11 каналов после Задача 319)
        p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw, \
            di_raw, ds_raw = self._calculate_6_channels_raw(
                ask_p, ask_v, bid_p, bid_v, vib_raw, pr_raw, ofi_precomputed=ofi_precomp
            )

        channels = torch.cat([
            p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw,
            di_raw, ds_raw
        ], dim=1).numpy()
        return pl.DataFrame(channels, schema=[f"feat_{i}" for i in range(550)])

    def _process_sample(self, x_raw, y, v, w, regime_id, ts, mid, idx=None, f_ret=None):
        # NaN protection (Задача 094-2)
        x_raw = np.nan_to_num(x_raw, nan=0.0)
        
        # Извлекаем каждый тип колонок напрямую
        ask_p = torch.from_numpy(x_raw[:, self.ask_p_indices].copy()).float()
        ask_v = torch.from_numpy(x_raw[:, self.ask_v_indices].copy()).float()
        bid_p = torch.from_numpy(x_raw[:, self.bid_p_indices].copy()).float()
        bid_v = torch.from_numpy(x_raw[:, self.bid_v_indices].copy()).float()
        
        # Аугментация
        if self.is_train and torch.rand(1, generator=self.generator).item() < self.augment_prob:
            x_temp = torch.cat([ask_p, ask_v, bid_p, bid_v], dim=1)
            if self.use_symmetric_flip and torch.rand(1, generator=self.generator).item() < 0.5:
                x_temp, y = apply_symmetric_flip(x_temp, y, self.price_cols, self.ask_cols, self.bid_cols)
            if self.volume_jitter_range > 0:
                x_temp = apply_volume_jitter(x_temp, self.volume_jitter_range, self.vol_cols, self.generator)
            ask_p, ask_v = x_temp[:, 0:50], x_temp[:, 50:100]
            bid_p, bid_v = x_temp[:, 100:150], x_temp[:, 150:200]
        
        # Подготовка VIB
        vib_raw = None
        if idx is not None and hasattr(self, 'vib_cache') and self.vib_cache is not None:
            vib_raw = torch.from_numpy(self.vib_cache[idx : idx + self.seq_len].copy()).float()
        elif self.vib_idx >= 0:
            vib_raw = torch.from_numpy(x_raw[:, self.vib_idx].copy()).float()
            
        # Подготовка PastRet (3 лага: 10, 50, 100)
        pr_raw = None
        if idx is not None and hasattr(self, 'past_ret_cache') and self.past_ret_cache is not None:
            r10 = self.past_ret_cache[10][idx : idx + self.seq_len]
            r50 = self.past_ret_cache[50][idx : idx + self.seq_len]
            r100 = self.past_ret_cache[100][idx : idx + self.seq_len]
            pr_raw = np.stack([r10, r50, r100], axis=1)  # (seq_len, 3)
            pr_raw = torch.from_numpy(pr_raw).float()
        elif self.past_ret_indices:
            # fallback для non-memory режима — берём первый доступный лаг
            pr_raw = torch.from_numpy(x_raw[:, self.past_ret_indices[0]].copy()).float().unsqueeze(-1)
            # pad до 3 колонок нулями
            pr_raw = torch.cat([pr_raw, torch.zeros_like(pr_raw), torch.zeros_like(pr_raw)], dim=-1)

        # Подготовка OFI из precomputed cache (Задача 318.2)
        ofi_precomp = None
        if idx is not None and hasattr(self, 'ofi_cache') and self.ofi_cache is not None:
            ofi_precomp = torch.from_numpy(self.ofi_cache[idx : idx + self.seq_len].copy()).float()

        # Расчет сырых каналов (Единый источник правды, 11 каналов после Задача 319)
        p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw, \
            di_raw, ds_raw = self._calculate_6_channels_raw(
                ask_p, ask_v, bid_p, bid_v, vib_raw, pr_raw, ofi_precomputed=ofi_precomp
            )

        # Нормализация (Векторизованная, 11 каналов)
        price_ch = self.normalize_channel(p_raw, channel_idx=0)
        vol_ch = self.normalize_channel(v_raw, channel_idx=1)
        imb_ch = self.normalize_channel(i_raw, channel_idx=2)
        ofi_ch = self.normalize_channel(o_raw, channel_idx=3)
        vib_ch = self.normalize_channel(vi_raw, channel_idx=4)
        ret10_ch = self.normalize_channel(ret10_raw, channel_idx=5)
        ret50_ch = self.normalize_channel(ret50_raw, channel_idx=6)
        ret100_ch = self.normalize_channel(ret100_raw, channel_idx=7)
        spread_ch = self.normalize_channel(sp_raw, channel_idx=8)
        delta_imb_ch = self.normalize_channel(di_raw, channel_idx=9)
        delta_spread_ch = self.normalize_channel(ds_raw, channel_idx=10)

        # Собираем итоговый тензор (Seq, 11, 50)
        x_final = torch.stack([
            price_ch, vol_ch, imb_ch, ofi_ch, vib_ch,
            ret10_ch, ret50_ch, ret100_ch, spread_ch,
            delta_imb_ch, delta_spread_ch
        ], dim=1)

        # Задача 320.5: Сохраняем копию до clamp для диагностики saturation
        x_pre_clip = x_final.clone()

        # Защита 1: Разный клиппинг для разных каналов согласно статистике
        # Каналы с высокой волатильностью (OFI, DeltaImb, DeltaSpread) ужесточем до [-3, 3]
        # Остальные оставляем [-5, 5]
        x_final = torch.clamp(x_final, -5.0, 5.0)  # Сначала общий клип
        # Ужесточение для критических каналов
        x_final[:, 3, :] = torch.clamp(x_final[:, 3, :], -3.0, 3.0)   # OFI
        x_final[:, 9, :] = torch.clamp(x_final[:, 9, :], -3.0, 3.0)   # DeltaImb
        x_final[:, 10, :] = torch.clamp(x_final[:, 10, :], -3.0, 3.0) # DeltaSpread

        # Сохраняем пост-клип версию (перед nan_to_num) для диагностики
        x_post_clip = x_final.clone()

        # Защита 2: Замена NaN/Inf на безопасные значения
        x_final = torch.nan_to_num(x_final, nan=0.0, posinf=5.0, neginf=-5.0)

        # Логирование saturation для отладки (только для ограниченного числа сэмплов)
        if idx is not None and self._clip_diag_prints < self.max_clip_diag_prints:
            self._log_clip_saturation(x_pre_clip, x_post_clip, idx)
            self._clip_diag_prints += 1

        # Расширенная диагностика (Задача 316)
        if idx is not None and 100 <= idx <= 101:
            print(f"\n[ДИАГНОСТИКА 316] Сэмпл idx={idx}:")
            if self.normalizer and self.normalizer.params:
                print("  ПАРАМЕТРЫ НОРМАЛИЗАЦИИ (Level 0):")
                for ch_idx, name in enumerate(["MicropriceDev", "Vol"]):
                    p = self.normalizer.params.get(f"feat_{ch_idx*50}", {})
                    print(f"    {name}: mean={p.get('mean', 0.0):.6f}, std={p.get('std', 1.0):.6f}")

            channels_names = ["MicropriceDev", "Vol", "Imb", "OFI", "VIB", "Ret_10", "Ret_50", "Ret_100", "Spread",
                              "DeltaImb", "DeltaSpread"]
            for i, name in enumerate(channels_names):
                ch = x_final[:, i, :]
                print(f"  Channel {i} ({name}) ПОСЛЕ CLAMP: min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}, std={ch.std():.4f}")

        # Формируем extra_data (совместимость с train.py)
        extra_data = {
            "vol": torch.tensor(v).float() if not isinstance(v, torch.Tensor) else v,
            "weight": torch.tensor(w).float() if not isinstance(w, torch.Tensor) else w,
            "regime_id": regime_id if isinstance(regime_id, torch.Tensor) else torch.tensor(int(regime_id)),
            "ts": int(ts),
            "mid": float(mid),
            "f_ret": f_ret if f_ret is not None else (torch.zeros(self.num_horizons) if self.is_multi_horizon else torch.tensor(0.0))
        }
        
        target = torch.tensor(y).long()

        # ЗАЩИТА: Валидация target (должен быть в [0, 2])
        if (target < 0).any() or (target > 2).any():
            invalid_min, invalid_max = target.min().item(), target.max().item()
            print(f"[WARN] Invalid target at idx={idx}: range [{invalid_min}, {invalid_max}]. Clamping to [0, 2].")
            target = torch.clamp(target, 0, 2)

        ts_val = int(ts) if ts is not None else 0
        mid_val = float(mid) if mid is not None else 0.0

        # ЗАЩИТА: Проверка x_final на NaN/Inf
        if torch.isnan(x_final).any() or torch.isinf(x_final).any():
            nan_cnt = torch.isnan(x_final).sum().item()
            inf_cnt = torch.isinf(x_final).sum().item()
            print(f"[WARN] NaN/Inf in x_final at idx={idx}: nan={nan_cnt}, inf={inf_cnt}. Replacing with 0.")
            x_final = torch.nan_to_num(x_final, nan=0.0, posinf=1e4, neginf=-1e4)

        return x_final, target, torch.tensor(ts_val).long(), torch.tensor(mid_val).float(), y, extra_data

    def _log_clip_saturation(self, x_pre: torch.Tensor, x_post: torch.Tensor, idx: int):
        """
        Логирует статистику saturation (доля значений, вышедших за лимиты) для каналов.
        x_pre: тензор до clamp (seq, ch, level)
        x_post: тензор после clamp (до nan_to_num)
        """
        print(f"\n[CLIP DIAG] Sample idx={idx} saturation report:")
        # Перемещаем на CPU для вычислений
        x_pre = x_pre.cpu()
        x_post = x_post.cpu()
        for ch_idx, ch_name in enumerate(self.channel_names):
            pre_flat = x_pre[:, ch_idx, :]
            post_flat = x_post[:, ch_idx, :]
            # Определяем лимит для канала
            limit = 3.0 if ch_idx in (3, 9, 10) else 5.0

            below = (pre_flat < -limit).sum().item()
            above = (pre_flat > limit).sum().item()
            total = pre_flat.numel()
            pct_below = below / total * 100 if total > 0 else 0.0
            pct_above = above / total * 100 if total > 0 else 0.0
            pct_total = (below + above) / total * 100 if total > 0 else 0.0

            pre_min = pre_flat.min().item()
            pre_max = pre_flat.max().item()
            pre_mean = pre_flat.mean().item()
            pre_std = pre_flat.std().item()
            post_min = post_flat.min().item()
            post_max = post_flat.max().item()
            post_mean = post_flat.mean().item()
            post_std = post_flat.std().item()

            print(f"  Channel {ch_idx} ({ch_name}): limit={limit}, %below={pct_below:.2f}, %above={pct_above:.2f}, %total={pct_total:.2f}")
            print(f"    pre: min={pre_min:.4f}, max={pre_max:.4f}, mean={pre_mean:.4f}, std={pre_std:.4f}")
            print(f"    post: min={post_min:.4f}, max={post_max:.4f}, mean={post_mean:.4f}, std={post_std:.4f}")

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
    
    # Задача 306.4.1: Переименование колонок стакана с добавлением префикса feat_
    # Это обеспечивает унификацию имен и автоматический поиск признаков по префиксу
    rename_map = {}
    for i in range(50):
        rename_map[f"ask_p_{i}"] = f"feat_ask_p_{i}"
        rename_map[f"ask_v_{i}"] = f"feat_ask_v_{i}"
        rename_map[f"bid_p_{i}"] = f"feat_bid_p_{i}"
        rename_map[f"bid_v_{i}"] = f"feat_bid_v_{i}"
    # Задача 318.1: Сохраняем last_update_id как feat_update_id для расчёта OFI
    rename_map["last_update_id"] = "feat_update_id"
    
    # Применяем переименование только для существующих колонок
    existing_cols = merged.collect_schema().names() if lazy else merged.columns
    rename_map = {k: v for k, v in rename_map.items() if k in existing_cols}
    if rename_map:
        merged = merged.rename(rename_map)
    
    # Задача 306.2.1: Удаляем служебные колонки ПЕРЕД конвертацией в numpy/memmap
    # Используем правильные имена согласно уточнению пользователя
    # Задача 318.1: last_update_id переименован в feat_update_id, не удаляем
    meta_cols = ["timestamp_ms", "symbol"]
    merged = merged.drop([c for c in meta_cols if c in merged.columns])
    
    # Задача 306.4.1: Сортировка колонок для детерминированного порядка
    # Это гарантирует, что порядок колонок не зависит от порядка записи в Parquet
    all_cols = merged.collect_schema().names() if lazy else merged.columns
    merged = merged.select(sorted(all_cols))
    
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
