#!/usr/bin/env python3
"""
Задача 223: Тепловые карты глубины стакана (Orderbook Depth Heatmaps)

Визуализация динамики ликвидности в стакане через тепловые карты.
Использует Signed BPS координаты (Asks > 0, Bids < 0) для анализа структуры стен.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import polars as pl
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Генерация тепловых карт глубины стакана (LOB Heatmaps)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Символ токена для анализа (например, BTCUSDT)"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Начало временного диапазона (ISO формат: YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Конец временного диапазона (ISO формат: YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--depth_bps",
        type=float,
        default=100.0,
        help="Глубина отображения в BPS (по умолчанию: 100)"
    )
    parser.add_argument(
        "--time_bin_seconds",
        type=float,
        default=1.0,
        help="Размер временного бина в секундах (по умолчанию: 1.0)"
    )
    parser.add_argument(
        "--bps_bin_size",
        type=float,
        default=1.0,
        help="Размер бина по BPS (по умолчанию: 1.0)"
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Экспортировать матрицу плотности в CSV"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["html", "png", "both"],
        default="html",
        help="Формат выходного файла (по умолчанию: html)"
    )
    
    return parser.parse_args()


def load_orderbook_data(
    symbol: str,
    start_time: datetime,
    end_time: datetime
) -> pl.DataFrame:
    """
    Загрузка данных стакана из Parquet файлов с использованием ленивой загрузки.
    
    Схема данных (dump.rs):
    - timestamp_ms: временная метка в миллисекундах
    - ask_p_0...ask_p_49, ask_v_0...ask_v_49: цены и объемы asks
    - bid_p_0...bid_p_49, bid_v_0...bid_v_49: цены и объемы bids
    
    Args:
        symbol: Символ токена
        start_time: Начало временного диапазона
        end_time: Конец временного диапазона
        
    Returns:
        DataFrame с развернутыми данными: timestamp, bps, volume
    """
    data_dir = Path(f"./bots/{symbol}/data/raw")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Директория с данными не найдена: {data_dir}")
    
    # Поиск всех parquet файлов
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"Parquet файлы не найдены в {data_dir}")
    
    print(f"Найдено {len(parquet_files)} parquet файлов")
    
    # Конвертируем временные границы в миллисекунды
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    # Ленивая загрузка с выбором ТОЛЬКО необходимых колонок (экономия RAM)
    # Выбор колонок происходит ДО .collect() для эффективной фильтрации на уровне Parquet
    df = (
        pl.scan_parquet(parquet_files)
        .filter(
            (pl.col("timestamp_ms") >= start_ms) &
            (pl.col("timestamp_ms") <= end_ms)
        )
        .select([
            pl.col("timestamp_ms"),
            pl.col("^ask_p_.*$"),  # Все колонки ask_p_0...ask_p_49
            pl.col("^ask_v_.*$"),  # Все колонки ask_v_0...ask_v_49
            pl.col("^bid_p_.*$"),  # Все колонки bid_p_0...bid_p_49
            pl.col("^bid_v_.*$"),  # Все колонки bid_v_0...bid_v_49
        ])
        .with_columns([
            ((pl.col("ask_p_0") + pl.col("bid_p_0")) / 2).alias("mid_price"),
            (pl.col("timestamp_ms") / 1000.0).alias("timestamp")
        ])
        .collect()
    )
    
    print(f"Загружено {len(df)} записей стакана")
    
    return df


def load_trades_data(
    symbol: str,
    start_time: datetime,
    end_time: datetime
) -> Optional[pl.DataFrame]:
    """
    Загрузка данных сделок из CSV файла.
    
    Формат (trade_logger.rs):
    - time: RFC3339 формат
    - side: "Buy" или "Sell"
    - price: цена исполнения
    - qty: объем сделки
    
    Args:
        symbol: Символ токена
        start_time: Начало временного диапазона
        end_time: Конец временного диапазона
        
    Returns:
        DataFrame с сделками или None если файл не найден
    """
    trades_file = Path(f"./bots/{symbol}/logs/trades.csv")
    
    if not trades_file.exists():
        print(f"Предупреждение: файл сделок не найден: {trades_file}")
        return None
    
    try:
        df = (
            pl.read_csv(trades_file)
            .with_columns([
                pl.col("time").str.to_datetime()
            ])
            .filter(
                (pl.col("time") >= start_time) &
                (pl.col("time") <= end_time)
            )
            .select(["time", "side", "price", "qty"])
        )
        
        print(f"Загружено {len(df)} сделок")
        return df
        
    except Exception as e:
        print(f"Ошибка при загрузке сделок: {e}")
        return None


def compute_signed_bps_vectorized(
    df: pl.DataFrame,
    depth_bps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Векторизованное преобразование данных стакана в Signed BPS координаты.
    
    Использует stack для развертывания колонок ask_p_*/ask_v_*/bid_p_*/bid_v_*
    в длинный формат, затем вычисляет BPS для каждого уровня.
    
    Args:
        df: DataFrame с данными стакана
        depth_bps: Максимальная глубина в BPS
        
    Returns:
        Кортеж (timestamps, bps_values, volumes) в плоском формате
    """
    LOB_DEPTH = 50
    
    # Развертываем Asks: создаем список (price, volume) пар для каждого уровня
    ask_data = []
    for i in range(LOB_DEPTH):
        ask_level = (
            df.select([
                "timestamp",
                "mid_price",
                pl.col(f"ask_p_{i}").alias("price"),
                pl.col(f"ask_v_{i}").alias("volume")
            ])
            .filter(pl.col("volume") > 0)  # Только ненулевые объемы
        )
        ask_data.append(ask_level)
    
    # Объединяем все уровни Asks
    asks_df = pl.concat(ask_data)
    
    # Вычисляем BPS для Asks
    asks_df = asks_df.with_columns([
        ((pl.col("price") - pl.col("mid_price")) / pl.col("mid_price") * 10000).alias("bps")
    ])
    
    # Фильтруем по глубине
    asks_df = asks_df.filter((pl.col("bps") >= 0) & (pl.col("bps") <= depth_bps))
    
    # Аналогично для Bids
    bid_data = []
    for i in range(LOB_DEPTH):
        bid_level = (
            df.select([
                "timestamp",
                "mid_price",
                pl.col(f"bid_p_{i}").alias("price"),
                pl.col(f"bid_v_{i}").alias("volume")
            ])
            .filter(pl.col("volume") > 0)  # Только ненулевые объемы
        )
        bid_data.append(bid_level)
    
    # Объединяем все уровни Bids
    bids_df = pl.concat(bid_data)
    
    # Вычисляем BPS для Bids
    bids_df = bids_df.with_columns([
        ((pl.col("price") - pl.col("mid_price")) / pl.col("mid_price") * 10000).alias("bps")
    ])
    
    # Фильтруем по глубине
    bids_df = bids_df.filter((pl.col("bps") >= -depth_bps) & (pl.col("bps") <= 0))
    
    # Объединяем Asks и Bids
    combined_df = pl.concat([
        asks_df.select(["timestamp", "bps", "volume"]),
        bids_df.select(["timestamp", "bps", "volume"])
    ])
    
    # Конвертируем в numpy массивы
    timestamps = combined_df["timestamp"].to_numpy()
    bps_values = combined_df["bps"].to_numpy()
    volumes = combined_df["volume"].to_numpy()
    
    print(f"Обработано {len(timestamps)} точек стакана")
    
    return timestamps, bps_values, volumes


def create_heatmap_matrix(
    timestamps: np.ndarray,
    bps_values: np.ndarray,
    volumes: np.ndarray,
    start_time: datetime,
    end_time: datetime,
    time_bin_seconds: float,
    bps_bin_size: float,
    depth_bps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Создание 2D гистограммы (матрицы плотности) через np.histogram2d.
    
    Args:
        timestamps: Массив временных меток (unix timestamp)
        bps_values: Массив значений BPS
        volumes: Массив объемов
        start_time: Начало временного диапазона
        end_time: Конец временного диапазона
        time_bin_seconds: Размер временного бина в секундах
        bps_bin_size: Размер бина по BPS
        depth_bps: Максимальная глубина в BPS
        
    Returns:
        Кортеж (matrix, time_edges, bps_edges)
    """
    # Определение границ бинов
    start_ts = start_time.timestamp()
    end_ts = end_time.timestamp()
    
    time_bins = np.arange(start_ts, end_ts + time_bin_seconds, time_bin_seconds)
    bps_bins = np.arange(-depth_bps, depth_bps + bps_bin_size, bps_bin_size)
    
    # Создание 2D гистограммы с весами (логарифм объема)
    matrix, time_edges, bps_edges = np.histogram2d(
        timestamps,
        bps_values,
        bins=[time_bins, bps_bins],
        weights=np.log1p(volumes)
    )
    
    print(f"Создана матрица плотности: {matrix.shape}")
    
    return matrix.T, time_edges, bps_edges  # Транспонируем для правильной ориентации


def plot_heatmap_plotly(
    matrix: np.ndarray,
    time_edges: np.ndarray,
    bps_edges: np.ndarray,
    trades_df: Optional[pl.DataFrame],
    orderbook_df: pl.DataFrame,
    symbol: str,
    output_path: Path
):
    """
    Создание интерактивной тепловой карты через Plotly.
    
    Args:
        matrix: Матрица плотности
        time_edges: Границы временных бинов
        bps_edges: Границы BPS бинов
        trades_df: DataFrame с сделками (опционально)
        orderbook_df: DataFrame с данными стакана (для вычисления BPS сделок)
        symbol: Символ токена
        output_path: Путь для сохранения HTML
    """
    # Преобразование временных меток в datetime для оси X
    time_labels = [datetime.fromtimestamp(ts).strftime("%H:%M:%S") for ts in time_edges[:-1]]
    
    # Создание тепловой карты
    fig = go.Figure()
    
    heatmap = go.Heatmap(
        z=matrix,
        x=time_labels,
        y=bps_edges[:-1],
        colorscale="Viridis",
        colorbar=dict(title="log1p(Volume)"),
        hovertemplate="Time: %{x}<br>BPS: %{y:.2f}<br>Density: %{z:.2f}<extra></extra>"
    )
    
    fig.add_trace(heatmap)
    
    # Добавление overlay маркеров сделок с использованием join_asof
    if trades_df is not None and len(trades_df) > 0:
        # Подготавливаем данные для join_asof
        orderbook_sorted = (
            orderbook_df
            .select(["timestamp", "mid_price"])
            .sort("timestamp")
        )
        
        trades_sorted = (
            trades_df
            .with_columns([
                pl.col("time").dt.timestamp().alias("timestamp")
            ])
            .sort("timestamp")
        )
        
        # join_asof для поиска ближайшего mid_price для каждой сделки
        trades_with_mid = trades_sorted.join_asof(
            orderbook_sorted,
            on="timestamp",
            strategy="backward"
        )
        
        # Вычисляем BPS для каждой сделки
        trades_with_mid = trades_with_mid.with_columns([
            ((pl.col("price") - pl.col("mid_price")) / pl.col("mid_price") * 10000).alias("bps")
        ])
        
        # Разделяем на Buy и Sell
        buy_trades = trades_with_mid.filter(pl.col("side") == "Buy")
        sell_trades = trades_with_mid.filter(pl.col("side") == "Sell")
        
        if len(buy_trades) > 0:
            buy_times = [datetime.fromtimestamp(t).strftime("%H:%M:%S") 
                        for t in buy_trades["timestamp"].to_list()]
            buy_bps = buy_trades["bps"].to_list()
            
            fig.add_trace(go.Scatter(
                x=buy_times,
                y=buy_bps,
                mode="markers",
                marker=dict(color="lime", size=10, symbol="triangle-up", 
                           line=dict(color="darkgreen", width=1)),
                name="Buy",
                hovertemplate="Buy<br>Time: %{x}<br>BPS: %{y:.2f}<extra></extra>"
            ))
        
        if len(sell_trades) > 0:
            sell_times = [datetime.fromtimestamp(t).strftime("%H:%M:%S") 
                         for t in sell_trades["timestamp"].to_list()]
            sell_bps = sell_trades["bps"].to_list()
            
            fig.add_trace(go.Scatter(
                x=sell_times,
                y=sell_bps,
                mode="markers",
                marker=dict(color="red", size=10, symbol="triangle-down", 
                           line=dict(color="darkred", width=1)),
                name="Sell",
                hovertemplate="Sell<br>Time: %{x}<br>BPS: %{y:.2f}<extra></extra>"
            ))
    
    # Настройка layout
    fig.update_layout(
        title=f"Orderbook Depth Heatmap - {symbol}",
        xaxis_title="Time",
        yaxis_title="Signed BPS (Ask > 0, Bid < 0)",
        height=800,
        hovermode="closest"
    )
    
    # Сохранение в HTML
    fig.write_html(str(output_path))
    print(f"Интерактивная тепловая карта сохранена: {output_path}")


def plot_heatmap_matplotlib(
    matrix: np.ndarray,
    time_edges: np.ndarray,
    bps_edges: np.ndarray,
    trades_df: Optional[pl.DataFrame],
    orderbook_df: pl.DataFrame,
    symbol: str,
    output_path: Path
):
    """
    Создание статической тепловой карты через Matplotlib.
    
    Args:
        matrix: Матрица плотности
        time_edges: Границы временных бинов
        bps_edges: Границы BPS бинов
        trades_df: DataFrame с сделками (опционально)
        orderbook_df: DataFrame с данными стакана (для вычисления BPS сделок)
        symbol: Символ токена
        output_path: Путь для сохранения PNG
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Создание тепловой карты
    mesh = ax.pcolormesh(
        time_edges,
        bps_edges,
        matrix,
        cmap="inferno",
        shading="auto",
        norm=LogNorm(vmin=matrix[matrix > 0].min() if matrix.max() > 0 else 1, vmax=matrix.max())
    )
    
    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("log1p(Volume)", rotation=270, labelpad=20)
    
    # Добавление overlay маркеров сделок
    if trades_df is not None and len(trades_df) > 0:
        # Подготавливаем данные для join_asof
        orderbook_sorted = (
            orderbook_df
            .select(["timestamp", "mid_price"])
            .sort("timestamp")
        )
        
        trades_sorted = (
            trades_df
            .with_columns([
                pl.col("time").dt.timestamp().alias("timestamp")
            ])
            .sort("timestamp")
        )
        
        # join_asof для поиска ближайшего mid_price для каждой сделки
        trades_with_mid = trades_sorted.join_asof(
            orderbook_sorted,
            on="timestamp",
            strategy="backward"
        )
        
        # Вычисляем BPS для каждой сделки
        trades_with_mid = trades_with_mid.with_columns([
            ((pl.col("price") - pl.col("mid_price")) / pl.col("mid_price") * 10000).alias("bps")
        ])
        
        # Разделяем на Buy и Sell
        buy_trades = trades_with_mid.filter(pl.col("side") == "Buy")
        sell_trades = trades_with_mid.filter(pl.col("side") == "Sell")
        
        if len(buy_trades) > 0:
            buy_times = buy_trades["timestamp"].to_numpy()
            buy_bps = buy_trades["bps"].to_numpy()
            ax.scatter(buy_times, buy_bps, color="lime", marker="^", s=100, 
                      label="Buy", zorder=5, edgecolors="darkgreen")
        
        if len(sell_trades) > 0:
            sell_times = sell_trades["timestamp"].to_numpy()
            sell_bps = sell_trades["bps"].to_numpy()
            ax.scatter(sell_times, sell_bps, color="red", marker="v", s=100, 
                      label="Sell", zorder=5, edgecolors="darkred")
        
        ax.legend()
    
    # Настройка осей
    ax.set_xlabel("Time")
    ax.set_ylabel("Signed BPS (Ask > 0, Bid < 0)")
    ax.set_title(f"Orderbook Depth Heatmap - {symbol}")
    ax.axhline(y=0, color="white", linestyle="--", linewidth=1, alpha=0.5)
    
    # Форматирование временной оси
    time_labels = [datetime.fromtimestamp(ts).strftime("%H:%M:%S") 
                   for ts in time_edges[::max(1, len(time_edges)//10)]]
    ax.set_xticks(time_edges[::max(1, len(time_edges)//10)])
    ax.set_xticklabels(time_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    print(f"Статическая тепловая карта сохранена: {output_path}")
    plt.close()


def export_matrix_to_csv(
    matrix: np.ndarray,
    time_edges: np.ndarray,
    bps_edges: np.ndarray,
    output_path: Path
):
    """
    Экспорт матрицы плотности в CSV.
    
    Args:
        matrix: Матрица плотности
        time_edges: Границы временных бинов
        bps_edges: Границы BPS бинов
        output_path: Путь для сохранения CSV
    """
    # Создание DataFrame с временными метками как индексом
    time_labels = [datetime.fromtimestamp(ts).isoformat() for ts in time_edges[:-1]]
    bps_labels = [f"{bps:.2f}" for bps in bps_edges[:-1]]
    
    df = pl.DataFrame(matrix.T, schema=bps_labels)
    df = df.with_columns(pl.Series("timestamp", time_labels))
    df = df.select(["timestamp"] + bps_labels)
    
    df.write_csv(str(output_path))
    print(f"Матрица плотности экспортирована в CSV: {output_path}")


def main():
    """Основная функция."""
    args = parse_args()
    
    # Парсинг временных диапазонов
    try:
        start_time = datetime.fromisoformat(args.start)
        end_time = datetime.fromisoformat(args.end)
    except ValueError as e:
        print(f"Ошибка парсинга времени: {e}")
        print("Используйте формат ISO: YYYY-MM-DDTHH:MM:SS")
        sys.exit(1)
    
    print(f"Анализ {args.symbol} с {start_time} по {end_time}")
    print(f"Глубина: ±{args.depth_bps} BPS")
    
    # Загрузка данных
    try:
        orderbook_df = load_orderbook_data(args.symbol, start_time, end_time)
        trades_df = load_trades_data(args.symbol, start_time, end_time)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        sys.exit(1)
    
    # Преобразование в Signed BPS (векторизованное)
    print("Преобразование в Signed BPS координаты...")
    timestamps, bps_values, volumes = compute_signed_bps_vectorized(orderbook_df, args.depth_bps)
    
    if len(timestamps) == 0:
        print("Ошибка: нет данных после преобразования")
        sys.exit(1)
    
    # Создание матрицы плотности
    print("Создание 2D гистограммы...")
    matrix, time_edges, bps_edges = create_heatmap_matrix(
        timestamps,
        bps_values,
        volumes,
        start_time,
        end_time,
        args.time_bin_seconds,
        args.bps_bin_size,
        args.depth_bps
    )
    
    # Создание директории для отчетов
    reports_dir = Path(f"./bots/{args.symbol}/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Генерация имени файла с датой
    date_str = start_time.strftime("%Y%m%d")
    
    # Визуализация
    if args.output_format in ["html", "both"]:
        output_html = reports_dir / f"heatmap_{date_str}.html"
        plot_heatmap_plotly(matrix, time_edges, bps_edges, trades_df, orderbook_df, args.symbol, output_html)
    
    if args.output_format in ["png", "both"]:
        output_png = reports_dir / f"heatmap_{date_str}.png"
        plot_heatmap_matplotlib(matrix, time_edges, bps_edges, trades_df, orderbook_df, args.symbol, output_png)
    
    # Экспорт CSV
    if args.export_csv:
        output_csv = reports_dir / f"heatmap_matrix_{date_str}.csv"
        export_matrix_to_csv(matrix, time_edges, bps_edges, output_csv)
    
    print("\n✓ Генерация тепловой карты завершена успешно")


if __name__ == "__main__":
    main()
