"""
Задача 237: Оптимизация Multi-Head Attention (Heads/Embedding Tuning)

Этот скрипт использует Optuna для поиска оптимального сочетания embed_dim и num_heads
для модели LiT, балансируя качество предсказаний (MCC) и скорость инференса (latency).

Основные возможности:
1. Байесовская оптимизация с Optuna
2. Пространство поиска: embed_dim [32,64,128,256], num_heads [2,4,8,16]
3. Целевая функция: MCC - lambda * latency_ms
4. Latency constraint: < 2.0ms на CPU через ONNX Runtime
5. Сохранение результатов в best_mha_config.json
6. График Парето (Accuracy vs Latency)

Использование:
    python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 30

Параметры:
    --symbol: Торговый символ (по умолчанию BTCUSDT)
    --trials: Количество trials для Optuna (по умолчанию 30)
    --epochs: Количество эпох обучения для каждого trial (по умолчанию 5)
    --lambda_latency: Коэффициент штрафа за latency (по умолчанию 0.1)
    --max_latency: Максимальная допустимая latency в ms (по умолчанию 2.0)
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import onnxruntime as ort
import numpy as np
import json
import time
import argparse
from pathlib import Path
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI

# Импорты из проекта
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.lit_model import LiTModel
from src.dataset import LOBPyTorchDataset, LOBDataLoader
from src.features import FeatureEngineer
from src.labels import Labeler
from src.normalization import Normalizer

# Глобальные переменные для данных
train_loader = None
val_loader = None
val_dataset = None

def prepare_data(symbol: str, seq_len: int = 100):
    """
    Загружает и подготавливает данные один раз для всех trials.
    
    Args:
        symbol: Торговый символ
        seq_len: Длина последовательности
    
    Returns:
        train_loader, val_loader, val_dataset
    """
    print(f"Загрузка данных для {symbol}...")
    
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "bots" / symbol / "data" / "raw"
    
    # 1. Загрузка сырых данных
    loader = LOBDataLoader(str(data_path), symbol)
    df = loader.load_data()
    
    # 2. Feature Engineering
    fe = FeatureEngineer(n_levels=50)
    df_feat = fe.transform(df)
    
    # 3. Добавление меток
    labeler = Labeler(horizon=100, threshold=0.0005)
    df_feat = labeler.add_labels(df_feat)
    
    # 4. Нормализация
    normalizer = Normalizer("/tmp/tune_attention_norm.json")
    normalizer.fit(df_feat)
    df_norm = normalizer.transform(df_feat)
    
    # 5. Создание Dataset
    past_returns_lags = [10, 50, 100]
    n_past_returns = len(past_returns_lags)
    full_dataset = LOBPyTorchDataset(
        df_norm,
        seq_len=seq_len,
        n_past_returns=n_past_returns,
        is_train=False
    )
    
    # 6. Разделение на train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 7. DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Данные загружены: train={len(train_ds)}, val={len(val_ds)}")
    
    return train_loader, val_loader, val_ds


def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device='cpu'):
    """
    Обучает модель на заданное количество эпох.
    
    Args:
        model: Модель LiT
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
        epochs: Количество эпох
        lr: Learning rate
        device: Устройство (cpu/cuda)
    
    Returns:
        val_mcc: MCC на валидационном наборе
    """
    model = model.to(device)
    model.train()
    
    # Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Обучение
    for epoch in range(epochs):
        train_loss = 0.0
        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, vol = model(x)
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
    
    # Валидация
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, _, _ in val_loader:
            x = x.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    # Вычисление MCC
    val_mcc = matthews_corrcoef(all_labels, all_preds)
    
    return val_mcc


def export_to_onnx(model, output_path, seq_len=100, in_channels=6):
    """
    Экспортирует модель в ONNX формат.
    
    Args:
        model: Модель LiT
        output_path: Путь для сохранения ONNX модели
        seq_len: Длина последовательности
        in_channels: Количество каналов
    """
    model.eval()
    model.cpu()
    
    # Dummy input
    dummy_input = torch.randn(1, seq_len, in_channels, 50)
    
    # Экспорт
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )


def measure_latency(onnx_path, seq_len=100, in_channels=6, n_warmup=10, n_runs=100):
    """
    Измеряет latency модели через ONNX Runtime на CPU.
    
    Args:
        onnx_path: Путь к ONNX модели
        seq_len: Длина последовательности
        in_channels: Количество каналов
        n_warmup: Количество warmup runs
        n_runs: Количество измерений
    
    Returns:
        median_latency_ms: Медианная latency в миллисекундах
    """
    # Создание InferenceSession с CPUExecutionProvider
    session = ort.InferenceSession(
        str(onnx_path),
        providers=['CPUExecutionProvider']
    )
    
    # Подготовка входных данных
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, seq_len, in_channels, 50).astype(np.float32)
    
    # Warmup runs
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy_input})
    
    # Measurement runs
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Конвертируем в ms
    
    # Возвращаем медианную latency (более стабильная метрика)
    median_latency = np.median(latencies)
    
    return median_latency


def objective(trial, args):
    """
    Целевая функция для Optuna.
    
    Задача 237: Пространство поиска ограничено только embed_dim и num_heads
    согласно плану (пункт Б.1).
    
    Args:
        trial: Optuna trial
        args: Аргументы командной строки
    
    Returns:
        score: Комбинированный score (MCC - lambda * latency)
    """
    global train_loader, val_loader, val_dataset
    
    # Задача 237: Строгое ограничение пространства поиска согласно плану
    # Пункт Б.1: только embed_dim и num_heads
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16])
    
    # Валидация: embed_dim должен делиться на num_heads
    if embed_dim % num_heads != 0:
        raise optuna.exceptions.TrialPruned()
    
    # Параметры слоев и дропаута фиксируются (не участвуют в тюнинге по плану 237)
    num_layers = 2
    dropout = 0.1
    
    print(f"\nTrial {trial.number}: embed_dim={embed_dim}, num_heads={num_heads}")
    
    # Создание модели
    model = LiTModel(
        seq_len=100,
        in_channels=6,  # 3 базовых + 3 past returns
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        activation='gelu_exact',
        multi_task=True
    )
    
    # Обучение модели
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_mcc = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=1e-3,
        device=device
    )
    
    print(f"  Validation MCC: {val_mcc:.4f}")
    
    # Экспорт в ONNX
    onnx_path = f"/tmp/tune_attention_trial_{trial.number}.onnx"
    export_to_onnx(model, onnx_path, seq_len=100, in_channels=6)
    
    # Измерение latency
    latency_ms = measure_latency(onnx_path, seq_len=100, in_channels=6)
    
    print(f"  Inference Latency: {latency_ms:.3f} ms")
    
    # Проверка latency constraint
    if latency_ms > args.max_latency:
        print(f"  ⚠️  Latency превышает порог {args.max_latency} ms - trial pruned")
        raise optuna.exceptions.TrialPruned()
    
    # Вычисление комбинированного score
    score = val_mcc - args.lambda_latency * latency_ms
    
    print(f"  Combined Score: {score:.4f}")
    
    # Сохранение метрик в trial
    trial.set_user_attr("val_mcc", val_mcc)
    trial.set_user_attr("latency_ms", latency_ms)
    trial.set_user_attr("score", score)
    
    # Удаление временного ONNX файла
    Path(onnx_path).unlink(missing_ok=True)
    
    return score


def plot_pareto_front(study, output_path):
    """
    Строит график Парето фронта (MCC vs Latency).
    
    Args:
        study: Optuna study
        output_path: Путь для сохранения графика
    """
    # Собираем данные из всех trials
    mccs = []
    latencies = []
    scores = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            mcc = trial.user_attrs.get("val_mcc", None)
            latency = trial.user_attrs.get("latency_ms", None)
            score = trial.user_attrs.get("score", None)
            
            if mcc is not None and latency is not None:
                mccs.append(mcc)
                latencies.append(latency)
                scores.append(score if score is not None else 0)
    
    if len(mccs) == 0:
        print("⚠️  Нет завершенных trials для построения графика")
        return
    
    # Находим Парето-фронт
    # Точка на Парето-фронте если нет другой точки которая лучше по обоим критериям
    pareto_indices = []
    for i in range(len(mccs)):
        is_pareto = True
        for j in range(len(mccs)):
            if i != j:
                # Точка j доминирует точку i если она лучше по обоим критериям
                if mccs[j] >= mccs[i] and latencies[j] <= latencies[i]:
                    if mccs[j] > mccs[i] or latencies[j] < latencies[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    # Все точки
    plt.scatter(latencies, mccs, c=scores, cmap='viridis', alpha=0.6, s=50, label='All trials')
    
    # Парето-фронт
    if len(pareto_indices) > 0:
        pareto_latencies = [latencies[i] for i in pareto_indices]
        pareto_mccs = [mccs[i] for i in pareto_indices]
        
        # Сортируем для линии
        sorted_pairs = sorted(zip(pareto_latencies, pareto_mccs))
        pareto_latencies_sorted = [p[0] for p in sorted_pairs]
        pareto_mccs_sorted = [p[1] for p in sorted_pairs]
        
        plt.plot(pareto_latencies_sorted, pareto_mccs_sorted, 'r--', linewidth=2, label='Pareto front')
        plt.scatter(pareto_latencies, pareto_mccs, c='red', s=100, marker='*', 
                   edgecolors='black', linewidths=1.5, label='Pareto optimal', zorder=5)
    
    # Лучший trial по combined score
    best_idx = scores.index(max(scores))
    plt.scatter([latencies[best_idx]], [mccs[best_idx]], c='gold', s=200, marker='D',
               edgecolors='black', linewidths=2, label='Best combined score', zorder=6)
    
    plt.colorbar(label='Combined Score')
    plt.xlabel('Inference Latency (ms)', fontsize=12)
    plt.ylabel('Validation MCC', fontsize=12)
    plt.title('Multi-Head Attention Optimization: Pareto Front', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохранение
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График Парето сохранен: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Tune Multi-Head Attention parameters")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per trial")
    parser.add_argument("--lambda_latency", type=float, default=0.1, 
                       help="Penalty coefficient for latency")
    parser.add_argument("--max_latency", type=float, default=2.0, 
                       help="Maximum allowed latency in ms")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Задача 237: Оптимизация Multi-Head Attention")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Trials: {args.trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Lambda (latency penalty): {args.lambda_latency}")
    print(f"Max latency constraint: {args.max_latency} ms")
    print("="*60)
    
    # Подготовка данных
    global train_loader, val_loader, val_dataset
    train_loader, val_loader, val_dataset = prepare_data(args.symbol, seq_len=100)
    
    # Создание Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"mha_tuning_{args.symbol}",
        storage="sqlite:///optuna_mha.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=2
        )
    )
    
    # Запуск оптимизации
    print("\nЗапуск оптимизации...")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    
    # Результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("="*60)
    print(f"Best Combined Score: {study.best_value:.4f}")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    best_mcc = study.best_trial.user_attrs.get("val_mcc", 0)
    best_latency = study.best_trial.user_attrs.get("latency_ms", 0)
    print(f"\nBest Trial Metrics:")
    print(f"  Validation MCC: {best_mcc:.4f}")
    print(f"  Inference Latency: {best_latency:.3f} ms")
    print("="*60)
    
    # Сохранение конфигурации
    base_path = Path(__file__).parent.parent.parent
    config_path = base_path / "bots" / args.symbol / "model" / "best_mha_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    best_config = {
        "embed_dim": study.best_params["embed_dim"],
        "num_heads": study.best_params["num_heads"],
        "num_layers": 2,
        "dropout": 0.1,
        "validation_mcc": best_mcc,
        "inference_latency_ms": best_latency,
        "combined_score": study.best_value,
        "lambda_latency": args.lambda_latency,
        "max_latency_constraint": args.max_latency
    }
    
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=4)
    
    print(f"\nКонфигурация сохранена: {config_path}")
    
    # Построение графика Парето
    reports_path = base_path / "reports"
    plot_path = reports_path / "mha_pareto_front.png"
    plot_pareto_front(study, plot_path)
    
    print("\n✓ Оптимизация завершена успешно!")


if __name__ == "__main__":
    main()
