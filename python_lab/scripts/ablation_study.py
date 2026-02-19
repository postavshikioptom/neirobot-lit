#!/usr/bin/env python3
"""
Задача 239: Автоматизированные исследования абляции (Automated Ablation Studies)

Скрипт для систематического тестирования влияния отдельных компонентов
(признаков, слоев, голов внимания) на качество модели.

Использование:
    python scripts/ablation_study.py --config ablation_config.yaml --data_path bots/BTCUSDT/data --symbol BTCUSDT
"""

import sys
import time
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Добавляем путь к модулям проекта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import LOBPyTorchDataset, LOBDataset
from train import LiTModule
import pytorch_lightning as ptl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MatthewsCorrCoef

# Optuna для pruning
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# SHAP для feature importance
import shap


def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def expand_feature_group(group_name: str, features: List[str]) -> List[str]:
    """
    Расширяет группу признаков, генерируя полный список.
    Например, для lob_depth_deep генерирует уровни 10-49.
    """
    if group_name == "lob_depth_deep":
        # Генерируем уровни 10-49 для ask/bid price/volume
        expanded = []
        for i in range(10, 50):
            expanded.extend([
                f"feat_ask_p_{i}",
                f"feat_ask_v_{i}",
                f"feat_bid_p_{i}",
                f"feat_bid_v_{i}"
            ])
        return expanded
    else:
        return features


def load_data(data_path: str, symbol: str, val_split: float = 0.8) -> tuple:
    """
    Загружает данные и разделяет на train/val.
    
    Returns:
        (train_df, val_df)
    """
    data_path = Path(data_path)
    pattern = f"{symbol}_*.parquet"
    files = list(data_path.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No parquet files found for {symbol} in {data_path}")
    
    print(f"[Data] Loading {len(files)} parquet files...")
    df = pl.read_parquet(data_path / pattern)
    
    # Разделяем на train/val по времени
    split_idx = int(len(df) * val_split)
    train_df = df.slice(0, split_idx)
    val_df = df.slice(split_idx)
    
    print(f"[Data] Train: {len(train_df)} samples, Val: {len(val_df)} samples")
    return train_df, val_df


def run_baseline(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    n_heads: int = None,
    n_layers: int = None,
    d_model: int = None
) -> Dict[str, float]:
    """
    Обучает базовую модель без исключения признаков.
    
    Args:
        n_heads: количество голов внимания (если None, берется максимум из конфига)
        n_layers: количество слоев (если None, берется максимум из конфига)
        d_model: размерность модели (если None, берется максимум из конфига)
    
    Returns:
        dict с метриками: {'mcc': float, 'latency_ms': float, 'val_loss': float}
    """
    print("\n" + "="*80)
    print("[Baseline] Training baseline model...")
    print("="*80)
    
    # Используем параметры из конфига если не переданы
    if n_heads is None:
        n_heads = config['arch_variants']['heads'][-1]
    if n_layers is None:
        n_layers = config['arch_variants']['layers'][-1]
    if d_model is None:
        d_model = config['arch_variants']['d_model'][-1]
    
    print(f"[Baseline] Architecture: heads={n_heads}, layers={n_layers}, d_model={d_model}")
    
    # Создаем датасеты
    train_ds = LOBPyTorchDataset(
        train_df,
        seq_len=100,
        n_past_returns=3,
        data_mode="memory",
        is_train=True
    )
    
    val_ds = LOBPyTorchDataset(
        val_df,
        seq_len=100,
        n_past_returns=3,
        data_mode="memory",
        is_train=False
    )
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Создаем модель
    model = LiTModule(
        seq_len=100,
        lr=config['training']['lr'],
        n_heads=n_heads,
        n_layers=n_layers,
        d_model=d_model
    )
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_mcc',
        patience=config['training']['early_stopping_patience'],
        mode='max',
        verbose=True
    )
    
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "baseline",
        filename='best',
        monitor='val_mcc',
        mode='max',
        save_top_k=1
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="baseline"
    )
    
    # Trainer
    trainer = ptl.Trainer(
        max_epochs=config['training']['epochs'],
        callbacks=[early_stop, checkpoint],
        logger=logger,
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        log_every_n_steps=50
    )
    
    # Обучение
    trainer.fit(model, train_loader, val_loader)
    
    # Валидация
    val_results = trainer.validate(model, val_loader)[0]
    
    # Измеряем latency
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Берем один батч для измерения latency
    sample_batch = next(iter(val_loader))
    x, _, _, _, regime_id = sample_batch
    x = x.to(model.device)
    regime_id = regime_id.to(model.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, regime_id)
    
    # Измеряем
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x, regime_id)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    latency_ms = (end - start) / 100 * 1000
    
    results = {
        'mcc': val_results['val_mcc'],
        'latency_ms': latency_ms,
        'val_loss': val_results['val_loss']
    }
    
    print(f"\n[Baseline] Results:")
    print(f"  MCC: {results['mcc']:.4f}")
    print(f"  Latency: {results['latency_ms']:.2f} ms")
    print(f"  Val Loss: {results['val_loss']:.4f}")
    
    return results


def run_shap_analysis(
    model: LiTModule,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Выполняет SHAP анализ для быстрой предварительной оценки важности признаков.
    
    Returns:
        DataFrame с колонками: feature, shap_value
    """
    print("\n[SHAP] Running SHAP analysis...")
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Берем n_samples для SHAP
    n_samples = config['shap']['n_samples']
    samples = []
    labels = []
    
    for batch in val_loader:
        x, y, _, _, regime_id = batch
        samples.append(x)
        labels.append(y)
        if sum(len(s) for s in samples) >= n_samples:
            break
    
    samples = torch.cat(samples, dim=0)[:n_samples].to(device)
    labels = torch.cat(labels, dim=0)[:n_samples]
    
    # Создаем GradientExplainer
    explainer = shap.GradientExplainer(model, samples)
    
    # Вычисляем SHAP values
    shap_values = explainer.shap_values(samples)
    
    # Агрегируем по признакам (усредняем по времени и сэмплам)
    # shap_values имеет форму (n_samples, seq_len, n_channels, n_levels)
    # Нужно свести к (n_features,)
    
    # Упрощенная агрегация: берем среднее абсолютное значение
    shap_importance = np.abs(shap_values).mean(axis=(0, 1, 3))  # (n_channels,)
    
    # Создаем DataFrame
    results = pd.DataFrame({
        'feature': feature_names[:len(shap_importance)],
        'shap_value': shap_importance
    })
    
    results = results.sort_values('shap_value', ascending=False)
    
    print(f"[SHAP] Top 10 important features:")
    print(results.head(10))
    
    return results


def run_ablation_experiment(
    group_name: str,
    exclude_features: List[str],
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    trial: Optional[optuna.Trial] = None,
    n_heads: int = None,
    n_layers: int = None,
    d_model: int = None
) -> Dict[str, float]:
    """
    Запускает один эксперимент абляции с исключением заданной группы признаков.
    
    Args:
        group_name: название группы признаков
        exclude_features: список признаков для исключения
        train_df: тренировочные данные
        val_df: валидационные данные
        config: конфигурация
        output_dir: директория для сохранения результатов
        trial: Optuna trial для pruning (опционально)
        n_heads: количество голов внимания (если None, берется из конфига)
        n_layers: количество слоев (если None, берется из конфига)
        d_model: размерность модели (если None, берется из конфига)
    
    Returns:
        dict с метриками: {'mcc': float, 'latency_ms': float, 'val_loss': float}
    """
    print(f"\n[Ablation] Testing group: {group_name}")
    print(f"[Ablation] Excluding {len(exclude_features)} features")
    
    # Используем параметры из конфига если не переданы
    if n_heads is None:
        n_heads = config['arch_variants']['heads'][-1]
    if n_layers is None:
        n_layers = config['arch_variants']['layers'][-1]
    if d_model is None:
        d_model = config['arch_variants']['d_model'][-1]
    
    # Создаем датасеты с исключением признаков
    train_ds = LOBPyTorchDataset(
        train_df,
        seq_len=100,
        n_past_returns=3,
        data_mode="memory",
        is_train=True,
        exclude_features=exclude_features  # Задача 239: Исключаем признаки
    )
    
    val_ds = LOBPyTorchDataset(
        val_df,
        seq_len=100,
        n_past_returns=3,
        data_mode="memory",
        is_train=False,
        exclude_features=exclude_features  # Задача 239: Исключаем признаки
    )
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Создаем модель
    model = LiTModule(
        seq_len=100,
        lr=config['training']['lr'],
        n_heads=n_heads,
        n_layers=n_layers,
        d_model=d_model
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_mcc',
            patience=config['training']['early_stopping_patience'],
            mode='max',
            verbose=False
        )
    ]
    
    # Добавляем Optuna Pruning если используется
    if trial is not None and config['training']['use_pruning']:
        callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor='val_mcc')
        )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name=f"ablation_{group_name}"
    )
    
    # Trainer
    trainer = ptl.Trainer(
        max_epochs=config['training']['epochs'],
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=False  # Отключаем для чистого вывода
    )
    
    # Обучение
    try:
        trainer.fit(model, train_loader, val_loader)
    except optuna.TrialPruned:
        print(f"[Ablation] Trial pruned for group: {group_name}")
        return {'mcc': 0.0, 'latency_ms': 0.0, 'val_loss': float('inf')}
    
    # Валидация
    val_results = trainer.validate(model, val_loader)[0]
    
    # Измеряем latency
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    sample_batch = next(iter(val_loader))
    x, _, _, _, regime_id = sample_batch
    x = x.to(model.device)
    regime_id = regime_id.to(model.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, regime_id)
    
    # Измеряем
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x, regime_id)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    latency_ms = (end - start) / 100 * 1000
    
    results = {
        'mcc': val_results['val_mcc'],
        'latency_ms': latency_ms,
        'val_loss': val_results['val_loss']
    }
    
    print(f"[Ablation] {group_name} - MCC: {results['mcc']:.4f}, Latency: {results['latency_ms']:.2f} ms")
    
    return results


def generate_report(
    baseline_results: Dict[str, float],
    feature_ablation_results: List[Dict[str, Any]],
    arch_ablation_results: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_path: Path
):
    """
    Генерирует markdown отчет с результатами абляции признаков и архитектуры.
    """
    print(f"\n[Report] Generating report at {output_path}")
    
    # Создаем директорию если не существует
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Вычисляем дельты относительно baseline для признаков
    for result in feature_ablation_results:
        result['delta_mcc'] = result['mcc'] - baseline_results['mcc']
        result['delta_latency'] = result['latency_ms'] - baseline_results['latency_ms']
    
    # Вычисляем дельты для архитектуры
    for result in arch_ablation_results:
        result['delta_mcc'] = result['mcc'] - baseline_results['mcc']
        result['delta_latency'] = result['latency_ms'] - baseline_results['latency_ms']
    
    # Сортируем по delta_mcc (по убыванию важности)
    feature_results_sorted = sorted(feature_ablation_results, key=lambda x: abs(x['delta_mcc']), reverse=True)
    arch_results_sorted = sorted(arch_ablation_results, key=lambda x: x['mcc'], reverse=True)
    
    # Генерируем markdown
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Ablation Study Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        f.write("## Baseline Results\n\n")
        f.write(f"- **MCC**: {baseline_results['mcc']:.4f}\n")
        f.write(f"- **Latency**: {baseline_results['latency_ms']:.2f} ms\n")
        f.write(f"- **Val Loss**: {baseline_results['val_loss']:.4f}\n\n")
        
        # Feature Ablation Results
        f.write("## Feature Ablation Results\n\n")
        f.write("Results sorted by impact on MCC (|ΔMCC|):\n\n")
        
        # Таблица результатов признаков
        f.write("| Feature Group | MCC | ΔMCC | Latency (ms) | ΔLatency (ms) | Val Loss |\n")
        f.write("|-------|-----|------|--------------|---------------|----------|\n")
        
        for result in feature_results_sorted:
            f.write(f"| {result['group']} | {result['mcc']:.4f} | ")
            f.write(f"{result['delta_mcc']:+.4f} | {result['latency_ms']:.2f} | ")
            f.write(f"{result['delta_latency']:+.2f} | {result['val_loss']:.4f} |\n")
        
        # Architecture Ablation Results
        f.write("\n## Architecture Ablation Results\n\n")
        f.write("Sorted by MCC (best first):\n\n")
        
        # Таблица результатов архитектуры
        f.write("| Config | Heads | Layers | D_Model | MCC | ΔMCC | Latency (ms) | ΔLatency (ms) |\n")
        f.write("|--------|-------|--------|---------|-----|------|--------------|---------------|\n")
        
        for result in arch_results_sorted:
            f.write(f"| {result['group']} | {result['heads']} | {result['layers']} | {result['d_model']} | ")
            f.write(f"{result['mcc']:.4f} | {result['delta_mcc']:+.4f} | {result['latency_ms']:.2f} | ")
            f.write(f"{result['delta_latency']:+.2f} |\n")
        
        # Dead Weight Analysis
        f.write("\n## Dead Weight Analysis (Features)\n\n")
        f.write(f"Features with |ΔMCC| < {config['report']['dead_weight_threshold']} are candidates for removal:\n\n")
        
        dead_weight = [r for r in feature_results_sorted if abs(r['delta_mcc']) < config['report']['dead_weight_threshold']]
        
        if dead_weight:
            for result in dead_weight:
                f.write(f"- **{result['group']}**: ΔMCC = {result['delta_mcc']:+.4f}\n")
        else:
            f.write("No dead weight features found.\n")
        
        # Architecture Recommendations
        f.write("\n## Architecture Recommendations\n\n")
        
        # Находим лучшую архитектуру по MCC
        best_arch = arch_results_sorted[0]
        f.write(f"**Best Architecture**: {best_arch['group']}\n")
        f.write(f"- MCC: {best_arch['mcc']:.4f} (ΔMCC: {best_arch['delta_mcc']:+.4f})\n")
        f.write(f"- Latency: {best_arch['latency_ms']:.2f} ms (ΔLatency: {best_arch['delta_latency']:+.2f} ms)\n\n")
        
        # Находим архитектуру с лучшей latency
        best_latency_arch = min(arch_results_sorted, key=lambda x: x['latency_ms'])
        f.write(f"**Fastest Architecture**: {best_latency_arch['group']}\n")
        f.write(f"- Latency: {best_latency_arch['latency_ms']:.2f} ms\n")
        f.write(f"- MCC: {best_latency_arch['mcc']:.4f}\n\n")
        
        # Находим оптимальный компромисс (MCC vs Latency)
        f.write("### Pareto Frontier (MCC vs Latency)\n\n")
        f.write("Architectures that are not dominated by others:\n\n")
        
        # Простой Pareto анализ
        pareto_archs = []
        for arch in arch_results_sorted:
            is_dominated = False
            for other in arch_results_sorted:
                if (other['mcc'] >= arch['mcc'] and other['latency_ms'] <= arch['latency_ms'] and
                    (other['mcc'] > arch['mcc'] or other['latency_ms'] < arch['latency_ms'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_archs.append(arch)
        
        for arch in pareto_archs:
            f.write(f"- {arch['group']}: MCC={arch['mcc']:.4f}, Latency={arch['latency_ms']:.2f}ms\n")
        
        # Feature Recommendations
        f.write("\n## Feature Recommendations\n\n")
        
        # Находим самые важные группы
        top_important = feature_results_sorted[:3]
        f.write("### Most Important Feature Groups\n\n")
        for i, result in enumerate(top_important, 1):
            f.write(f"{i}. **{result['group']}**: Removing this group decreases MCC by {abs(result['delta_mcc']):.4f}\n")
        
        # Находим кандидатов на удаление
        if dead_weight:
            f.write("\n### Candidates for Removal\n\n")
            f.write("The following feature groups have minimal impact on model performance:\n\n")
            for result in dead_weight:
                f.write(f"- {result['group']}\n")
    
    print(f"[Report] Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Automated Ablation Studies")
    parser.add_argument('--config', type=str, default='python_lab/ablation_config.yaml',
                        help='Path to ablation config YAML')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Trading symbol')
    parser.add_argument('--output_dir', type=str, default='python_lab/ablation_results',
                        help='Output directory for results')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline training (use cached results)')
    parser.add_argument('--skip_shap', action='store_true',
                        help='Skip SHAP analysis')
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Automated Ablation Studies - Task 239")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Data: {args.data_path}/{args.symbol}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Загружаем данные
    train_df, val_df = load_data(args.data_path, args.symbol)
    
    # 1. Baseline
    if not args.skip_baseline:
        baseline_results = run_baseline(train_df, val_df, config, output_dir)
    else:
        # Загружаем из кэша (если есть)
        baseline_path = output_dir / "baseline_results.yaml"
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_results = yaml.safe_load(f)
            print(f"[Baseline] Loaded cached results: MCC={baseline_results['mcc']:.4f}")
        else:
            print("[Error] No cached baseline results found. Run without --skip_baseline first.")
            return
    
    # Сохраняем baseline results
    with open(output_dir / "baseline_results.yaml", 'w') as f:
        yaml.dump(baseline_results, f)
    
    # 2. SHAP Analysis (опционально)
    if not args.skip_shap:
        print("\n" + "="*80)
        print("[SHAP] Running SHAP analysis...")
        print("="*80)
        
        # Загружаем лучшую baseline модель
        best_model_path = output_dir / "baseline" / "best.ckpt"
        if best_model_path.exists():
            try:
                model = LiTModule.load_from_checkpoint(str(best_model_path))
                
                # Создаем val_loader для SHAP
                val_ds = LOBPyTorchDataset(
                    val_df,
                    seq_len=100,
                    n_past_returns=3,
                    data_mode="memory",
                    is_train=False
                )
                
                val_loader = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=config['training']['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Получаем имена признаков
                feature_names = [c for c in train_df.columns if c.startswith("feat_")]
                
                # Запускаем SHAP анализ
                shap_results = run_shap_analysis(model, val_loader, config, feature_names)
                shap_results.to_csv(output_dir / "shap_importance.csv", index=False)
                print(f"[SHAP] Results saved to {output_dir / 'shap_importance.csv'}")
            except Exception as e:
                print(f"[SHAP] Error loading model: {e}")
        else:
            print(f"[SHAP] Baseline checkpoint not found at {best_model_path}")
    
    # 3. Architecture Ablation (Задача 239: Тестирование вариантов архитектуры)
    print("\n" + "="*80)
    print("[Ablation] Testing architecture variants...")
    print("="*80)
    
    arch_results = []
    heads_list = config['arch_variants']['heads']
    layers_list = config['arch_variants']['layers']
    d_model_list = config['arch_variants']['d_model']
    
    for heads in tqdm(heads_list, desc="Heads"):
        for layers in layers_list:
            for d_model in d_model_list:
                group_name = f"arch_h{heads}_l{layers}_d{d_model}"
                
                # Запускаем эксперимент с архитектурными параметрами
                results = run_ablation_experiment(
                    group_name=group_name,
                    exclude_features=[],  # Не исключаем признаки
                    train_df=train_df,
                    val_df=val_df,
                    config=config,
                    output_dir=output_dir,
                    n_heads=heads,
                    n_layers=layers,
                    d_model=d_model
                )
                
                arch_results.append({
                    'group': group_name,
                    'heads': heads,
                    'layers': layers,
                    'd_model': d_model,
                    **results
                })
    
    # 4. Feature Ablation (Задача 239: Тестирование групп признаков)
    print("\n" + "="*80)
    print("[Ablation] Running feature ablation experiments...")
    print("="*80)
    
    feature_ablation_results = []
    feature_groups = config['feature_groups']
    
    for group_name, features in tqdm(feature_groups.items(), desc="Features"):
        # Расширяем группу признаков
        expanded_features = expand_feature_group(group_name, features)
        
        # Запускаем эксперимент
        results = run_ablation_experiment(
            group_name=group_name,
            exclude_features=expanded_features,
            train_df=train_df,
            val_df=val_df,
            config=config,
            output_dir=output_dir
        )
        
        feature_ablation_results.append({
            'group': group_name,
            'n_features': len(expanded_features),
            **results
        })
    
    # 5. Генерируем отчет
    report_path = Path(config['report']['output_path'])
    generate_report(baseline_results, feature_ablation_results, arch_results, config, report_path)
    
    print("\n" + "="*80)
    print("[Done] Ablation study completed!")
    print(f"[Done] Report: {report_path}")
    print("="*80)


if __name__ == "__main__":
    main()
