import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from src.train import LiTModule
from src.dataset import LOBDataset, LOBPyTorchDataset
from src.features import FeatureEngineer
from src.labels import Labeler
from src.normalization import Normalizer
from src.utils import calculate_uncertainty
from src.interpretability import run_shap_analysis, prune_features, plot_shap_results
from evaluate_uncertainty import plot_rejection_curve

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate LiT model and perform SHAP analysis")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--mc_passes", type=int, default=20, help="Number of MC Dropout passes")
    parser.add_argument("--shap_samples", type=int, default=50, help="Number of samples for SHAP analysis")
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Загрузка модели
    print(f"Loading model from {args.checkpoint}...")
    model = LiTModule.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()
    
    # Извлекаем параметры из hparams модели
    seq_len = model.hparams.get("seq_len", 100)
    past_returns_lags = model.hparams.get("past_returns_lags", [10, 50, 100])
    horizon = model.hparams.get("horizon", 100)
    threshold = model.hparams.get("threshold", 0.0005)
    
    # 2. Загрузка данных (аналогично train.py для консистентности)
    data_path = base_path / "bots" / args.symbol / "data" / "raw"
    norm_params_path = base_path / "bots" / args.symbol / "models" / "norm_params.json"
    
    loader = LOBDataset(str(data_path), args.symbol)
    df = loader.load_data(lazy=False)
    
    fe = FeatureEngineer(n_levels=50)
    df = fe.transform(df)
    
    labeler = Labeler(horizon=horizon, threshold=threshold)
    df = labeler.add_labels(df)
    
    normalizer = Normalizer(norm_params_path)
    if norm_params_path.exists():
        normalizer.load()
    else:
        normalizer.fit(df)
    df = normalizer.transform(df)
    
    # Создаем тестовый датасет (последние 15%)
    full_dataset = LOBPyTorchDataset(df, seq_len=seq_len, n_past_returns=len(past_returns_lags), is_train=False)
    total_len = len(full_dataset)
    test_start = int(0.85 * total_len)
    test_ds = Subset(full_dataset, range(test_start, total_len))
    train_ds_subset = Subset(full_dataset, range(0, 500)) # Для фоновых данных SHAP
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    train_loader_subset = DataLoader(train_ds_subset, batch_size=500, shuffle=False)
    
    # 3. MC Dropout Uncertainty
    print("\n[1/2] Starting MC Dropout Uncertainty Estimation...")
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()
            
    model.apply(enable_dropout)
    
    # Warm-up
    with torch.no_grad():
        dummy_input = torch.randn(1, seq_len, 3 + len(past_returns_lags), 50).to(device)
        for _ in range(5):
            _ = model(dummy_input)
            
    all_mc_logits = []
    y_true = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for _ in range(args.mc_passes):
                pass_logits = []
                for batch in tqdm(test_loader, desc=f"MC Pass {_ + 1}", leave=False):
                    x, y = batch
                    logits = model(x.to(device))
                    pass_logits.append(logits.cpu())
                    if _ == 0: y_true.append(y.numpy())
                all_mc_logits.append(torch.cat(pass_logits, dim=0))
            
    y_true = np.concatenate(y_true)
    mc_logits_tensor = torch.stack(all_mc_logits)
    mean_probs, entropy, mi = calculate_uncertainty(mc_logits_tensor)
    
    reports_dir = base_path / "reports" / args.symbol
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    plot_rejection_curve(y_true, mean_probs.unsqueeze(0), entropy.numpy(), str(reports_dir / "rejection_curve_entropy.png"))
    plot_rejection_curve(y_true, mean_probs.unsqueeze(0), mi.numpy(), str(reports_dir / "rejection_curve_mi.png"))
    print(f"✓ Uncertainty analysis completed. Reports saved to: {reports_dir}")

    # 4. SHAP Interpretability
    print("\n[2/2] Starting SHAP Interpretability Analysis...")
    model.eval()
    
    # Подготовка данных
    train_x, _ = next(iter(train_loader_subset))
    test_batch = next(iter(test_loader))
    test_x, _ = test_batch
    test_x_subset = test_x[:args.shap_samples]
    
    shap_dir = base_path / "python_lab" / "results" / "interpret" / args.symbol
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    shap_values, feature_names = run_shap_analysis(
        model, 
        train_x.to(device), 
        test_x_subset.to(device), 
        depth=50, 
        seq_len=seq_len,
        past_returns_lags=past_returns_lags,
        save_path=str(shap_dir)
    )
    
    plot_shap_results(shap_values, test_x_subset, feature_names, str(shap_dir))
    
    # Прунинг
    to_keep, to_drop, importance_pct, level_pct = prune_features(shap_values, feature_names, threshold=0.01)
    
    print(f"\n=== SHAP Feature Pruning Report ===")
    print(f"Features recommended to drop (impact < 1%): {len(to_drop)}/{len(importance_pct)}")
    
    if len(level_pct) > 0:
        print("\nImportance by LOB Levels:")
        for level_name, imp in list(level_pct.items())[:10]:
            print(f"  - {level_name}: {imp:.4f}")
            
    if len(to_drop) > 0:
        print("\nTop 10 candidates to drop:")
        # Сортируем drop-список по важности (хотя они и так мелкие)
        drop_sorted = sorted([(n, importance_pct[n]) for n in to_drop], key=lambda x: x[1])
        for name, imp in drop_sorted[:10]:
            print(f"  - {name}: {imp:.4f}")
            
    print(f"\nTop 10 most important features:")
    keep_sorted = sorted([(n, importance_pct[n]) for n in to_keep], key=lambda x: x[1], reverse=True)
    for name, imp in keep_sorted[:10]:
        print(f"  - {name}: {imp:.4f}")

    print(f"\n✓ SHAP analysis completed. Results saved to: {shap_dir}")

if __name__ == "__main__":
    evaluate()
