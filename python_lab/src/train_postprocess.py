"""
train_postprocess.py — Post-training этапы: MC Dropout, pruning, holdout, copy checkpoint.
Вынесено из train.py в рамках задачи 322.9.
"""
import shutil
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, matthews_corrcoef
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .train_module import LiTModule
from .train_runtime import build_dataloader_kwargs, resolve_trainer_precision
from .utils import save_confusion_matrices


def enable_dropout(m):
    """Включает dropout для MC Dropout inference."""
    if type(m).__name__.startswith('Dropout'):
        m.train()


def run_mc_dropout_uncertainty(model, val_loader, checkpoint_dir, in_channels, seq_len, n_mc_passes=20):
    """
    MC Dropout для оценки неопределённости (Задача 125).
    Сохраняет mc_dropout_uncertainty.pt в checkpoint_dir.
    """
    from .utils import calculate_uncertainty

    print(f"\n{'='*60}")
    print("MC DROPOUT UNCERTAINTY ESTIMATION")
    print(f"{'='*60}\n")

    try:
        model.model.apply(enable_dropout)
        model.model.eval()

        print("Warming up model...")
        device = next(model.parameters()).device
        with torch.no_grad():
            dummy_input = torch.randn(1, seq_len, in_channels, 50, device=device)
            for _ in range(5):
                _ = model(dummy_input)
        print("Warm-up completed.\n")

        print(f"Performing {n_mc_passes} MC Dropout passes on validation set...")
        mc_logits_list = []
        val_labels_list = []

        for mc_pass in tqdm(range(n_mc_passes), desc="MC Passes"):
            pass_logits, pass_labels = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Pass {mc_pass+1}/{n_mc_passes}", leave=False):
                    x, y, ts, mid, label, extra_data = batch
                    device = next(model.parameters()).device
                    x = x.to(device)
                    y = y.to(device)
                    r_id = extra_data["regime_id"].to(device) if extra_data["regime_id"] is not None else None
                    logits, _ = model(x, regime_id=r_id)
                    pass_logits.append(logits.cpu())
                    pass_labels.append(y.cpu())

            mc_logits_list.append(torch.cat(pass_logits, dim=0))
            if mc_pass == 0:
                val_labels_list = torch.cat(pass_labels, dim=0)

        mc_logits = torch.stack(mc_logits_list, dim=0)
        mean_probs, entropy, mutual_info = calculate_uncertainty(mc_logits)

        print(f"\n{'='*60}")
        print("Uncertainty Statistics:")
        print(f"  Entropy - Mean: {entropy.mean().item():.4f}, Std: {entropy.std().item():.4f}")
        print(f"  MI      - Mean: {mutual_info.mean().item():.4f}, Std: {mutual_info.std().item():.4f}")
        print(f"{'='*60}\n")

        uncertainty_data = {
            'mc_logits': mc_logits,
            'entropy': entropy,
            'mutual_info': mutual_info,
            'val_labels': val_labels_list,
            'mean_probs': mean_probs,
        }
        uncertainty_path = Path(checkpoint_dir) / "mc_dropout_uncertainty.pt"
        torch.save(uncertainty_data, uncertainty_path)
        print(f"MC Dropout uncertainty data saved to: {uncertainty_path}\n")

    except Exception as e:
        print(f"Warning: MC Dropout uncertainty estimation failed: {str(e)}")
        print("Continuing with model pruning...\n")


def run_model_pruning(model, args, train_loader, val_loader, checkpoint_callback, logger, base_path, symbol):
    """
    Итеративный pruning модели (Задача 159).
    Возвращает обновлённый model (после pruning).
    """
    if args.prune_mode == "none":
        return model

    from .utils import (
        apply_iterative_pruning,
        apply_structured_pruning_2_4,
        remove_pruning_reparametrization,
        calculate_sparsity,
        save_pruned_model,
        log_pruning_progress,
        print_pruning_warning,
    )

    print(f"\n{'='*60}")
    print(f"STARTING MODEL PRUNING")
    print(f"{'='*60}")
    print(f"Mode: {args.prune_mode}, Target Sparsity: {args.prune_amount:.2%}")
    print(f"Iterations: {args.prune_iterations}, Fine-tune Epochs: {args.prune_finetune_epochs}")
    print(f"{'='*60}\n")

    if args.prune_mode == "unstructured":
        print_pruning_warning()

    if checkpoint_callback.best_model_path:
        print(f"Loading best model for pruning: {checkpoint_callback.best_model_path}")
        model_module = LiTModule.load_from_checkpoint(checkpoint_callback.best_model_path, map_location="cpu")
        model = model_module.model

    # Baseline MCC
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing baseline MCC"):
            x, y, ts, mid, label, extra_data = batch
            device = model.device
            x = x.to(device)
            y = y.to(device)
            r_id = extra_data["regime_id"].to(device) if extra_data["regime_id"] is not None else None
            logits_cls, _ = model(x, regime_id=r_id)
            preds = torch.argmax(logits_cls, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    baseline_mcc = matthews_corrcoef(
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
    )
    print(f"\nBaseline MCC (before pruning): {baseline_mcc:.4f}")

    sparsifier = None
    current_mcc = baseline_mcc

    for iteration in range(1, args.prune_iterations + 1):
        current_amount = (iteration / args.prune_iterations) * args.prune_amount

        if args.prune_mode == "unstructured":
            apply_iterative_pruning(model, current_amount, prune_mode='unstructured')
        elif args.prune_mode == "structured_2_4":
            if iteration == 1:
                sparsifier = apply_structured_pruning_2_4(model)
                if sparsifier is None:
                    print("⚠️  Structured 2:4 pruning failed. Skipping pruning.")
                    break

        sparsity_stats = calculate_sparsity(model, detailed=False)

        print(f"\nFine-tuning for {args.prune_finetune_epochs} epochs...")
        trainer_precision = resolve_trainer_precision(args)
        finetune_trainer = pl.Trainer(
            max_epochs=args.prune_finetune_epochs,
            accelerator="auto",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            deterministic=False,
            log_every_n_steps=100,
            accumulate_grad_batches=args.accumulate_grad_batches,
            num_sanity_val_steps=args.num_sanity_val_steps,
            limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else 1.0,
            limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
        )
        model.hparams.val_batch_log_interval = args.val_batch_log_interval
        model.hparams.enable_epoch_end_plots = args.enable_epoch_end_plots
        model.hparams.skip_epoch0_artifacts = args.skip_epoch0_artifacts
        model.hparams.enable_tb_embeddings = args.enable_tb_embeddings
        finetune_trainer.fit(model, train_loader, val_loader)

        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating iteration {iteration}"):
                x, y, ts, mid, label, extra_data = batch
                device = model.device
                x = x.to(device)
                y = y.to(device)
                r_id = extra_data["regime_id"].to(device) if extra_data["regime_id"] is not None else None
                logits_cls, _ = model(x, regime_id=r_id)
                preds = torch.argmax(logits_cls, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        current_mcc = matthews_corrcoef(
            torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy(),
        )
        log_pruning_progress(iteration, args.prune_iterations, current_amount, args.prune_amount,
                             sparsity_stats, current_mcc, baseline_mcc)

    if args.prune_mode == "unstructured":
        remove_pruning_reparametrization(model)
    elif args.prune_mode == "structured_2_4" and sparsifier is not None:
        sparsifier.squash_mask()
        print("✓ Squashed 2:4 sparsity masks")
        from .utils import convert_to_sparse_semi_structured
        convert_to_sparse_semi_structured(model)

    final_sparsity_stats = calculate_sparsity(model, detailed=True)
    print(f"\n{'='*60}")
    print(f"PRUNING COMPLETED")
    print(f"Final Sparsity: {final_sparsity_stats['global_sparsity']:.2%}, Final MCC: {current_mcc:.4f}")
    print(f"MCC Drop: {baseline_mcc - current_mcc:.4f}")
    print(f"{'='*60}\n")

    pruned_model_path = Path(base_path) / "bots" / symbol / "models" / f"pruned_{args.prune_mode}.pt"
    save_pruned_model(model, pruned_model_path, final_sparsity_stats, baseline_mcc)
    checkpoint_callback.best_model_path = str(pruned_model_path)

    return model


def run_holdout_evaluation(best_model_path, test_loader, base_path, symbol):
    """
    Финальная оценка на holdout (test) выборке.
    Возвращает (y_true, y_pred, best_model).
    """
    print("\nStarting final Holdout evaluation...")
    if not best_model_path:
        print("No best model found, skipping evaluation.")
        return None, None, None

    print(f"Loading best model from: {best_model_path}")
    best_model = LiTModule.load_from_checkpoint(best_model_path)
    best_model.eval()
    best_model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for batch in tqdm(test_loader, desc="Testing"):
                x, y, ts, mid, label, extra_data = batch
                regime_id = extra_data["regime_id"]
                x = x.to(device)
                r_id = regime_id.to(device) if regime_id is not None else None
                logits, _ = best_model(x, regime_id=r_id)
                preds = torch.argmax(logits, dim=1)
                y_true.append(y.numpy())
                y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    class_names = ["Flat", "Up", "Down"]
    save_confusion_matrices(y_true, y_pred, class_names, Path(base_path) / "bots" / symbol / "model")

    print("\nHoldout Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return y_true, y_pred, best_model


def compare_teacher_student(args, y_true, y_pred, best_model, test_loader, teacher_model, base_path, symbol):
    """
    Сравнение Teacher vs Student (только для distill режима).
    Сохраняет distillation_metrics.json.
    """
    if args.mode != "distill":
        return

    print("\n" + "=" * 60)
    print("KNOWLEDGE DISTILLATION: Teacher vs Student Comparison")
    print("=" * 60)

    from .utils import measure_latency, count_parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_mcc = matthews_corrcoef(y_true, y_pred)

    teacher_model.to(device)
    teacher_model.eval()
    y_pred_teacher = []

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for batch in tqdm(test_loader, desc="Testing Teacher"):
                x, y, ts, mid, label, extra_data = batch
                regime_id = extra_data["regime_id"]
                r_id = regime_id.to(device) if regime_id is not None else None
                teacher_logits, _ = teacher_model(x.to(device), regime_id=r_id)
                preds = torch.argmax(teacher_logits, dim=1)
                y_pred_teacher.append(preds.cpu().numpy())

    y_pred_teacher = np.concatenate(y_pred_teacher)
    teacher_mcc = matthews_corrcoef(y_true, y_pred_teacher)

    sample_batch = next(iter(test_loader))
    sample_x = sample_batch[0]

    teacher_latency = measure_latency(teacher_model, sample_x, device=str(device), warmup_runs=10, test_runs=100)
    student_latency = measure_latency(best_model.model, sample_x, device=str(device), warmup_runs=10, test_runs=100)

    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(best_model.model)

    speedup = teacher_latency / student_latency
    compression_ratio = teacher_params / student_params
    mcc_retention = (student_mcc / teacher_mcc) * 100 if teacher_mcc != 0 else 0

    print(f"\n{'Metric':<25} {'Teacher':<15} {'Student':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'MCC':<25} {teacher_mcc:<15.4f} {student_mcc:<15.4f} {mcc_retention:<15.2f}%")
    print(f"{'Latency (ms)':<25} {teacher_latency:<15.2f} {student_latency:<15.2f} {speedup:<15.2f}x")
    print(f"{'Parameters':<25} {teacher_params:<15,} {student_params:<15,} {compression_ratio:<15.2f}x")
    print("-" * 70)

    metrics_dict = {
        "teacher": {"mcc": float(teacher_mcc), "latency_ms": float(teacher_latency), "parameters": int(teacher_params)},
        "student": {"mcc": float(student_mcc), "latency_ms": float(student_latency), "parameters": int(student_params)},
        "comparison": {
            "speedup": float(speedup),
            "compression_ratio": float(compression_ratio),
            "mcc_retention_percent": float(mcc_retention),
        },
        "distillation_params": {"alpha": args.alpha, "temperature": args.temperature},
    }

    metrics_path = Path(base_path) / "bots" / symbol / "models" / "distillation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nDistillation metrics saved to: {metrics_path}")
    print("\n" + "=" * 60)


def copy_best_checkpoint_to_target(args, best_model_path, base_path, symbol):
    """
    Копирует лучший checkpoint в teacher_lit.pt или lit.pt (Задача 151).
    """
    if not best_model_path:
        return

    base_path = Path(base_path)

    if args.mode == "distill":
        target_path = base_path / "bots" / symbol / "models" / "lit.pt"
        model_type = "Student"
    else:
        target_path = base_path / "bots" / symbol / "models" / "teacher_lit.pt"
        model_type = "Teacher"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model_path, target_path)

    print(f"\n{'='*60}")
    print(f"✓ {model_type} model automatically saved to:")
    print(f"  {target_path}")
    print(f"  Source: {best_model_path}")

    if args.mode == "train":
        print(f"\nNext step: Use this teacher for distillation:")
        print(f"  python -m python_lab.src.train \\")
        print(f"    --symbol {symbol} \\")
        print(f"    --mode distill \\")
        print(f"    --teacher_path {target_path}")
    else:
        print(f"\nNext step: Export to ONNX:")
        print(f"  python -m python_lab.scripts.export_onnx \\")
        print(f"    --checkpoint {target_path}")

    print(f"{'='*60}\n")
