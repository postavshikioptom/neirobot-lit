"""
train_module.py — Training core: TrainSubset, ProfilerCallback, compute_hft_metrics, LiTModule.
Вынесено из train.py в рамках задачи 322.1.
"""
import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from torch.profiler import profile, ProfilerActivity, schedule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
)

from .lit_model import LiTModel
from .utils import (
    compute_metrics,
    FocalLoss,
    CalibrationMetrics,
    plot_reliability_diagram,
)


class TrainSubset(torch.utils.data.Subset):
    """
    Subset для тренировочных данных с безопасной аугментацией.
    Включает is_train только во время вызова __getitem__, не затрагивая val/test.
    """
    def __getitem__(self, idx):
        original_is_train = self.dataset.is_train
        self.dataset.is_train = True
        try:
            result = super().__getitem__(idx)
        finally:
            self.dataset.is_train = original_is_train
        return result


class ProfilerCallback(pl.Callback):
    """
    PyTorch Profiler Callback для анализа производительности (Задача 312).
    Профилирует CPU и CUDA операции для выявления узких мест.
    """
    def __init__(self, wait_steps=1, warmup_steps=1, active_steps=3, profiler_dir="profiler_logs"):
        super().__init__()
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.profiler_dir = profiler_dir
        self.profiler = None
        self.step_count = 0

        os.makedirs(profiler_dir, exist_ok=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Запускаем профилер на нужном шаге"""
        if self.step_count == self.wait_steps:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=1
                ),
                on_trace_ready=lambda p: p.export_chrome_trace(
                    os.path.join(self.profiler_dir, f"trace_epoch_{trainer.current_epoch}.json")
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.__enter__()
            print(f"\n[PROFILER] Started profiling at step {self.step_count}")

        if self.profiler is not None:
            self.profiler.step()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Завершаем профилер после нужного количества шагов"""
        self.step_count += 1

        if self.profiler is not None:
            if self.step_count > self.wait_steps + self.warmup_steps + self.active_steps:
                self.profiler.__exit__(None, None, None)
                self.profiler = None
                print(f"\n[PROFILER] Profiling completed. Results saved to {self.profiler_dir}")


def compute_hft_metrics(all_preds, all_labels, all_logits, all_f_rets, all_imbalances):
    """
    Вычисляет расширенные HFT-метрики для анализа качества сигналов (Задача 313.4/5).
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    total = len(all_preds)
    if total == 0:
        return {}

    # 1. Распределение предсказаний
    dist = {
        "dist_flat": np.sum(all_preds == 0) / total * 100,
        "dist_up": np.sum(all_preds == 1) / total * 100,
        "dist_down": np.sum(all_preds == 2) / total * 100
    }

    # 2. Точность по классам
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    hit_rates = {
        "hit_rate_flat": cm[0, 0] / np.sum(cm[:, 0]) if np.sum(cm[:, 0]) > 0 else 0,
        "hit_rate_up": cm[1, 1] / np.sum(cm[:, 1]) if np.sum(cm[:, 1]) > 0 else 0,
        "hit_rate_down": cm[2, 2] / np.sum(cm[:, 2]) if np.sum(cm[:, 2]) > 0 else 0
    }

    # 3. Детали матрицы ошибок
    error_details = {
        "missed_up": cm[1, 0] / np.sum(cm[1, :]) * 100 if np.sum(cm[1, :]) > 0 else 0,
        "missed_down": cm[2, 0] / np.sum(cm[2, :]) * 100 if np.sum(cm[2, :]) > 0 else 0,
        "false_up": cm[0, 1] / np.sum(cm[0, :]) * 100 if np.sum(cm[0, :]) > 0 else 0,
        "false_down": cm[0, 2] / np.sum(cm[0, :]) * 100 if np.sum(cm[0, :]) > 0 else 0
    }

    # 4. Theoretical Edge
    edge_up = np.mean(all_f_rets[all_preds == 1]) if np.sum(all_preds == 1) > 0 else 0
    edge_down = np.mean(all_f_rets[all_preds == 2]) if np.sum(all_preds == 2) > 0 else 0

    # 5. Анализ уверенности (Signal Confidence)
    from torch.nn.functional import softmax
    import torch
    probs = softmax(torch.from_numpy(all_logits), dim=1).numpy()
    correct_mask = (all_labels == all_preds)
    wrong_mask = (all_labels != all_preds)

    conf_correct = np.mean(np.max(probs[correct_mask], axis=1)) if np.sum(correct_mask) > 0 else 0
    conf_wrong = np.mean(np.max(probs[wrong_mask], axis=1)) if np.sum(wrong_mask) > 0 else 0

    # 6. Directional Accuracy (DA)
    dir_mask = (all_preds != 0)
    da = np.mean(all_labels[dir_mask] == all_preds[dir_mask]) if np.sum(dir_mask) > 0 else 0

    # 7. Imbalance Alignment
    sig_num = np.zeros_like(all_preds)
    sig_num[all_preds == 1] = 1
    sig_num[all_preds == 2] = -1

    if np.std(sig_num) > 0 and np.std(all_imbalances) > 0:
        imb_corr = np.corrcoef(sig_num, all_imbalances)[0, 1]
    else:
        imb_corr = 0.0

    return {
        **dist, **hit_rates, **error_details,
        "edge_up": edge_up, "edge_down": edge_down,
        "conf_correct": conf_correct, "conf_wrong": conf_wrong,
        "conf_gap": conf_correct - conf_wrong,
        "da": da, "imb_corr": imb_corr
    }



class LiTModule(pl.LightningModule):
    """
    LightningModule для обучения модели LiT.
    Обертка над nn.Module, добавляющая логику обучения, валидации и оптимизации.
    """
    def __init__(self, seq_len=100, lr=1e-4, class_weights=None, label_smoothing=0.0, loss_type="ce", focal_gamma=2.0, activation='gelu_exact', use_time_weighting=False, teacher_model=None, alpha=0.9, temperature=3.0, use_regime_weighting=False, regime_weights=None, num_horizons=1, horizon_weights=None, use_horizon_embedding=False, use_curvature_reg=False, curvature_lambda=1e-4, input_noise_std=0.005, scaler_type="robust", winsor_limits=None, past_returns_lags=None, scheduler=None, div_factor=None, final_div_factor=None, pct_start=None, plateau_factor=None, plateau_patience=None, step_size=None, gamma=None, weight_decay=None, clip_mode=None, clip_val=None, tb_hist_freq=None, tb_embedding_samples=None, use_gradient_checkpointing=False, **model_params):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "teacher_model", "regime_weights", "horizon_weights"])
        self.model = LiTModel(seq_len=seq_len, activation=activation, num_horizons=num_horizons, use_horizon_embedding=use_horizon_embedding, use_gradient_checkpointing=use_gradient_checkpointing, **model_params)
        self.use_time_weighting = use_time_weighting
        self.use_regime_weighting = use_regime_weighting
        self.teacher_model = teacher_model
        self.is_distillation = teacher_model is not None
        self.num_horizons = num_horizons
        self.is_multi_horizon = (num_horizons > 1)

        # Параметры Curvature Regularization (Задача 238)
        self.use_curvature_reg = use_curvature_reg
        self.curvature_lambda = curvature_lambda
        self.input_noise_std = input_noise_std

        # Логика взаимного исключения: Focal Loss и Label Smoothing - альтернативы
        effective_label_smoothing = 0.0 if loss_type == "focal" else label_smoothing

        if self.is_distillation:
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)
            from .utils import DistillationLoss
            self.distillation_criterion = DistillationLoss(
                alpha=alpha,
                temperature=temperature,
                reduction='none' if (use_time_weighting or use_regime_weighting) else 'batchmean',
                label_smoothing=effective_label_smoothing,
                horizon_weights=horizon_weights if self.is_multi_horizon else None
            )

        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if regime_weights is not None and not isinstance(regime_weights, torch.Tensor):
            regime_weights = torch.tensor(regime_weights, dtype=torch.float32)
        self.regime_weights = regime_weights

        if not self.is_distillation:
            if self.is_multi_horizon:
                from .utils import MultiHorizonLoss
                self.criterion = MultiHorizonLoss(
                    num_horizons=num_horizons,
                    horizon_weights=horizon_weights,
                    class_weights=class_weights,
                    label_smoothing=effective_label_smoothing,
                    reduction='mean'
                )
            else:
                if use_time_weighting or use_regime_weighting:
                    if loss_type == "focal":
                        self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, label_smoothing=effective_label_smoothing, reduction='none')
                    else:
                        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=effective_label_smoothing, reduction='none')
                else:
                    if loss_type == "focal":
                        self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, label_smoothing=effective_label_smoothing)
                    else:
                        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=effective_label_smoothing)

        # Лосс для волатильности (Задача 319: SmoothL1)
        self.vol_criterion = nn.SmoothL1Loss(reduction='none' if (use_time_weighting or use_regime_weighting) else 'mean')
        self.vol_mae = nn.L1Loss()
        self.vol_clamp_val = 10.0
        self.max_grad_warn_prints = 30
        self._grad_warn_prints = 0
        self.max_vol_diag_prints = 20
        self._vol_diag_prints = 0

        # Обучаемые веса для Multi-Task Loss (Uncertainty Weighting)
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_vol = nn.Parameter(torch.zeros(1))

        # Метрики для мониторинга (3 класса: 0=Flat, 1=Up, 2=Down)
        self.acc = MulticlassAccuracy(num_classes=3)
        self.mcc = MulticlassMatthewsCorrCoef(num_classes=3)
        self.f1_macro = MulticlassF1Score(num_classes=3, average="macro")
        self.precision_per_class = MulticlassPrecision(num_classes=3, average=None)
        self.recall_per_class = MulticlassRecall(num_classes=3, average=None)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=3)

        # Списки для накопления результатов валидации
        self.val_y_true = []
        self.val_y_pred = []
        self.val_logits = []
        self.val_vol_true = []
        self.val_vol_pred = []
        self.val_f_ret = []
        self.val_imbalance = []
        self.val_regime_ids = []

        self.calibration_metrics = CalibrationMetrics(n_bins=15)

    def forward(self, x, regime_id=None):
        return self.model(x, regime_id=regime_id)

    def on_train_epoch_start(self):
        self._grad_warn_prints = 0
        self._vol_diag_prints = 0
        self.epoch_start_time = time.time()
        print(f"\n[TRAIN] Epoch {self.current_epoch} started")
        tb_hist_freq = self.hparams.get("tb_hist_freq", 10)
        if self.logger and hasattr(self.logger, 'experiment'):
            from .utils import setup_activation_hooks
            writer = self.logger.experiment
            self.activation_hooks = setup_activation_hooks(
                self.model, writer, self.current_epoch, hist_freq=tb_hist_freq
            )

    def on_train_epoch_end(self):
        if hasattr(self, 'activation_hooks'):
            for handle in self.activation_hooks:
                handle.remove()
            delattr(self, 'activation_hooks')
        epoch_time = time.time() - self.epoch_start_time if hasattr(self, 'epoch_start_time') else 0
        epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s"  # noqa: F841

    def on_before_optimizer_step(self, optimizer):
        if hasattr(self, 'log_var_cls'):
            self.log_var_cls.data = torch.clamp(self.log_var_cls.data, -5.0, 5.0)
        if hasattr(self, 'log_var_vol'):
            self.log_var_vol.data = torch.clamp(self.log_var_vol.data, -5.0, 5.0)

        grad_check_interval = self.hparams.get("grad_finite_check_interval", 0)
        if grad_check_interval > 0 and self.global_step % grad_check_interval == 0:
            warn_count = 0
            max_warns = 5
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        bad_count = (~torch.isfinite(param.grad)).sum().item()
                        if self._grad_warn_prints < self.max_grad_warn_prints:
                            print(f"[WARN] Non-finite gradient in {name} (count={bad_count}), sanitize with nan_to_num + clamp")
                            self._grad_warn_prints += 1
                        param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                        param.grad.data = torch.clamp(param.grad.data, -1.0, 1.0)
                        warn_count += 1
            if warn_count > max_warns and self._grad_warn_prints < self.max_grad_warn_prints:
                print(f"[WARN] Total {warn_count} parameters with non-finite gradients. Warnings suppressed for the rest.")

    def log_channel_statistics(self, batch, batch_idx):
        if batch_idx != 0:
            return
        x, y, ts, mid, label, extra_data = batch
        channel_names = ["MicropriceDev", "Vol", "Imb", "OFI", "VIB", "Ret_10", "Ret_50", "Ret_100", "Spread",
                         "DeltaImb", "DeltaSpread"]
        print("\n[ДИАГНОСТИКА] Статистика каналов ПОСЛЕ нормализации:")
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_idx < x.shape[2]:
                ch_data = x[:, :, ch_idx, :]
                print(f"  Channel {ch_idx} ({ch_name}): "
                      f"min={ch_data.min():.4f}, max={ch_data.max():.4f}, "
                      f"mean={ch_data.mean():.4f}, std={ch_data.std():.4f}")
        if self.logger and hasattr(self.logger, 'experiment'):
            writer = self.logger.experiment
            for ch_idx, ch_name in enumerate(channel_names):
                if ch_idx < x.shape[2]:
                    ch_data = x[:, :, ch_idx, :]
                    writer.add_scalar(f"Stats_train/{ch_name}_Mean", ch_data.mean(), self.global_step)
                    writer.add_scalar(f"Stats_train/{ch_name}_Std", ch_data.std(), self.global_step)

    def training_step(self, batch, batch_idx):
        x, y, ts, mid, label, extra_data = batch

        if y.dtype not in [torch.long, torch.int64]:
            print(f"[WARN] Labels dtype is {y.dtype}, converting to long")
            y = y.long()
        if (y < 0).any() or (y > 2).any():
            invalid_min, invalid_max = y.min().item(), y.max().item()
            print(f"[ERROR] Labels out of range [0,2]: min={invalid_min}, max={invalid_max}. Clamping.")
            y = torch.clamp(y, 0, 2)

        vol_target = extra_data["vol"]
        weights = extra_data["weight"]
        regime_id = extra_data["regime_id"]

        train_batch_log_interval = self.hparams.get("train_batch_log_interval", 0)
        if train_batch_log_interval > 0 and batch_idx % train_batch_log_interval == 0:
            unique, counts = torch.unique(y, return_counts=True)
            print(f"[BATCH {batch_idx}] Label counts: {dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))}")
            print(f"[DIAG] Batch stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}")
            print(f"[DIAG] Contains NaN: {torch.isnan(x).any()}, Contains Inf: {torch.isinf(x).any()}")

        if self.current_epoch == 0 and batch_idx == 0:
            self.log_channel_statistics(batch, batch_idx)

        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            print(f"[WARN] NaN/Inf in input at batch {batch_idx} (nan={nan_count}, inf={inf_count}). Clipping to [-10, 10]")
            x = torch.clamp(x, -10.0, 10.0)
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

        if self.input_noise_std > 0:
            from .lit_model import apply_input_noise
            x = apply_input_noise(x, std=self.input_noise_std)

        logits, vol_pred = self(x, regime_id=regime_id)
        vol_pred = vol_pred.squeeze(-1)

        if self.hparams.get("enable_vol_debug", False) and batch_idx <= 300 and batch_idx % 50 == 0 and self._vol_diag_prints < self.max_vol_diag_prints:
            vol_pred_min = float(torch.nan_to_num(vol_pred, nan=0.0, posinf=0.0, neginf=0.0).min().detach().cpu())
            vol_pred_max = float(torch.nan_to_num(vol_pred, nan=0.0, posinf=0.0, neginf=0.0).max().detach().cpu())
            vol_t_min = float(torch.nan_to_num(vol_target, nan=0.0, posinf=0.0, neginf=0.0).min().detach().cpu())
            vol_t_max = float(torch.nan_to_num(vol_target, nan=0.0, posinf=0.0, neginf=0.0).max().detach().cpu())
            print(f"[VOL_DIAG][BATCH {batch_idx}] "
                  f"finite(pred)={torch.isfinite(vol_pred).all().item()} finite(target)={torch.isfinite(vol_target).all().item()} "
                  f"pred[min,max]=({vol_pred_min:.4f},{vol_pred_max:.4f}) "
                  f"target[min,max]=({vol_t_min:.4f},{vol_t_max:.4f})")
            self._vol_diag_prints += 1

        if not torch.isfinite(vol_pred).all() or not torch.isfinite(vol_target).all():
            if self._vol_diag_prints < self.max_vol_diag_prints:
                print(f"[WARN] Non-finite vol tensors at batch {batch_idx}, applying nan_to_num")
                self._vol_diag_prints += 1
            vol_pred = torch.nan_to_num(vol_pred, nan=0.0, posinf=self.vol_clamp_val, neginf=-self.vol_clamp_val)
            vol_target = torch.nan_to_num(vol_target, nan=0.0, posinf=self.vol_clamp_val, neginf=-self.vol_clamp_val)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            print(f"[WARN] NaN/Inf in logits at step {self.global_step}: nan={nan_count}, inf={inf_count}. Skipping batch.")
            self.zero_grad()
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 1. Лосс классификации
        if self.is_distillation:
            with torch.no_grad():
                teacher_logits, _ = self.teacher_model(x, regime_id=regime_id)
            loss_cls_raw = self.distillation_criterion(logits, teacher_logits, y)
            if self.use_time_weighting or self.use_regime_weighting:
                combined_weights = weights
                if self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
                    regime_w = self.regime_weights[regime_id].to(weights.device)
                    combined_weights = combined_weights * regime_w
                if self.is_multi_horizon and loss_cls_raw.dim() > 1:
                    loss_cls = (loss_cls_raw * combined_weights.unsqueeze(-1)).mean()
                else:
                    loss_cls = (loss_cls_raw * combined_weights).mean()
            else:
                loss_cls = loss_cls_raw.mean() if self.is_multi_horizon else loss_cls_raw
        else:
            if self.is_multi_horizon:
                combined_weights = weights if (self.use_time_weighting or self.use_regime_weighting) else None
                if combined_weights is not None and self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
                    regime_w = self.regime_weights[regime_id].to(weights.device)
                    combined_weights = combined_weights * regime_w
                loss_cls = self.criterion(logits, y, sample_weights=combined_weights)
            else:
                loss_cls_raw = self.criterion(logits, y)
                if self.use_time_weighting or self.use_regime_weighting:
                    combined_weights = weights
                    if self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
                        regime_w = self.regime_weights[regime_id].to(weights.device)
                        combined_weights = combined_weights * regime_w
                    loss_cls = (loss_cls_raw * combined_weights).mean()
                else:
                    loss_cls = loss_cls_raw

        # 2. Лосс волатильности (Задача 319)
        vol_pred_loss = torch.clamp(vol_pred.float(), -self.vol_clamp_val, self.vol_clamp_val)
        vol_target_loss = torch.clamp(vol_target.float(), -self.vol_clamp_val, self.vol_clamp_val)
        loss_vol_raw = self.vol_criterion(vol_pred_loss, vol_target_loss)

        if self.hparams.get("enable_vol_debug", False) and batch_idx <= 300 and batch_idx % 50 == 0 and self._vol_diag_prints < self.max_vol_diag_prints:
            lv = torch.nan_to_num(loss_vol_raw, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"[VOL_LOSS_DIAG][BATCH {batch_idx}] loss_vol_raw[min,max]=({float(lv.min().detach().cpu()):.6f},{float(lv.max().detach().cpu()):.6f})")
            self._vol_diag_prints += 1

        if self.use_time_weighting or self.use_regime_weighting:
            combined_weights = weights
            if self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
                regime_w = self.regime_weights[regime_id].to(weights.device)
                combined_weights = combined_weights * regime_w
            loss_vol = (loss_vol_raw * combined_weights).mean()
        else:
            loss_vol = loss_vol_raw

        # Задача 238: Curvature Regularization
        if self.use_curvature_reg:
            from .lit_model import compute_curvature_penalty
            reg_loss = compute_curvature_penalty(self.model, x, logits, lambda_=self.curvature_lambda, regime_id=regime_id)
            self.log("train_loss_reg", reg_loss, on_step=False, on_epoch=True)
        else:
            reg_loss = 0.0

        # 3. Комбинированный Multi-Task Loss (Задача 304/319)
        self.log_var_cls.data = torch.clamp(self.log_var_cls.data, -5.0, 5.0)
        self.log_var_vol.data = torch.clamp(self.log_var_vol.data, -5.0, 5.0)
        precision_cls = torch.exp(-self.log_var_cls)
        precision_vol = torch.exp(-self.log_var_vol)
        loss = precision_cls * loss_cls + 0.5 * self.log_var_cls + \
               precision_vol * loss_vol + 0.5 * self.log_var_vol + reg_loss

        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at step {self.global_step}: {loss.item()}. Zeroing grads and skipping.")
            self.zero_grad()
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train_loss_vol", loss_vol, on_step=False, on_epoch=True)
        self.log("weight_cls", precision_cls, on_step=False, on_epoch=True)
        self.log("weight_vol", precision_vol, on_step=False, on_epoch=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        current_momentum = self.optimizers().param_groups[0].get('momentum', 0.0)
        self.log("momentum", current_momentum, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_start_time = time.time()
        is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
        phase = 'SANITY' if is_sanity else 'VAL'
        print(f"\n[{phase}] Epoch {self.current_epoch} validation started")

    def validation_step(self, batch, batch_idx):
        is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
        phase = 'SANITY' if is_sanity else 'VAL'
        val_batch_log_interval = self.hparams.get('val_batch_log_interval', 100)
        if batch_idx == 0:
            print(f"[{phase}] first validation batch received")
        elif val_batch_log_interval > 0 and batch_idx % val_batch_log_interval == 0:
            print(f"[{phase}] batch {batch_idx}")

        x, y, ts, mid, label, extra_data = batch
        vol_target = extra_data["vol"]
        regime_id = extra_data["regime_id"]
        f_ret = extra_data["f_ret"]

        logits, vol_pred = self(x, regime_id=regime_id)
        vol_pred = vol_pred.squeeze(-1)

        if self.is_multi_horizon:
            loss_cls = self.criterion(logits, y, sample_weights=None)
        else:
            loss_cls = nn.functional.cross_entropy(logits, y)

        loss_vol = nn.functional.mse_loss(vol_pred, vol_target)
        mae_vol = nn.functional.l1_loss(vol_pred, vol_target)

        if self.is_multi_horizon:
            preds = torch.argmax(logits, dim=2)
        else:
            preds = torch.argmax(logits, dim=1)

        self.val_y_true.append(y.detach().cpu().numpy())
        self.val_y_pred.append(preds.detach().cpu().numpy())
        self.val_logits.append(logits.detach().cpu())
        self.val_vol_true.append(vol_target.detach().cpu().numpy())
        self.val_vol_pred.append(vol_pred.detach().cpu().numpy())
        self.val_f_ret.append(f_ret.detach().cpu().numpy())
        current_imbalance = x[:, -1, 2, 0].detach().cpu().numpy()
        self.val_imbalance.append(current_imbalance)
        if regime_id is not None:
            self.val_regime_ids.append(regime_id.detach().cpu().numpy())

        self.log("val_loss", loss_cls, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mse_vol", loss_vol, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae_vol", mae_vol, on_step=False, on_epoch=True)

        if not self.is_multi_horizon:
            self.log("val_mcc", self.mcc(logits, y), prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_f1_macro", self.f1_macro(logits, y), on_step=False, on_epoch=True)
            self.conf_matrix.update(logits, y)
            self.precision_per_class.update(logits, y)
            self.recall_per_class.update(logits, y)

        return loss_cls + loss_vol

    def on_validation_epoch_end(self):
        import time as _time
        is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
        phase = 'SANITY' if is_sanity else 'VAL'
        print(f"\n[{phase}] Entering on_validation_epoch_end, samples collected: {len(self.val_y_true)}")
        epoch_end_start_time = _time.time()

        if not self.val_y_true:
            print(f"[{phase}] No validation outputs collected; skipping epoch_end")
            return

        if hasattr(self, 'validation_start_time'):
            val_loop_time = _time.time() - self.validation_start_time
            val_loop_str = f"{int(val_loop_time // 60)}m {int(val_loop_time % 60)}s"
            print(f"[{phase}] Validation loop took {val_loop_str}, proceeding with epoch_end processing")

        skip_epoch0_artifacts = self.hparams.get('skip_epoch0_artifacts', True) and self.current_epoch == 0
        skip_heavy_artifacts = is_sanity or skip_epoch0_artifacts

        y_true = np.concatenate(self.val_y_true)
        y_pred = np.concatenate(self.val_y_pred)
        logits = torch.cat(self.val_logits, dim=0)

        if not self.is_multi_horizon:
            f_ret = np.concatenate(self.val_f_ret)
            imbalance = np.concatenate(self.val_imbalance)
            self._log_extended_analytics(y_true, y_pred, logits, f_ret, imbalance)

        if self.is_multi_horizon:
            from .utils import compute_multi_horizon_metrics
            metrics = compute_multi_horizon_metrics(y_true, y_pred, self.num_horizons)
            for name, value in metrics.items():
                self.log(f"val_{name}", value, logger=True)
            epoch_time = _time.time() - self.epoch_start_time if hasattr(self, 'epoch_start_time') else 0
            epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s"
            print(f"\nEpoch {self.current_epoch} ({epoch_time_str}) Multi-Horizon Validation:")
            for h in range(self.num_horizons):
                mcc_h = metrics.get(f"mcc_h{h}", 0.0)
                f1_h = metrics.get(f"f1_h{h}", 0.0)
                samples_h = metrics.get(f"samples_h{h}", 0)
                print(f"  Horizon {h}: MCC={mcc_h:.4f}, F1={f1_h:.4f}, Samples={samples_h}")
            avg_mcc = np.mean([metrics.get(f"mcc_h{h}", 0.0) for h in range(self.num_horizons)])
            self.log("val_mcc", avg_mcc, logger=True, prog_bar=True)
        else:
            class_weights = self.criterion.weight.cpu().numpy() if hasattr(self.criterion, 'weight') and self.criterion.weight is not None else None
            metrics = compute_metrics(y_true, y_pred, class_weights=class_weights)
            y_true_tensor = torch.from_numpy(y_true).long()
            if not torch.isfinite(logits).all():
                print("\n" + "!" * 80)
                print("⚠️  CRITICAL WARNING: NaN or Inf detected in model logits during validation!")
                print("   This indicates extreme numerical instability (exploding gradients).")
                print("   Metrics and visualizations for this epoch will be unreliable.")
                print("!" * 80 + "\n")
            ece, mce, bin_data = self.calibration_metrics.calculate(logits, y_true_tensor)
            for name, value in metrics.items():
                self.log(f"val_{name}", value, logger=True)
            self.log("val_ece", ece, logger=True)
            self.log("val_mce", mce, logger=True)
            epoch_time = _time.time() - self.epoch_start_time if hasattr(self, 'epoch_start_time') else 0
            epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s"
            print(f"\nEpoch {self.current_epoch} ({epoch_time_str}) Validation: MCC={metrics['mcc']:.4f}, "
                  f"Macro-F1={metrics['f1_macro']:.4f}, ECE={ece:.4f}, MCE={mce:.4f}")

            if (self.hparams.get('enable_epoch_end_plots', False) and not skip_heavy_artifacts
                    and self.current_epoch % 20 == 0):
                symbol = getattr(self.trainer, 'symbol', 'UNKNOWN')
                base_path = Path(__file__).parent.parent.parent
                reports_dir = base_path / "reports" / symbol
                reports_dir.mkdir(parents=True, exist_ok=True)
                save_path = reports_dir / f"reliability_diagram_epoch_{self.current_epoch}.png"
                plot_reliability_diagram(bin_data, ece, mce, str(save_path))

            prec = self.precision_per_class.compute()
            rec = self.recall_per_class.compute()
            self.log("val_prec_flat", prec[0])
            self.log("val_rec_flat", rec[0])
            self.log("val_prec_up", prec[1])
            self.log("val_rec_up", rec[1])
            self.log("val_prec_down", prec[2])
            self.log("val_rec_down", rec[2])

            if self.logger and hasattr(self.logger, 'experiment'):
                writer = self.logger.experiment
                if (self.hparams.get('enable_epoch_end_plots', False) and not skip_heavy_artifacts
                        and self.current_epoch % 20 == 0):
                    from .utils import plot_confusion_matrix_tensorboard, plot_pr_curves_tensorboard
                    class_names = ["Flat", "Up", "Down"]
                    plot_confusion_matrix_tensorboard(y_true, y_pred, class_names, writer, self.current_epoch)
                    y_pred_probs = torch.softmax(logits, dim=1).numpy()
                    plot_pr_curves_tensorboard(y_true, y_pred_probs, class_names, writer, self.current_epoch)

        # 3. Метрики регрессии волатильности
        y_vol_true = np.concatenate(self.val_vol_true)
        y_vol_pred = np.concatenate(self.val_vol_pred)
        vol_mse = np.mean((y_vol_true - y_vol_pred)**2)
        vol_mae = np.mean(np.abs(y_vol_true - y_vol_pred))
        self.log("val_vol_mse", vol_mse, logger=True)
        self.log("val_vol_mae", vol_mae, logger=True)
        print(f"  Vol-MSE={vol_mse:.6f}, Vol-MAE={vol_mae:.6f}")

        # 4. Метрики по режимам (только single horizon)
        if len(self.val_regime_ids) > 0 and not self.is_multi_horizon:
            regime_ids = np.concatenate(self.val_regime_ids)
            unique_regimes = np.unique(regime_ids)
            print("\nMetrics by Market Regime:")
            for regime in unique_regimes:
                mask = regime_ids == regime
                regime_y_true = y_true[mask]
                regime_y_pred = y_pred[mask]
                if len(regime_y_true) > 0:
                    from sklearn.metrics import matthews_corrcoef, f1_score
                    regime_mcc = matthews_corrcoef(regime_y_true, regime_y_pred)
                    regime_f1 = f1_score(regime_y_true, regime_y_pred, average='macro', zero_division=0)
                    self.log(f"val_mcc_regime_{regime}", regime_mcc, logger=True)
                    self.log(f"val_f1_regime_{regime}", regime_f1, logger=True)
                    print(f"  Regime {regime}: MCC={regime_mcc:.4f}, F1={regime_f1:.4f}, Samples={len(regime_y_true)}")

        # 5. TensorBoard визуализация
        if self.logger and hasattr(self.logger, 'experiment'):
            writer = self.logger.experiment
            if not is_sanity:
                from .utils import log_gradient_norms
                log_gradient_norms(self.model, writer, self.current_epoch)
            if (self.hparams.get('enable_tb_embeddings', False) and not skip_heavy_artifacts
                    and self.current_epoch % 30 == 0):
                from .utils import log_embeddings
                tb_embedding_samples = self.hparams.get("tb_embedding_samples", 1000)
                val_dataloaders = self.trainer.val_dataloaders
                val_dataloader = val_dataloaders[0] if isinstance(val_dataloaders, (list, tuple)) else val_dataloaders
                if val_dataloader is not None:
                    log_embeddings(self.model, val_dataloader, writer, self.current_epoch, max_samples=tb_embedding_samples)

        # 7. Сброс накопленных данных и метрик
        self.val_y_true.clear()
        self.val_y_pred.clear()
        self.val_logits.clear()
        self.val_vol_true.clear()
        self.val_vol_pred.clear()
        self.val_f_ret.clear()
        self.val_imbalance.clear()
        self.val_regime_ids.clear()
        self.acc.reset()
        self.f1_macro.reset()
        self.mcc.reset()
        self.precision_per_class.reset()
        self.recall_per_class.reset()
        self.conf_matrix.reset()

        epoch_end_duration = _time.time() - epoch_end_start_time
        epoch_end_str = f"{int(epoch_end_duration // 60)}m {int(epoch_end_duration % 60)}s"
        print(f"\n[{phase}] on_validation_epoch_end completed in {epoch_end_str}")

    def _log_extended_analytics(self, y_true, y_pred, logits, f_ret, imbalance):
        hft_metrics = compute_hft_metrics(y_pred, y_true, logits.numpy(), f_ret, imbalance)
        if not hft_metrics:
            return
        print("\n" + "="*85)
        print(f"{'LOB-SPECIFIC CLASS ANALYTICS (Validation)':^85}")
        print("="*85)
        print(f"{'Metric':<30} | {'Flat (0)':<15} | {'Up (1)':<15} | {'Down (2)':<15}")
        print("-" * 85)
        print(f"{'Процент предсказаний (%)':<30} | {hft_metrics['dist_flat']:<15.2f} | {hft_metrics['dist_up']:<15.2f} | {hft_metrics['dist_down']:<15.2f}")
        print(f"{'Hit Rate (Точность)':<30} | {hft_metrics['hit_rate_flat']:<15.4f} | {hft_metrics['hit_rate_up']:<15.4f} | {hft_metrics['hit_rate_down']:<15.4f}")
        print(f"{'Процент пропущенных сигналов (%)':<30} | {'-':<15} | {hft_metrics['missed_up']:<15.2f} | {hft_metrics['missed_down']:<15.2f}")
        print(f"{'Ложные входы (%)':<30} | {'-':<15} | {hft_metrics['false_up']:<15.2f} | {hft_metrics['false_down']:<15.2f}")
        print("-" * 85)
        print(f"{'Теоретический Edge (Future Ret)':<30} | {'-':<15} | {hft_metrics['edge_up']:<15.6f} | {hft_metrics['edge_down']:<15.6f}")
        print(f"{'Directional Accuracy (DA) без Flat':<30} | {hft_metrics['da']:<15.4f} (Accuracy where pred != Flat)")
        print(f"{'Средняя Уверенность (C/W)':<30} | Уверенность при правильном: {hft_metrics['conf_correct']:.4f} | Уверенности при ложном: {hft_metrics['conf_wrong']:.4f} | Разница уверенности: {hft_metrics['conf_gap']:.4f}")
        print(f"{'Корреляция с LOB Imbalance':<30} | {hft_metrics['imb_corr']:<15.4f} (Corr with Signal -1/0/1)")
        print("="*85 + "\n")
        for name, val in hft_metrics.items():
            self.log(f"class_stats/{name}", val, logger=True)

    def configure_optimizers(self):
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-5)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_type = self.hparams.get("scheduler", "plateau")

        if scheduler_type == "none":
            return optimizer

        elif scheduler_type == "onecycle":
            total_steps = self.trainer.estimated_stepping_batches
            div_factor = self.hparams.get("div_factor", 25.0)
            final_div_factor = self.hparams.get("final_div_factor", 10000.0)
            pct_start = self.hparams.get("pct_start", 0.3)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps, pct_start=pct_start,
                div_factor=div_factor, final_div_factor=final_div_factor,
                anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}}

        elif scheduler_type == "cosine":
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(0.1 * total_steps)
            cosine_steps = total_steps - warmup_steps
            div_factor = self.hparams.get("div_factor", 25.0)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0/div_factor, end_factor=1.0, total_iters=warmup_steps)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=lr / 10000.0)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}

        elif scheduler_type == "step":
            step_size = self.hparams.get("step_size", 10)
            gamma = self.hparams.get("gamma", 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        else:  # "plateau" (default)
            factor = self.hparams.get("plateau_factor", 0.5)
            patience = self.hparams.get("plateau_patience", 2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        clip_mode = self.hparams.get("clip_mode", "none")
        clip_val = self.hparams.get("clip_val", 1.0)

        if clip_mode == "none":
            return
        elif clip_mode == "norm":
            self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")
        elif clip_mode == "agc":
            from .utils import adaptive_gradient_clipping, log_grad_stats
            clip_stats = adaptive_gradient_clipping(self.model, clip_factor=clip_val, eps=1e-6)
            if self.global_step % 100 == 0:
                grad_stats = log_grad_stats(
                    self.model, clip_stats=clip_stats,
                    logger=self.logger.experiment if self.logger else None,
                    global_step=self.global_step
                )
                print(f"\n[Step {self.global_step}] Gradient Stats:")
                print(f"  Clipped: {clip_stats['clipped_pct']:.1f}% ({clip_stats['clipped_count']}/{clip_stats['total_count']})")
                print(f"  Max Ratio (All): {clip_stats['max_ratio']:.4f}")
                print(f"  Max Ratio (Attention): {clip_stats['max_ratio_attention']:.4f}")
                print(f"  Global Grad Norm: {grad_stats['global_grad_norm']:.4f}")

