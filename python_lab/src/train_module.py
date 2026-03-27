"""
train_module.py — Training core: TrainSubset, ProfilerCallback, LiTModule.
Вынесено из train.py в рамках задачи 322.1.
"""
import json
import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from torch.profiler import profile, ProfilerActivity, schedule
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
)

from .lit_model import LiTModel
from .utils import (
    compute_classification_metrics,
    compute_directional_metrics,
    safe_matthews_corrcoef,
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


class LiTModule(pl.LightningModule):
    """
    LightningModule для обучения модели LiT.
    Обертка над nn.Module, добавляющая логику обучения, валидации и оптимизации.
    """
    def __init__(self, seq_len=100, lr=1e-4, class_weights=None, label_smoothing=0.0, loss_type="ce", focal_gamma=2.0, activation='gelu_exact', use_time_weighting=False, teacher_model=None, alpha=0.9, temperature=3.0, use_regime_weighting=False, regime_weights=None, num_horizons=1, horizon_weights=None, use_horizon_embedding=False, use_curvature_reg=False, curvature_lambda=1e-4, input_noise_std=0.005, scaler_type="robust", winsor_limits=None, past_returns_lags=None, scheduler=None, div_factor=None, final_div_factor=None, pct_start=None, plateau_factor=None, plateau_patience=None, step_size=None, gamma=None, weight_decay=None, clip_mode=None, clip_val=None, tb_hist_freq=None, tb_embedding_samples=None, use_gradient_checkpointing=False, metric_contract="standard", metric_log_prefix="val", metric_directional_base="predicted", report_fee_bps=0.0, report_slippage_bps=0.0, report_half_spread_bps=0.0, **model_params):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "teacher_model", "regime_weights", "horizon_weights"])
        class_weight_metadata = model_params.pop("class_weight_metadata", {})
        model_label_columns = model_params.pop("model_label_columns", [])
        label_mode = model_params.pop("label_mode", "legacy_mid_return")
        time_mode = model_params.pop("time_mode", "row")
        if class_weight_metadata:
            if class_weight_metadata.get("label_cols") != model_label_columns:
                raise ValueError(
                    "Class weight label columns do not match model label columns: "
                    f"{class_weight_metadata.get('label_cols')} vs {model_label_columns}"
                )
            if class_weight_metadata.get("label_mode") != label_mode:
                raise ValueError(
                    "Class weight label_mode does not match model label_mode: "
                    f"{class_weight_metadata.get('label_mode')} vs {label_mode}"
                )
            if class_weight_metadata.get("time_mode") != time_mode:
                raise ValueError(
                    "Class weight time_mode does not match model time_mode: "
                    f"{class_weight_metadata.get('time_mode')} vs {time_mode}"
                )
        self.model = LiTModel(seq_len=seq_len, activation=activation, num_horizons=num_horizons, use_horizon_embedding=use_horizon_embedding, use_gradient_checkpointing=use_gradient_checkpointing, **model_params)
        self.use_time_weighting = use_time_weighting
        self.use_regime_weighting = use_regime_weighting
        self.teacher_model = teacher_model
        self.is_distillation = teacher_model is not None
        self.num_horizons = num_horizons
        self.is_multi_horizon = (num_horizons > 1)
        self.label_mode = label_mode
        self.time_mode = time_mode

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
        self.mcc = MulticlassMatthewsCorrCoef(num_classes=3)
        self.f1_macro = MulticlassF1Score(num_classes=3, average="macro")

        # Списки для накопления результатов валидации
        self._validation_accumulator = []

        self.calibration_metrics = CalibrationMetrics(n_bins=15)
        self.class_weight_metadata = class_weight_metadata
        self.model_label_columns = model_label_columns

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
        self._reset_validation_state()
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

        current_imbalance = x[:, -1, 2, 0]
        self._accumulate_validation_outputs(
            logits=logits,
            labels=y,
            f_ret=f_ret,
            imbalance=current_imbalance,
            regime_id=regime_id,
            vol_true=vol_target,
            vol_pred=vol_pred,
            loss_cls=loss_cls,
            loss_vol=loss_vol,
            mae_vol=mae_vol,
        )

        return loss_cls + loss_vol

    def on_validation_epoch_end(self):
        import time as _time
        is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
        phase = 'SANITY' if is_sanity else 'VAL'
        print(f"\n[{phase}] Entering on_validation_epoch_end, samples collected: {len(self._validation_accumulator)}")
        epoch_end_start_time = _time.time()

        if not self._validation_accumulator:
            print(f"[{phase}] No validation outputs collected; skipping epoch_end")
            return

        if hasattr(self, 'validation_start_time'):
            val_loop_time = _time.time() - self.validation_start_time
            val_loop_str = f"{int(val_loop_time // 60)}m {int(val_loop_time % 60)}s"
            print(f"[{phase}] Validation loop took {val_loop_str}, proceeding with epoch_end processing")

        skip_epoch0_artifacts = self.hparams.get('skip_epoch0_artifacts', True) and self.current_epoch == 0
        skip_heavy_artifacts = is_sanity or skip_epoch0_artifacts

        validation_payload = self._gather_validation_payload()
        y_true = validation_payload["y_true"]
        logits = validation_payload["logits"]
        y_pred = validation_payload["y_pred"]
        finalized = None

        if self.is_multi_horizon:
            from .utils import compute_multi_horizon_metrics
            metrics = compute_multi_horizon_metrics(y_true, y_pred, self.num_horizons)
            self.log("val_loss", float(validation_payload["loss_cls"]), logger=True, add_dataloader_idx=False)
            self.log("val_loss_cls", float(validation_payload["loss_cls"]), logger=True, add_dataloader_idx=False)
            self.log("val_mse_vol", float(validation_payload["loss_vol"]), logger=True, add_dataloader_idx=False)
            self.log("val_mae_vol", float(validation_payload["mae_vol"]), logger=True, add_dataloader_idx=False)
            self.log("val_vol_mse", float(np.mean((validation_payload["vol_true"] - validation_payload["vol_pred"])**2)), logger=True, add_dataloader_idx=False)
            self.log("val_vol_mae", float(np.mean(np.abs(validation_payload["vol_true"] - validation_payload["vol_pred"]))), logger=True, add_dataloader_idx=False)
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
            self.log("val_mcc_primary", avg_mcc, logger=True, prog_bar=True, add_dataloader_idx=False)
            finalized = {
                "epoch": int(self.current_epoch),
                "metric_contract": self.hparams.get("metric_contract", "standard"),
                "metric_log_prefix": self.hparams.get("metric_log_prefix", "val"),
                "metric_directional_base": self.hparams.get("metric_directional_base", "predicted"),
                "quality": {
                    "val_loss": float(validation_payload["loss_cls"]),
                    "val_loss_cls": float(validation_payload["loss_cls"]),
                    "val_mse_vol": float(validation_payload["loss_vol"]),
                    "val_mae_vol": float(validation_payload["mae_vol"]),
                    "val_vol_mse": float(np.mean((validation_payload["vol_true"] - validation_payload["vol_pred"])**2)),
                    "val_vol_mae": float(np.mean(np.abs(validation_payload["vol_true"] - validation_payload["vol_pred"]))),
                    "val_mcc_primary": float(avg_mcc),
                    "val_mcc_np": float(avg_mcc),
                },
                "calibration": {},
                "coverage": {},
                "trade": {},
                "class_metrics": {},
                "regime_metrics": {},
            }
            for h in range(self.num_horizons):
                finalized["quality"][f"mcc_h{h}"] = float(metrics.get(f"mcc_h{h}", 0.0))
                finalized["quality"][f"f1_h{h}"] = float(metrics.get(f"f1_h{h}", 0.0))
                finalized["quality"][f"samples_h{h}"] = int(metrics.get(f"samples_h{h}", 0))
        else:
            finalized = self._finalize_validation_metrics(validation_payload)
            y_true_tensor = torch.from_numpy(y_true).long()
            if not torch.isfinite(logits).all():
                print("\n" + "!" * 80)
                print("⚠️  CRITICAL WARNING: NaN or Inf detected in model logits during validation!")
                print("   This indicates extreme numerical instability (exploding gradients).")
                print("   Metrics and visualizations for this epoch will be unreliable.")
                print("!" * 80 + "\n")
            ece = finalized["calibration"]["val_ece"]
            mce = finalized["calibration"]["val_mce"]
            bin_data = finalized["calibration"]["bin_data"]
            self._log_final_validation_metrics(finalized)
            epoch_time = _time.time() - self.epoch_start_time if hasattr(self, 'epoch_start_time') else 0
            epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s"
            print(f"\nEpoch {self.current_epoch} ({epoch_time_str}) Validation: MCC={finalized['quality']['val_mcc_primary']:.4f}, "
                  f"Macro-F1={finalized['quality']['val_f1_macro_np']:.4f}, ECE={ece:.4f}, MCE={mce:.4f}")

            self._log_extended_analytics(finalized["trade"])

            if (self.hparams.get('enable_epoch_end_plots', False) and not skip_heavy_artifacts
                    and self.current_epoch % 20 == 0):
                symbol = getattr(self.trainer, 'symbol', 'UNKNOWN')
                base_path = Path(__file__).parent.parent.parent
                reports_dir = base_path / "reports" / symbol
                reports_dir.mkdir(parents=True, exist_ok=True)
                save_path = reports_dir / f"reliability_diagram_epoch_{self.current_epoch}.png"
                plot_reliability_diagram(bin_data, ece, mce, str(save_path))

            if self.logger and hasattr(self.logger, 'experiment'):
                writer = self.logger.experiment
                if (self.hparams.get('enable_epoch_end_plots', False) and not skip_heavy_artifacts
                        and self.current_epoch % 20 == 0):
                    from .utils import plot_confusion_matrix_tensorboard, plot_pr_curves_tensorboard
                    class_names = ["Flat", "Up", "Down"]
                    plot_confusion_matrix_tensorboard(y_true, y_pred, class_names, writer, self.current_epoch)
                    y_pred_probs = torch.softmax(logits, dim=1).numpy()
                    plot_pr_curves_tensorboard(y_true, y_pred_probs, class_names, writer, self.current_epoch)

        # 4. Метрики по режимам (только single horizon)
        if validation_payload["regime_ids"] is not None and not self.is_multi_horizon:
            regime_ids = validation_payload["regime_ids"]
            unique_regimes = np.unique(regime_ids)
            print("\nMetrics by Market Regime:")
            regime_report = {}
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
                    regime_report[str(int(regime))] = {
                        "val_mcc": float(regime_mcc),
                        "val_f1_macro": float(regime_f1),
                        "samples": int(len(regime_y_true)),
                    }
                    print(f"  Regime {regime}: MCC={regime_mcc:.4f}, F1={regime_f1:.4f}, Samples={len(regime_y_true)}")
            finalized["regime_metrics"] = regime_report

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
        if not is_sanity and not self.is_multi_horizon:
            self._save_validation_report(finalized)
        self._reset_validation_state()
        self.f1_macro.reset()
        self.mcc.reset()

        epoch_end_duration = _time.time() - epoch_end_start_time
        epoch_end_str = f"{int(epoch_end_duration // 60)}m {int(epoch_end_duration % 60)}s"
        print(f"\n[{phase}] on_validation_epoch_end completed in {epoch_end_str}")

    def _accumulate_validation_outputs(self, logits, labels, f_ret, imbalance, regime_id, vol_true, vol_pred, loss_cls, loss_vol, mae_vol):
        self._validation_accumulator.append(
            {
                "logits": logits.detach().cpu(),
                "labels": labels.detach().cpu(),
                "f_ret": f_ret.detach().cpu(),
                "imbalance": imbalance.detach().cpu(),
                "regime_id": regime_id.detach().cpu() if regime_id is not None else None,
                "vol_true": vol_true.detach().cpu(),
                "vol_pred": vol_pred.detach().cpu(),
                "loss_cls": loss_cls.detach().cpu(),
                "loss_vol": loss_vol.detach().cpu(),
                "mae_vol": mae_vol.detach().cpu(),
            }
        )

    def _gather_validation_payload(self):
        payload = {
            "logits": torch.cat([item["logits"] for item in self._validation_accumulator], dim=0),
            "y_true": np.concatenate([item["labels"].numpy() for item in self._validation_accumulator]),
            "f_ret": np.concatenate([item["f_ret"].numpy() for item in self._validation_accumulator]),
            "imbalance": np.concatenate([item["imbalance"].numpy() for item in self._validation_accumulator]),
            "vol_true": np.concatenate([item["vol_true"].numpy() for item in self._validation_accumulator]),
            "vol_pred": np.concatenate([item["vol_pred"].numpy() for item in self._validation_accumulator]),
            "loss_cls": torch.stack([item["loss_cls"] for item in self._validation_accumulator]).mean().item(),
            "loss_vol": torch.stack([item["loss_vol"] for item in self._validation_accumulator]).mean().item(),
            "mae_vol": torch.stack([item["mae_vol"] for item in self._validation_accumulator]).mean().item(),
        }
        payload["y_pred"] = torch.argmax(payload["logits"], dim=2 if self.is_multi_horizon else 1).cpu().numpy()

        regime_values = [item["regime_id"].numpy() for item in self._validation_accumulator if item["regime_id"] is not None]
        payload["regime_ids"] = np.concatenate(regime_values) if regime_values else None
        return payload

    def _finalize_validation_metrics(self, payload):
        class_weights = self.criterion.weight.cpu().numpy() if hasattr(self.criterion, 'weight') and self.criterion.weight is not None else None
        quality_metrics = compute_classification_metrics(payload["y_true"], payload["y_pred"], class_weights=class_weights)
        direction_metrics = compute_directional_metrics(
            payload["y_true"],
            payload["y_pred"],
            payload["logits"].numpy(),
            payload["f_ret"],
            payload["imbalance"],
            directional_base=self.hparams.get("metric_directional_base", "predicted"),
            fee_bps=self.hparams.get("report_fee_bps", 0.0),
            slippage_bps=self.hparams.get("report_slippage_bps", 0.0),
            half_spread_bps=self.hparams.get("report_half_spread_bps", 0.0),
        )

        y_true_tensor = torch.from_numpy(payload["y_true"]).long()
        val_mcc_torch = float(self.mcc(payload["logits"], y_true_tensor).detach().cpu())
        val_f1_torch = float(self.f1_macro(payload["logits"], y_true_tensor).detach().cpu())
        ece, mce, bin_data = self.calibration_metrics.calculate(payload["logits"], y_true_tensor)
        val_direction_mcc = float(direction_metrics.get("direction_mcc", 0.0))

        quality = {
            "val_loss": float(payload["loss_cls"]),
            "val_loss_cls": float(payload["loss_cls"]),
            "val_mse_vol": float(payload["loss_vol"]),
            "val_mae_vol": float(payload["mae_vol"]),
            "val_vol_mse": float(np.mean((payload["vol_true"] - payload["vol_pred"])**2)),
            "val_vol_mae": float(np.mean(np.abs(payload["vol_true"] - payload["vol_pred"]))),
            "val_mcc_primary": float(quality_metrics["mcc"]),
            "val_mcc_np": float(quality_metrics["mcc"]),
            "val_mcc_torch": val_mcc_torch,
            "val_direction_mcc": val_direction_mcc,
            "val_f1_macro_np": float(quality_metrics["f1_macro"]),
            "val_f1_macro_torch": val_f1_torch,
            "val_balanced_acc": float(quality_metrics["balanced_acc"]),
            "val_accuracy": float(quality_metrics["accuracy"]),
        }
        class_metrics = {
            "precision_flat": float(quality_metrics["precision_flat"]),
            "precision_up": float(quality_metrics["precision_up"]),
            "precision_down": float(quality_metrics["precision_down"]),
            "recall_flat": float(quality_metrics["recall_flat"]),
            "recall_up": float(quality_metrics["recall_up"]),
            "recall_down": float(quality_metrics["recall_down"]),
            "f1_flat": float(quality_metrics["f1_flat"]),
            "f1_up": float(quality_metrics["f1_up"]),
            "f1_down": float(quality_metrics["f1_down"]),
            "false_up": float(direction_metrics["false_up"]),
            "false_down": float(direction_metrics["false_down"]),
            "missed_up": float(direction_metrics["missed_up"]),
            "missed_down": float(direction_metrics["missed_down"]),
        }
        calibration = {
            "val_ece": float(ece),
            "val_mce": float(mce),
            "bin_data": bin_data,
        }
        coverage = {
            "coverage_directional": float(direction_metrics["coverage_directional"]),
            "coverage_long": float(direction_metrics["coverage_long"]),
            "coverage_short": float(direction_metrics["coverage_short"]),
        }
        trade = {
            "val_da_without_flat": float(direction_metrics["da_without_flat"]),
            "gross_edge_long": float(direction_metrics["gross_edge_long"]),
            "gross_edge_short": float(direction_metrics["gross_edge_short"]),
            "gross_edge_total": float(direction_metrics["gross_edge_total"]),
            "net_edge_long": float(direction_metrics["net_edge_long"]),
            "net_edge_short": float(direction_metrics["net_edge_short"]),
            "net_edge_total": float(direction_metrics["net_edge_total"]),
            "trade_count_total": int(direction_metrics["trade_count_total"]),
            "trade_count_long": int(direction_metrics["trade_count_long"]),
            "trade_count_short": int(direction_metrics["trade_count_short"]),
            "edge_down_signed": float(direction_metrics["edge_down_signed"]),
            "edge_up": float(direction_metrics["edge_up"]),
            "roundtrip_cost": float(direction_metrics["roundtrip_cost"]),
            "dist_flat": float(direction_metrics["dist_flat"]),
            "dist_up": float(direction_metrics["dist_up"]),
            "dist_down": float(direction_metrics["dist_down"]),
            "precision_flat": float(direction_metrics["precision_flat"]),
            "precision_up": float(direction_metrics["precision_up"]),
            "precision_down": float(direction_metrics["precision_down"]),
            "recall_flat": float(direction_metrics["recall_flat"]),
            "recall_up": float(direction_metrics["recall_up"]),
            "recall_down": float(direction_metrics["recall_down"]),
            "false_up": float(direction_metrics["false_up"]),
            "false_down": float(direction_metrics["false_down"]),
            "missed_up": float(direction_metrics["missed_up"]),
            "missed_down": float(direction_metrics["missed_down"]),
            "coverage_directional": float(direction_metrics["coverage_directional"]),
            "coverage_long": float(direction_metrics["coverage_long"]),
            "coverage_short": float(direction_metrics["coverage_short"]),
            "conf_correct": float(direction_metrics["conf_correct"]),
            "conf_wrong": float(direction_metrics["conf_wrong"]),
            "conf_gap": float(direction_metrics["conf_gap"]),
            "imb_corr": float(direction_metrics["imb_corr"]),
        }
        return {
            "epoch": int(self.current_epoch),
            "metric_contract": self.hparams.get("metric_contract", "standard"),
            "metric_log_prefix": self.hparams.get("metric_log_prefix", "val"),
            "metric_directional_base": self.hparams.get("metric_directional_base", "predicted"),
            "quality": quality,
            "calibration": calibration,
            "coverage": coverage,
            "trade": trade,
            "class_metrics": class_metrics,
            "regime_metrics": {},
        }

    def _log_final_validation_metrics(self, finalized):
        for name, value in finalized["quality"].items():
            self.log(name, float(value), logger=True, prog_bar=(name == "val_mcc_primary"), add_dataloader_idx=False)
        for name, value in finalized["calibration"].items():
            if name != "bin_data":
                self.log(name, float(value), logger=True, add_dataloader_idx=False)
        for section in ("coverage", "trade", "class_metrics"):
            for name, value in finalized[section].items():
                self.log(name, float(value), logger=True, add_dataloader_idx=False)

    def _save_validation_report(self, finalized):
        symbol = getattr(self.trainer, 'symbol', 'UNKNOWN')
        base_path = Path(__file__).parent.parent.parent
        report_dir = base_path / "artifacts" / symbol / "validation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"validation_report_epoch_{self.current_epoch}.json"
        serializable = {
            "epoch": finalized["epoch"],
            "metric_contract": finalized["metric_contract"],
            "metric_log_prefix": finalized["metric_log_prefix"],
            "metric_directional_base": finalized["metric_directional_base"],
            "quality": finalized["quality"],
            "calibration": {k: v for k, v in finalized["calibration"].items() if k != "bin_data"},
            "coverage": finalized["coverage"],
            "trade": finalized["trade"],
            "class_metrics": finalized["class_metrics"],
            "regime_metrics": finalized["regime_metrics"],
        }
        report_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def _reset_validation_state(self):
        self._validation_accumulator.clear()

    def _log_extended_analytics(self, trade_metrics):
        if not trade_metrics:
            return
        print("\n" + "="*85)
        print(f"{'LOB-SPECIFIC CLASS ANALYTICS (Validation)':^85}")
        print("="*85)
        print(f"{'Metric':<30} | {'Flat (0)':<15} | {'Up (1)':<15} | {'Down (2)':<15}")
        print("-" * 85)
        print(f"{'Процент предсказаний (%)':<30} | {trade_metrics['dist_flat']:<15.2f} | {trade_metrics['dist_up']:<15.2f} | {trade_metrics['dist_down']:<15.2f}")
        print(f"{'Precision':<30} | {trade_metrics.get('precision_flat', 0.0):<15.4f} | {trade_metrics.get('precision_up', 0.0):<15.4f} | {trade_metrics.get('precision_down', 0.0):<15.4f}")
        print(f"{'Recall':<30} | {trade_metrics.get('recall_flat', 0.0):<15.4f} | {trade_metrics.get('recall_up', 0.0):<15.4f} | {trade_metrics.get('recall_down', 0.0):<15.4f}")
        print(f"{'Процент пропущенных сигналов (%)':<30} | {'-':<15} | {trade_metrics.get('missed_up', 0.0):<15.2f} | {trade_metrics.get('missed_down', 0.0):<15.2f}")
        print(f"{'Ложные входы (%)':<30} | {'-':<15} | {trade_metrics.get('false_up', 0.0):<15.2f} | {trade_metrics.get('false_down', 0.0):<15.2f}")
        print("-" * 85)
        print(f"{'Gross Edge (Future Ret)':<30} | {'-':<15} | {trade_metrics['edge_up']:<15.6f} | {trade_metrics['edge_down_signed']:<15.6f}")
        print(f"{'Net Edge (после costs)':<30} | {'-':<15} | {trade_metrics['net_edge_long']:<15.6f} | {trade_metrics['net_edge_short']:<15.6f}")
        print(f"{'Coverage directional':<30} | {trade_metrics['coverage_directional']:<15.4f} (share pred != Flat)")
        print(f"{'Directional Accuracy (DA) без Flat':<30} | {trade_metrics['val_da_without_flat']:<15.4f} (Accuracy where pred != Flat)")
        print(f"{'Средняя Уверенность (C/W)':<30} | Уверенность при правильном: {trade_metrics['conf_correct']:.4f} | Уверенности при ложном: {trade_metrics['conf_wrong']:.4f} | Разница уверенности: {trade_metrics['conf_gap']:.4f}")
        print(f"{'Корреляция с LOB Imbalance':<30} | {trade_metrics['imb_corr']:<15.4f} (Corr with Signal -1/0/1)")
        print("="*85 + "\n")

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


