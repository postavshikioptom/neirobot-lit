"""
train_module.py — Training core: TrainSubset, ProfilerCallback, LiTModule.
Вынесено из train.py в рамках задачи 322.1.
"""
import json
import csv
import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from typing import Any
from torch.profiler import profile, ProfilerActivity, schedule
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
)

from .lit_model import LiTModel
from .dataset import CHANNEL_CONTRACT
from .utils import (
    compute_classification_metrics,
    compute_directional_metrics,
    compute_desc_ranks,
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
    def __init__(
        self,
        seq_len=100,
        lr=1e-4,
        class_weights=None,
        label_smoothing=0.0,
        loss_type="ce",
        focal_gamma=2.0,
        activation='gelu_exact',
        use_time_weighting=False,
        teacher_model=None,
        alpha=0.9,
        temperature=3.0,
        use_regime_weighting=False,
        regime_weights=None,
        num_horizons=1,
        horizon_weights=None,
        use_horizon_embedding=False,
        use_curvature_reg=False,
        curvature_lambda=1e-4,
        input_noise_std=0.005,
        scaler_type="robust",
        winsor_limits=None,
        past_returns_lags=None,
        scheduler=None,
        div_factor=None,
        final_div_factor=None,
        pct_start=None,
        plateau_factor=None,
        plateau_patience=None,
        step_size=None,
        gamma=None,
        weight_decay=None,
        clip_mode=None,
        clip_val=None,
        tb_hist_freq=None,
        tb_embedding_samples=None,
        use_gradient_checkpointing=False,
        metric_contract="standard",
        metric_log_prefix="val",
        metric_directional_base="predicted",
        decision_rule="argmax",
        decision_confidence=0.5,
        decision_hold_threshold=0.6,
        flat_prob_threshold=0.34,
        up_prob_threshold=0.34,
        down_prob_threshold=0.34,
        margin_threshold=0.0,
        cls_loss_weight=1.0,
        vol_loss_weight=1.0,
        multi_task=True,
        report_fee_bps=0.0,
        report_slippage_bps=0.0,
        report_half_spread_bps=0.0,
        enable_channel_attribution=False,
        channel_attribution_samples=128,
        channel_attribution_method="grad_x_input",
        freeze_experimental_features=False,
        **model_params
    ):
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
        model_multi_task = model_params.pop("multi_task", multi_task)
        self.model = LiTModel(
            seq_len=seq_len,
            activation=activation,
            num_horizons=num_horizons,
            use_horizon_embedding=use_horizon_embedding,
            use_gradient_checkpointing=use_gradient_checkpointing,
            multi_task=model_multi_task,
            **model_params
        )
        self.use_time_weighting = use_time_weighting
        self.use_regime_weighting = use_regime_weighting
        self.teacher_model = teacher_model
        self.is_distillation = teacher_model is not None
        self.num_horizons = num_horizons
        self.is_multi_horizon = (num_horizons > 1)
        self.freeze_experimental_features = bool(freeze_experimental_features)
        if self.freeze_experimental_features and self.is_distillation:
            raise ValueError("Experimental distillation path is frozen for this run.")
        if self.freeze_experimental_features and self.is_multi_horizon:
            raise ValueError("Experimental multi-horizon path is frozen for this run.")
        self.enable_multi_task = bool(multi_task)
        self.cls_loss_weight = float(cls_loss_weight)
        self.vol_loss_weight = float(vol_loss_weight)
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
        self._validation_attr_samples = []

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
        channel_names = list(CHANNEL_CONTRACT)
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

        # 3. Комбинированный Multi-Task Loss с явными весами
        cls_weight = float(self.hparams.get("cls_loss_weight", self.cls_loss_weight))
        vol_weight = float(self.hparams.get("vol_loss_weight", self.vol_loss_weight))
        if not self.hparams.get("multi_task", self.enable_multi_task):
            vol_weight = 0.0
        loss = cls_weight * loss_cls + vol_weight * loss_vol + reg_loss

        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at step {self.global_step}: {loss.item()}. Zeroing grads and skipping.")
            self.zero_grad()
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train_loss_vol", loss_vol, on_step=False, on_epoch=True)
        self.log("weight_cls", cls_weight, on_step=False, on_epoch=True)
        self.log("weight_vol", vol_weight, on_step=False, on_epoch=True)
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
        weights = extra_data["weight"]

        logits, vol_pred = self(x, regime_id=regime_id)
        vol_pred = vol_pred.squeeze(-1)

        if self.is_multi_horizon:
            loss_cls = self.criterion(logits, y, sample_weights=None)
        else:
            if self.is_distillation:
                loss_cls = nn.functional.cross_entropy(logits, y)
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

        loss_vol = nn.functional.mse_loss(vol_pred, vol_target)
        mae_vol = nn.functional.l1_loss(vol_pred, vol_target)

        current_imbalance = x[:, -1, 2, 0]
        spread_proxy = x[:, -1, 8, :].mean(dim=1)
        activity_proxy = x[:, -1, 3, :].abs().mean(dim=1)
        self._accumulate_validation_outputs(
            logits=logits,
            labels=y,
            f_ret=f_ret,
            imbalance=current_imbalance,
            spread_proxy=spread_proxy,
            activity_proxy=activity_proxy,
            regime_id=regime_id,
            vol_true=vol_target,
            vol_pred=vol_pred,
            loss_cls=loss_cls,
            loss_vol=loss_vol,
            mae_vol=mae_vol,
        )
        self._accumulate_validation_attr_samples(x=x, labels=y)

        cls_weight = float(self.hparams.get("cls_loss_weight", self.cls_loss_weight))
        vol_weight = float(self.hparams.get("vol_loss_weight", self.vol_loss_weight))
        if not self.hparams.get("multi_task", self.enable_multi_task):
            vol_weight = 0.0
        return cls_weight * loss_cls + vol_weight * loss_vol

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

            from .utils import apply_decision_rule, apply_temperature_scaling, fit_temperature_scaler, compute_directional_metrics

            f_ret = validation_payload["f_ret"]
            imbalance = validation_payload["imbalance"]
            decision_rule = self.hparams.get("decision_rule", "argmax")

            argmax_preds = np.zeros_like(y_true)
            decision_preds = np.zeros_like(y_true)
            temp_per_h = []
            ece_before_list = []
            ece_after_list = []
            mce_before_list = []
            mce_after_list = []
            horizon_trade = []

            for h in range(self.num_horizons):
                logits_h = logits[:, h, :]
                y_true_h = y_true[:, h]
                mask = y_true_h != -100
                if not np.any(mask):
                    temp_per_h.append(1.0)
                    ece_before_list.append(0.0)
                    ece_after_list.append(0.0)
                    mce_before_list.append(0.0)
                    mce_after_list.append(0.0)
                    horizon_trade.append({"coverage_directional": 0.0, "net_edge_total": 0.0, "samples": 0})
                    continue

                logits_h_t = logits_h[mask]
                y_true_h_t = torch.from_numpy(y_true_h[mask]).long()
                temperature_value = fit_temperature_scaler(logits_h_t, y_true_h_t)
                scaled_logits_h = apply_temperature_scaling(logits_h_t, temperature_value)
                scaled_probs_h = torch.softmax(scaled_logits_h, dim=1).cpu().numpy()

                raw_probs_h = torch.softmax(logits_h_t, dim=1).cpu().numpy()
                argmax_preds[mask, h] = np.argmax(raw_probs_h, axis=1)
                decision_preds[mask, h] = apply_decision_rule(
                    scaled_probs_h,
                    decision_rule,
                    decision_confidence=float(self.hparams.get("decision_confidence", 0.5)),
                    decision_hold_threshold=float(self.hparams.get("decision_hold_threshold", 0.6)),
                    flat_prob_threshold=float(self.hparams.get("flat_prob_threshold", 0.34)),
                    up_prob_threshold=float(self.hparams.get("up_prob_threshold", 0.34)),
                    down_prob_threshold=float(self.hparams.get("down_prob_threshold", 0.34)),
                    margin_threshold=float(self.hparams.get("margin_threshold", 0.0)),
                )

                ece_before, mce_before, _ = self.calibration_metrics.calculate(logits_h_t, y_true_h_t)
                ece_after, mce_after, _ = self.calibration_metrics.calculate(scaled_logits_h, y_true_h_t)
                temp_per_h.append(float(temperature_value))
                ece_before_list.append(float(ece_before))
                ece_after_list.append(float(ece_after))
                mce_before_list.append(float(mce_before))
                mce_after_list.append(float(mce_after))

                f_ret_h = f_ret[:, h] if f_ret.ndim == 2 else f_ret
                trade_metrics = compute_directional_metrics(
                    y_true_h[mask],
                    decision_preds[mask, h] if decision_rule != "argmax" else argmax_preds[mask, h],
                    logits_h_t.cpu().numpy(),
                    f_ret_h[mask],
                    imbalance[mask],
                    directional_base=self.hparams.get("metric_directional_base", "predicted"),
                    fee_bps=self.hparams.get("report_fee_bps", 0.0),
                    slippage_bps=self.hparams.get("report_slippage_bps", 0.0),
                    half_spread_bps=self.hparams.get("report_half_spread_bps", 0.0),
                    probs=scaled_probs_h if decision_rule != "argmax" else raw_probs_h,
                )
                horizon_trade.append(
                    {
                        "coverage_directional": float(trade_metrics.get("coverage_directional", 0.0)),
                        "net_edge_total": float(trade_metrics.get("net_edge_total", 0.0)),
                        "samples": int(np.sum(mask)),
                    }
                )

            argmax_metrics = compute_multi_horizon_metrics(y_true, argmax_preds, self.num_horizons)
            decision_metrics = compute_multi_horizon_metrics(y_true, decision_preds, self.num_horizons)
            avg_mcc_argmax = np.mean([argmax_metrics.get(f"mcc_h{h}", 0.0) for h in range(self.num_horizons)])
            avg_mcc_decision = np.mean([decision_metrics.get(f"mcc_h{h}", 0.0) for h in range(self.num_horizons)])
            primary_avg_mcc = avg_mcc_decision if decision_rule != "argmax" else avg_mcc_argmax

            self.log("val_mcc_primary", primary_avg_mcc, logger=True, prog_bar=True, add_dataloader_idx=False)

            total_samples = sum(item["samples"] for item in horizon_trade)
            if total_samples > 0:
                coverage_directional = sum(item["coverage_directional"] * item["samples"] for item in horizon_trade) / total_samples
                net_edge_total = sum(item["net_edge_total"] * item["samples"] for item in horizon_trade) / total_samples
            else:
                coverage_directional = 0.0
                net_edge_total = 0.0

            ece_before_avg = float(np.mean(ece_before_list)) if ece_before_list else 0.0
            ece_after_avg = float(np.mean(ece_after_list)) if ece_after_list else 0.0
            mce_before_avg = float(np.mean(mce_before_list)) if mce_before_list else 0.0
            mce_after_avg = float(np.mean(mce_after_list)) if mce_after_list else 0.0
            calibration_improved = float(ece_after_avg <= ece_before_avg)
            self.log("val_ece_after", ece_after_avg, logger=True, add_dataloader_idx=False)
            self.log("val_ece_before", ece_before_avg, logger=True, add_dataloader_idx=False)
            self.log("val_mce_after", mce_after_avg, logger=True, add_dataloader_idx=False)
            self.log("val_mce_before", mce_before_avg, logger=True, add_dataloader_idx=False)
            self.log("val_calibration_improved", calibration_improved, logger=True, add_dataloader_idx=False)
            self.log("coverage_directional", coverage_directional, logger=True, add_dataloader_idx=False)
            self.log("net_edge_total", net_edge_total, logger=True, add_dataloader_idx=False)

            if ece_after_avg > ece_before_avg:
                print(f"[WARN] Calibration regressed: ece_after={ece_after_avg:.6f} > ece_before={ece_before_avg:.6f}")

            finalized = {
                "epoch": int(self.current_epoch),
                "metric_contract": self.hparams.get("metric_contract", "standard"),
                "metric_log_prefix": self.hparams.get("metric_log_prefix", "val"),
                "metric_directional_base": self.hparams.get("metric_directional_base", "predicted"),
                "decision_rule": decision_rule,
                "decision_rule_config": {
                    "decision_confidence": float(self.hparams.get("decision_confidence", 0.5)),
                    "decision_hold_threshold": float(self.hparams.get("decision_hold_threshold", 0.6)),
                    "flat_prob_threshold": float(self.hparams.get("flat_prob_threshold", 0.34)),
                    "up_prob_threshold": float(self.hparams.get("up_prob_threshold", 0.34)),
                    "down_prob_threshold": float(self.hparams.get("down_prob_threshold", 0.34)),
                    "margin_threshold": float(self.hparams.get("margin_threshold", 0.0)),
                },
                "quality": {
                    "val_loss": float(validation_payload["loss_cls"]),
                    "val_loss_cls": float(validation_payload["loss_cls"]),
                    "val_mse_vol": float(validation_payload["loss_vol"]),
                    "val_mae_vol": float(validation_payload["mae_vol"]),
                    "val_vol_mse": float(np.mean((validation_payload["vol_true"] - validation_payload["vol_pred"])**2)),
                    "val_vol_mae": float(np.mean(np.abs(validation_payload["vol_true"] - validation_payload["vol_pred"]))),
                    "val_mcc_primary": float(primary_avg_mcc),
                    "val_mcc_np": float(primary_avg_mcc),
                },
                "calibration": {
                    "val_ece": float(ece_after_avg),
                    "val_mce": float(mce_after_avg),
                    "val_ece_before": float(ece_before_avg),
                    "val_mce_before": float(mce_before_avg),
                    "val_ece_after": float(ece_after_avg),
                    "val_mce_after": float(mce_after_avg),
                    "temperature": temp_per_h,
                },
                "coverage": {
                    "coverage_directional": float(coverage_directional),
                },
                "trade": {
                    "net_edge_total": float(net_edge_total),
                },
                "class_metrics": {},
                "regime_metrics": {},
                "market_regime_buckets": self._build_market_regime_bucket_report(
                    validation_payload,
                    decision_preds if decision_rule != "argmax" else argmax_preds,
                ),
                "argmax_metrics": argmax_metrics,
                "decision_rule_metrics": decision_metrics,
            }
            for h in range(self.num_horizons):
                finalized["quality"][f"mcc_h{h}"] = float((decision_metrics if decision_rule != "argmax" else argmax_metrics).get(f"mcc_h{h}", 0.0))
                finalized["quality"][f"f1_h{h}"] = float((decision_metrics if decision_rule != "argmax" else argmax_metrics).get(f"f1_h{h}", 0.0))
                finalized["quality"][f"samples_h{h}"] = int(metrics.get(f"samples_h{h}", 0))

            symbol = getattr(self.trainer, 'symbol', 'UNKNOWN')
            base_path = Path(__file__).parent.parent.parent
            calibration_dir = base_path / "artifacts" / symbol / "calibration"
            calibration_dir.mkdir(parents=True, exist_ok=True)
            scaler_path = calibration_dir / "temperature_scaler.json"
            scaler_payload = {
                "temperature": temp_per_h,
                "ece_before": ece_before_list,
                "mce_before": mce_before_list,
                "ece_after": ece_after_list,
                "mce_after": mce_after_list,
                "ece_before_avg": ece_before_avg,
                "ece_after_avg": ece_after_avg,
                "mce_before_avg": mce_before_avg,
                "mce_after_avg": mce_after_avg,
                "epoch": int(self.current_epoch),
            }
            scaler_path.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")
        else:
            y_true_tensor = torch.from_numpy(y_true).long()
            if not torch.isfinite(logits).all():
                print("\n" + "!" * 80)
                print("⚠️  CRITICAL WARNING: NaN or Inf detected in model logits during validation!")
                print("   This indicates extreme numerical instability (exploding gradients).")
                print("   Metrics and visualizations for this epoch will be unreliable.")
                print("!" * 80 + "\n")
            raw_probs = torch.softmax(logits, dim=1).cpu().numpy()
            argmax_pred = np.argmax(raw_probs, axis=1)
            argmax_metrics = self._finalize_validation_metrics(
                validation_payload,
                y_pred=argmax_pred,
                probs=raw_probs,
                use_torch_metrics=True,
            )

            from .utils import apply_decision_rule, apply_temperature_scaling, fit_temperature_scaler

            temperature_value = fit_temperature_scaler(logits, y_true_tensor)
            scaled_logits = apply_temperature_scaling(logits, temperature_value)
            scaled_probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()

            ece_before, mce_before, bin_data_before = self.calibration_metrics.calculate(logits, y_true_tensor)
            ece_after, mce_after, bin_data_after = self.calibration_metrics.calculate(scaled_logits, y_true_tensor)

            decision_rule = self.hparams.get("decision_rule", "argmax")
            decision_pred = apply_decision_rule(
                scaled_probs,
                decision_rule,
                decision_confidence=float(self.hparams.get("decision_confidence", 0.5)),
                decision_hold_threshold=float(self.hparams.get("decision_hold_threshold", 0.6)),
                flat_prob_threshold=float(self.hparams.get("flat_prob_threshold", 0.34)),
                up_prob_threshold=float(self.hparams.get("up_prob_threshold", 0.34)),
                down_prob_threshold=float(self.hparams.get("down_prob_threshold", 0.34)),
                margin_threshold=float(self.hparams.get("margin_threshold", 0.0)),
            )
            decision_metrics = self._finalize_validation_metrics(
                validation_payload,
                y_pred=decision_pred,
                probs=scaled_probs,
                use_torch_metrics=False,
            )

            primary_metrics = decision_metrics if decision_rule != "argmax" else argmax_metrics
            y_pred = decision_pred if decision_rule != "argmax" else argmax_pred
            calibration = {
                "val_ece": float(ece_after),
                "val_mce": float(mce_after),
                "val_ece_before": float(ece_before),
                "val_mce_before": float(mce_before),
                "val_ece_after": float(ece_after),
                "val_mce_after": float(mce_after),
                "temperature": float(temperature_value),
                "bin_data": bin_data_after,
                "bin_data_before": bin_data_before,
            }
            calibration_improved = float(ece_after <= ece_before)
            self.log("val_calibration_improved", calibration_improved, logger=True, add_dataloader_idx=False)
            if ece_after > ece_before:
                print(f"[WARN] Calibration regressed: ece_after={ece_after:.6f} > ece_before={ece_before:.6f}")
            finalized = {
                "epoch": int(self.current_epoch),
                "metric_contract": self.hparams.get("metric_contract", "standard"),
                "metric_log_prefix": self.hparams.get("metric_log_prefix", "val"),
                "metric_directional_base": self.hparams.get("metric_directional_base", "predicted"),
                "decision_rule": decision_rule,
                "decision_rule_config": {
                    "decision_confidence": float(self.hparams.get("decision_confidence", 0.5)),
                    "decision_hold_threshold": float(self.hparams.get("decision_hold_threshold", 0.6)),
                    "flat_prob_threshold": float(self.hparams.get("flat_prob_threshold", 0.34)),
                    "up_prob_threshold": float(self.hparams.get("up_prob_threshold", 0.34)),
                    "down_prob_threshold": float(self.hparams.get("down_prob_threshold", 0.34)),
                    "margin_threshold": float(self.hparams.get("margin_threshold", 0.0)),
                },
                "quality": primary_metrics["quality"],
                "calibration": calibration,
                "coverage": primary_metrics["coverage"],
                "trade": primary_metrics["trade"],
                "class_metrics": primary_metrics["class_metrics"],
                "regime_metrics": {},
                "argmax_metrics": argmax_metrics,
                "decision_rule_metrics": decision_metrics,
            }

            symbol = getattr(self.trainer, 'symbol', 'UNKNOWN')
            base_path = Path(__file__).parent.parent.parent
            calibration_dir = base_path / "artifacts" / symbol / "calibration"
            calibration_dir.mkdir(parents=True, exist_ok=True)
            scaler_path = calibration_dir / "temperature_scaler.json"
            scaler_payload = {
                "temperature": float(temperature_value),
                "ece_before": float(ece_before),
                "mce_before": float(mce_before),
                "ece_after": float(ece_after),
                "mce_after": float(mce_after),
                "epoch": int(self.current_epoch),
            }
            scaler_path.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

            self._log_final_validation_metrics(finalized)
            epoch_time = _time.time() - self.epoch_start_time if hasattr(self, 'epoch_start_time') else 0
            epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s"
            print(f"\nEpoch {self.current_epoch} ({epoch_time_str}) Validation: MCC={finalized['quality']['val_mcc_primary']:.4f}, "
                  f"Macro-F1={finalized['quality']['val_f1_macro_np']:.4f}, ECE={ece_after:.4f}, MCE={mce_after:.4f}")

            self._log_extended_analytics(finalized["trade"])

            if (self.hparams.get('enable_epoch_end_plots', False) and not skip_heavy_artifacts
                    and self.current_epoch % 20 == 0):
                reports_dir = base_path / "reports" / symbol
                reports_dir.mkdir(parents=True, exist_ok=True)
                save_path = reports_dir / f"reliability_diagram_epoch_{self.current_epoch}.png"
                plot_reliability_diagram(bin_data_after, ece_after, mce_after, str(save_path))

            if self.logger and hasattr(self.logger, 'experiment'):
                writer = self.logger.experiment
                if (self.hparams.get('enable_epoch_end_plots', False) and not skip_heavy_artifacts
                        and self.current_epoch % 20 == 0):
                    from .utils import plot_confusion_matrix_tensorboard, plot_pr_curves_tensorboard
                    class_names = ["Flat", "Up", "Down"]
                    plot_confusion_matrix_tensorboard(y_true, y_pred, class_names, writer, self.current_epoch)
                    plot_pr_curves_tensorboard(y_true, scaled_probs, class_names, writer, self.current_epoch)

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

        self._run_channel_attribution_epoch_end(validation_payload, y_pred)

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

    def _accumulate_validation_outputs(self, logits, labels, f_ret, imbalance, spread_proxy, activity_proxy, regime_id, vol_true, vol_pred, loss_cls, loss_vol, mae_vol):
        self._validation_accumulator.append(
            {
                "logits": logits.detach().cpu(),
                "labels": labels.detach().cpu(),
                "f_ret": f_ret.detach().cpu(),
                "imbalance": imbalance.detach().cpu(),
                "spread_proxy": spread_proxy.detach().cpu(),
                "activity_proxy": activity_proxy.detach().cpu(),
                "regime_id": regime_id.detach().cpu() if regime_id is not None else None,
                "vol_true": vol_true.detach().cpu(),
                "vol_pred": vol_pred.detach().cpu(),
                "loss_cls": loss_cls.detach().cpu(),
                "loss_vol": loss_vol.detach().cpu(),
                "mae_vol": mae_vol.detach().cpu(),
            }
        )

    def _accumulate_validation_attr_samples(self, x, labels):
        if not bool(self.hparams.get("enable_channel_attribution", False)):
            return
        max_samples = int(self.hparams.get("channel_attribution_samples", 128))
        if max_samples <= 0:
            return
        if self.is_multi_horizon:
            return
        if x.ndim != 4:
            return
        collected = sum(int(item["x"].shape[0]) for item in self._validation_attr_samples)
        if collected >= max_samples:
            return
        remaining = max_samples - collected
        take = min(int(x.shape[0]), remaining)
        if take <= 0:
            return
        self._validation_attr_samples.append(
            {
                "x": x[:take].detach().cpu(),
                "labels": labels[:take].detach().cpu(),
            }
        )

    def _gather_validation_payload(self):
        payload = {
            "logits": torch.cat([item["logits"] for item in self._validation_accumulator], dim=0),
            "y_true": np.concatenate([item["labels"].numpy() for item in self._validation_accumulator]),
            "f_ret": np.concatenate([item["f_ret"].numpy() for item in self._validation_accumulator]),
            "imbalance": np.concatenate([item["imbalance"].numpy() for item in self._validation_accumulator]),
            "spread_proxy": np.concatenate([item["spread_proxy"].numpy() for item in self._validation_accumulator]),
            "activity_proxy": np.concatenate([item["activity_proxy"].numpy() for item in self._validation_accumulator]),
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

    def _build_market_regime_bucket_report(self, payload, y_pred, probs=None):
        from .utils import compute_directional_metrics

        y_true = payload["y_true"]
        logits_np = payload["logits"].numpy()
        f_ret = payload["f_ret"]
        imbalance = payload["imbalance"]

        if y_true.ndim > 1:
            y_true = y_true[:, 0]
            y_pred = y_pred[:, 0] if y_pred.ndim > 1 else y_pred
            f_ret = f_ret[:, 0] if f_ret.ndim > 1 else f_ret
            logits_np = logits_np[:, 0, :] if logits_np.ndim == 3 else logits_np

        valid_mask = y_true != -100
        if not np.any(valid_mask):
            return {}

        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        f_ret = f_ret[valid_mask]
        imbalance = imbalance[valid_mask]
        logits_np = logits_np[valid_mask]
        spread_values = payload["spread_proxy"][valid_mask]
        volatility_values = payload["vol_true"][valid_mask]
        activity_values = payload["activity_proxy"][valid_mask]

        def _edges(values: np.ndarray) -> list[float]:
            q1, q2 = np.quantile(values, [0.33, 0.66])
            return [-np.inf, float(q1), float(q2), np.inf]

        def _bucket_rows(values: np.ndarray, edges: list[float], metric_name: str):
            rows = []
            for idx, bucket_name in enumerate(("low", "mid", "high")):
                lo = edges[idx]
                hi = edges[idx + 1]
                mask = (values >= lo) & (values < hi)
                samples = int(np.sum(mask))
                if samples == 0:
                    rows.append(
                        {
                            "bucket": bucket_name,
                            "samples": 0,
                            "coverage_directional": 0.0,
                            "mcc_primary": 0.0,
                            "net_edge_total": 0.0,
                        }
                    )
                    continue
                bucket_true = y_true[mask]
                bucket_pred = y_pred[mask]
                bucket_f_ret = f_ret[mask]
                bucket_imb = imbalance[mask]
                bucket_logits = logits_np[mask]
                direction = compute_directional_metrics(
                    bucket_true,
                    bucket_pred,
                    bucket_logits,
                    bucket_f_ret,
                    bucket_imb,
                    directional_base=self.hparams.get("metric_directional_base", "predicted"),
                    fee_bps=self.hparams.get("report_fee_bps", 0.0),
                    slippage_bps=self.hparams.get("report_slippage_bps", 0.0),
                    half_spread_bps=self.hparams.get("report_half_spread_bps", 0.0),
                    probs=probs[mask] if probs is not None else None,
                )
                bucket_mcc = float(safe_matthews_corrcoef(bucket_true, bucket_pred))
                rows.append(
                    {
                        "bucket": bucket_name,
                        "samples": samples,
                        "coverage_directional": float(direction.get("coverage_directional", 0.0)),
                        "mcc_primary": bucket_mcc,
                        "net_edge_total": float(direction.get("net_edge_total", 0.0)),
                    }
                )
            print(f"Market bucket report [{metric_name}]: {rows}")
            return rows

        return {
            "spread": _bucket_rows(spread_values, _edges(spread_values), "spread"),
            "volatility": _bucket_rows(volatility_values, _edges(volatility_values), "volatility"),
            "activity": _bucket_rows(activity_values, _edges(activity_values), "activity"),
        }

    def _finalize_validation_metrics(self, payload, *, y_pred, probs=None, use_torch_metrics=True):
        class_weights = self.criterion.weight.cpu().numpy() if hasattr(self.criterion, 'weight') and self.criterion.weight is not None else None
        quality_metrics = compute_classification_metrics(payload["y_true"], y_pred, class_weights=class_weights)
        direction_metrics = compute_directional_metrics(
            payload["y_true"],
            y_pred,
            payload["logits"].numpy(),
            payload["f_ret"],
            payload["imbalance"],
            directional_base=self.hparams.get("metric_directional_base", "predicted"),
            fee_bps=self.hparams.get("report_fee_bps", 0.0),
            slippage_bps=self.hparams.get("report_slippage_bps", 0.0),
            half_spread_bps=self.hparams.get("report_half_spread_bps", 0.0),
            probs=probs,
        )

        y_true_tensor = torch.from_numpy(payload["y_true"]).long()
        if use_torch_metrics:
            val_mcc_torch = float(self.mcc(payload["logits"], y_true_tensor).detach().cpu())
            val_f1_torch = float(self.f1_macro(payload["logits"], y_true_tensor).detach().cpu())
        else:
            val_mcc_torch = float(quality_metrics["mcc"])
            val_f1_torch = float(quality_metrics["f1_macro"])
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
            "coverage": coverage,
            "trade": trade,
            "class_metrics": class_metrics,
            "regime_metrics": {},
            "market_regime_buckets": self._build_market_regime_bucket_report(payload, y_pred, probs=probs),
        }

    def _log_final_validation_metrics(self, finalized):
        for name, value in finalized["quality"].items():
            self.log(name, float(value), logger=True, prog_bar=(name == "val_mcc_primary"), add_dataloader_idx=False)
        for name, value in finalized["calibration"].items():
            if "bin_data" in name:
                continue
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
            "decision_rule": finalized.get("decision_rule"),
            "decision_rule_config": finalized.get("decision_rule_config", {}),
            "quality": finalized["quality"],
            "calibration": {k: v for k, v in finalized["calibration"].items() if "bin_data" not in k},
            "coverage": finalized["coverage"],
            "trade": finalized["trade"],
            "class_metrics": finalized["class_metrics"],
            "regime_metrics": finalized["regime_metrics"],
            "market_regime_buckets": finalized.get("market_regime_buckets", {}),
            "argmax_metrics": finalized.get("argmax_metrics", {}),
            "decision_rule_metrics": finalized.get("decision_rule_metrics", {}),
        }
        report_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def _reset_validation_state(self):
        self._validation_accumulator.clear()
        self._validation_attr_samples.clear()

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

    def _run_channel_attribution_epoch_end(self, payload: dict[str, Any], y_pred):
        if not bool(self.hparams.get("enable_channel_attribution", False)):
            return
        if self.is_multi_horizon:
            print("[ATTR] Skip: channel attribution поддерживается только для single-horizon.")
            return
        if not self._validation_attr_samples:
            print("[ATTR] Skip: нет сохранённых validation samples для attribution.")
            return

        x = torch.cat([item["x"] for item in self._validation_attr_samples], dim=0)
        y_true = torch.cat([item["labels"] for item in self._validation_attr_samples], dim=0).long()
        if x.shape[2] != len(CHANNEL_CONTRACT):
            raise RuntimeError(
                f"Channel contract mismatch for attribution: input has {x.shape[2]} channels, "
                f"expected {len(CHANNEL_CONTRACT)}"
            )

        method = str(self.hparams.get("channel_attribution_method", "grad_x_input"))
        attr_scores, pred_labels = self._compute_channel_attribution_scores(x, method=method)

        stats = self._build_channel_attribution_stats(
            attr_scores=attr_scores,
            y_true=y_true.numpy(),
            y_pred=pred_labels,
        )
        self._save_channel_attribution_artifacts(stats, method=method, sample_count=int(attr_scores.shape[0]))
        self._print_channel_attribution_summary(stats)

    def _compute_channel_attribution_scores(self, x_cpu: torch.Tensor, method: str) -> tuple[np.ndarray, np.ndarray]:
        device = self.device
        x = x_cpu.to(device=device, dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            base_logits, _ = self(x)
            pred_labels = torch.argmax(base_logits, dim=1)
            target_logits = base_logits.gather(1, pred_labels.unsqueeze(1)).squeeze(1)

        if method == "grad_x_input":
            x_grad = x.detach().clone().requires_grad_(True)
            logits, _ = self(x_grad)
            target = torch.argmax(logits, dim=1)
            target_sum = logits.gather(1, target.unsqueeze(1)).sum()
            grads = torch.autograd.grad(target_sum, x_grad, retain_graph=False, create_graph=False)[0]
            attr = grads * x_grad
            # x shape: [B, seq, channels, levels] -> aggregation to [B, channels]
            scores = attr.mean(dim=(1, 3)).detach().cpu().numpy()
            return scores, pred_labels.detach().cpu().numpy()

        if method == "occlusion":
            channel_count = x.shape[2]
            scores = torch.zeros((x.shape[0], channel_count), device=device, dtype=torch.float32)
            with torch.no_grad():
                for ch_idx in range(channel_count):
                    x_occ = x.clone()
                    x_occ[:, :, ch_idx, :] = 0.0
                    occ_logits, _ = self(x_occ)
                    occ_target = occ_logits.gather(1, pred_labels.unsqueeze(1)).squeeze(1)
                    scores[:, ch_idx] = target_logits - occ_target
            return scores.detach().cpu().numpy(), pred_labels.detach().cpu().numpy()

        raise ValueError(f"Unknown --channel_attribution_method: {method}")

    def _build_channel_attribution_stats(self, attr_scores: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        class_names = {0: "Flat", 1: "Up", 2: "Down"}
        stats: dict[str, Any] = {
            "channels": list(CHANNEL_CONTRACT),
            "groups": {},
        }

        def _add_group(group: str, mask: np.ndarray):
            if mask.dtype != np.bool_:
                mask = mask.astype(bool)
            if mask.sum() == 0:
                stats["groups"][group] = []
                return
            selected = attr_scores[mask]
            mean_abs = np.mean(np.abs(selected), axis=0)
            signed_mean = np.mean(selected, axis=0)
            ranks = compute_desc_ranks(mean_abs)
            rows = []
            for idx, channel in enumerate(CHANNEL_CONTRACT):
                rows.append(
                    {
                        "channel": channel,
                        "mean_abs_attr": float(mean_abs[idx]),
                        "signed_attr_mean": float(signed_mean[idx]),
                        "rank": int(ranks[idx]),
                        "group": group,
                    }
                )
            stats["groups"][group] = rows

        n = attr_scores.shape[0]
        _add_group("general", np.ones(n, dtype=bool))
        for class_id, class_name in class_names.items():
            _add_group(f"predicted:{class_name}", y_pred == class_id)
        for class_id, class_name in class_names.items():
            _add_group(f"true:{class_name}", y_true == class_id)
        correct_mask = y_pred == y_true
        _add_group("correctness:correct", correct_mask)
        _add_group("correctness:wrong", ~correct_mask)
        return stats

    def _save_channel_attribution_artifacts(self, stats: dict[str, Any], method: str, sample_count: int):
        symbol = getattr(self.trainer, "symbol", "UNKNOWN")
        base_path = Path(__file__).parent.parent.parent
        out_dir = base_path / "artifacts" / symbol / "attribution"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"epoch_{self.current_epoch}.json"
        csv_path = out_dir / f"epoch_{self.current_epoch}.csv"

        json_payload = {
            "epoch": int(self.current_epoch),
            "method": method,
            "sample_count": int(sample_count),
            "channels": list(CHANNEL_CONTRACT),
            "groups": stats["groups"],
        }
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

        csv_rows = []
        for group_name, rows in stats["groups"].items():
            for row in rows:
                csv_rows.append(
                    {
                        "group": group_name,
                        "channel": row["channel"],
                        "mean_abs_attr": row["mean_abs_attr"],
                        "signed_attr_mean": row["signed_attr_mean"],
                        "rank": row["rank"],
                    }
                )
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["group", "channel", "mean_abs_attr", "signed_attr_mean", "rank"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

    def _print_channel_attribution_summary(self, stats: dict[str, Any]):
        print("\n[ATTR] High-signal channel attribution summary:")
        for class_name in ("Flat", "Up", "Down"):
            group = f"predicted:{class_name}"
            rows = stats["groups"].get(group, [])
            if not rows:
                print(f"[ATTR] Top-5 {class_name}: no samples")
                continue
            top5 = sorted(rows, key=lambda item: item["rank"])[:5]
            summary = ", ".join(f"{row['channel']}({row['mean_abs_attr']:.4f})" for row in top5)
            print(f"[ATTR] Top-5 {class_name}: {summary}")

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


