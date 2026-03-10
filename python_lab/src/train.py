import torch
import torch.nn as nn
import pytorch_lightning as pl
import polars as pl_pol
import numpy as np
import argparse
import psutil
import json
import os
import datetime
from datetime import UTC
from tqdm import tqdm
from sklearn.metrics import classification_report, matthews_corrcoef
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix
)
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner, PatientPruner
from optuna.exceptions import TrialPruned

from .lit_model import LiTModel
from .dataset import LOBDataset, LOBDataLoader, balance_dataset, apply_symmetric_flip, apply_volume_jitter
from .features import FeatureEngineer
from .labels import Labeler
from .normalization import Normalizer
from .utils import compute_metrics, FocalLoss, save_confusion_matrices, CalibrationMetrics, plot_reliability_diagram


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



def _streaming_worker_init_fn(worker_id: int):
    """
    Worker initialization function for streaming mode DataLoader.
    Each worker creates its own LazyFrame to avoid file descriptor conflicts.
    
    Args:
        worker_id: ID of the current worker process
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    
    dataset = worker_info.dataset
    
    # Если это streaming режим, создаем новый LazyFrame для воркера
    if hasattr(dataset, 'data_mode') and dataset.data_mode == "streaming":
        if hasattr(dataset, 'file_path') and dataset.file_path is not None:
            # Каждый воркер создает свой собственный LazyFrame
            import polars as pl
            dataset.lazy_df = pl.scan_parquet(dataset.file_path, low_memory=True)
            # Переинициализируем row_offsets для нового LazyFrame
            if hasattr(dataset, '_build_row_offsets'):
                total_rows = dataset.lazy_df.select(pl_pol.len()).collect(engine="streaming").item()
                dataset.row_offsets = dataset._build_row_offsets(dataset.file_path, total_rows)
            print(f"[Worker {worker_id}] Initialized private LazyFrame for streaming mode")

"""
Knowledge Distillation Support (Задача 151):

Этот модуль поддерживает два режима обучения:
1. Обычное обучение (--mode train) - обучение модели с нуля
2. Knowledge Distillation (--mode distill) - дистилляция знаний от teacher к student

Пример использования:

# Шаг 1: Обучить тяжелую teacher модель
python -m python_lab.src.train --symbol BTCUSDT --mode train --epochs 50 \\
    --d_model 256 --nhead 8 --num_layers 8

# Шаг 2: Дистиллировать в компактную student модель
python -m python_lab.src.train --symbol BTCUSDT --mode distill \\
    --teacher_path bots/BTCUSDT/models/checkpoints/teacher_lit.pt \\
    --student_d_model 64 --student_nhead 4 --student_num_layers 2 \\
    --alpha 0.9 --temperature 3.0 --epochs 30

Параметры distillation:
- --alpha: вес soft loss (0.9 рекомендуется для LOB данных)
- --temperature: температура для размягчения логитов (2-5)
- --student_*: параметры архитектуры student модели

После обучения выводится сравнение Teacher vs Student:
- MCC (Matthews Correlation Coefficient)
- Latency (миллисекунды)
- Parameters (количество параметров)
- Speedup и Compression ratio
"""

class LiTModule(pl.LightningModule):
    """
    LightningModule для обучения модели LiT.
    Обертка над nn.Module, добавляющая логику обучения, валидации и оптимизации.
    """
    def __init__(self, seq_len=100, lr=1e-4, class_weights=None, label_smoothing=0.0, loss_type="ce", focal_gamma=2.0, activation='gelu_exact', use_time_weighting=False, teacher_model=None, alpha=0.9, temperature=3.0, use_regime_weighting=False, regime_weights=None, num_horizons=1, horizon_weights=None, use_horizon_embedding=False, use_curvature_reg=True, curvature_lambda=1e-4, input_noise_std=0.005, scaler_type="robust", winsor_limits=None, past_returns_lags=None, scheduler=None, div_factor=None, final_div_factor=None, pct_start=None, plateau_factor=None, plateau_patience=None, step_size=None, gamma=None, weight_decay=None, clip_mode=None, clip_val=None, tb_hist_freq=None, tb_embedding_samples=None, **model_params):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "teacher_model", "regime_weights", "horizon_weights"])
        self.model = LiTModel(seq_len=seq_len, activation=activation, num_horizons=num_horizons, use_horizon_embedding=use_horizon_embedding, **model_params)
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
        # Если используется Focal Loss, отключаем label_smoothing
        effective_label_smoothing = 0.0 if loss_type == "focal" else label_smoothing

        # Если используется distillation, настраиваем teacher
        if self.is_distillation:
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)
            # Импортируем DistillationLoss из utils
            from .utils import DistillationLoss
            self.distillation_criterion = DistillationLoss(
                alpha=alpha, 
                temperature=temperature, 
                reduction='none' if (use_time_weighting or use_regime_weighting) else 'batchmean',
                label_smoothing=effective_label_smoothing,
                horizon_weights=horizon_weights if self.is_multi_horizon else None
            )
        
        # Подготовка весов классов
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        # Подготовка весов режимов (для Regime-Weighted Loss)
        if regime_weights is not None and not isinstance(regime_weights, torch.Tensor):
            regime_weights = torch.tensor(regime_weights, dtype=torch.float32)
        self.regime_weights = regime_weights

        # Выбор функции потерь для классификации (только если не distillation)
        if not self.is_distillation:
            if self.is_multi_horizon:
                # Multi-Horizon Loss (Задача 160)
                from .utils import MultiHorizonLoss
                self.criterion = MultiHorizonLoss(
                    num_horizons=num_horizons,
                    horizon_weights=horizon_weights,
                    class_weights=class_weights,
                    label_smoothing=effective_label_smoothing,
                    reduction='mean'  # MultiHorizonLoss внутренне обрабатывает sample_weights
                )
            else:
                # Single horizon - обычный loss
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
        
        # Лосс для волатильности (Регрессия)
        self.vol_criterion = nn.MSELoss(reduction='none' if (use_time_weighting or use_regime_weighting) else 'mean')
        self.vol_mae = nn.L1Loss()

        # Обучаемые веса для Multi-Task Loss (Uncertainty Weighting)
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_vol = nn.Parameter(torch.zeros(1))

        # Метрики для мониторинга (3 класса: 0=Flat, 1=Up, 2=Down)
        self.acc = MulticlassAccuracy(num_classes=3)
        self.mcc = MulticlassMatthewsCorrCoef(num_classes=3)
        self.f1_macro = MulticlassF1Score(num_classes=3, average="macro")
        
        # Метрики по классам (0=Flat, 1=Up, 2=Down)
        self.precision_per_class = MulticlassPrecision(num_classes=3, average=None)
        self.recall_per_class = MulticlassRecall(num_classes=3, average=None)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=3)

        # Списки для накопления результатов валидации (для sklearn метрик)
        self.val_y_true = []
        self.val_y_pred = []
        self.val_logits = []  # Для расчета ECE/MCE
        self.val_vol_true = []
        self.val_vol_pred = []
        
        # Для логирования метрик по режимам
        self.val_regime_ids = []
        
        # Инициализация метрик калибровки
        self.calibration_metrics = CalibrationMetrics(n_bins=15)

    def forward(self, x, regime_id=None):
        return self.model(x, regime_id=regime_id)
    
    def on_train_epoch_start(self):
        """
        Вызывается в начале каждой эпохи обучения.
        Настраиваем activation hooks для мониторинга (Задача 158).
        """
        # Засекаем время начала эпохи
        import time
        self.epoch_start_time = time.time()
        
        # Получаем параметры из hparams или используем значения по умолчанию
        tb_hist_freq = self.hparams.get("tb_hist_freq", 10)
        
        if self.logger and hasattr(self.logger, 'experiment'):
            from .utils import setup_activation_hooks
            writer = self.logger.experiment
            
            # Настраиваем hooks (они будут автоматически удалены после эпохи)
            self.activation_hooks = setup_activation_hooks(
                self.model, writer, self.current_epoch, hist_freq=tb_hist_freq
            )
    
    def on_train_epoch_end(self):
        """
        Вызывается в конце каждой эпохи обучения.
        Удаляем activation hooks.
        """
        # Удаляем hooks после эпохи
        if hasattr(self, 'activation_hooks'):
            for handle in self.activation_hooks:
                handle.remove()
            delattr(self, 'activation_hooks')

    def training_step(self, batch, batch_idx):
        # Распаковываем батч: x, y, vol, weights, regime_id
        if len(batch) == 5:
            x, y, vol_target, weights, regime_id = batch
        else:
            # Обратная совместимость: если regime_id нет, используем None
            x, y, vol_target, weights = batch
            regime_id = None
        
        # Задача 238: Применяем input noise injection во время обучения
        if self.input_noise_std > 0:
            from .lit_model import apply_input_noise
            x = apply_input_noise(x, std=self.input_noise_std)
        
        # Выход модели теперь содержит логиты и предсказание волатильности
        logits, vol_pred = self(x, regime_id=regime_id)
        vol_pred = vol_pred.squeeze(-1)
        
        # 1. Лосс классификации
        if self.is_distillation:
            # Knowledge Distillation: получаем логиты от teacher
            with torch.no_grad():
                teacher_logits, _ = self.teacher_model(x, regime_id=regime_id)
            
            # Используем DistillationLoss
            loss_cls_raw = self.distillation_criterion(logits, teacher_logits, y)
            if self.use_time_weighting or self.use_regime_weighting:
                # Комбинируем временные веса и веса режимов
                combined_weights = weights
                if self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
                    regime_w = self.regime_weights[regime_id].to(weights.device)
                    combined_weights = combined_weights * regime_w
                
                # Если multi-horizon, loss_cls_raw имеет форму (B, H), а combined_weights (B,)
                # Применяем веса к каждому примеру
                if self.is_multi_horizon and loss_cls_raw.dim() > 1:
                    # Приводим веса к форме (B, 1) для корректного broadcasting
                    loss_cls = (loss_cls_raw * combined_weights.unsqueeze(-1)).mean()
                else:
                    loss_cls = (loss_cls_raw * combined_weights).mean()
            else:
                loss_cls = loss_cls_raw.mean() if self.is_multi_horizon else loss_cls_raw
        else:
            # Обычное обучение
            if self.is_multi_horizon:
                # Multi-Horizon Loss (Задача 160)
                # logits: (B, H, 3), y: (B, H), weights: (B,)
                
                # Комбинируем временные веса и веса режимов для Multi-Horizon
                combined_weights = weights if (self.use_time_weighting or self.use_regime_weighting) else None
                if combined_weights is not None and self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
                    regime_w = self.regime_weights[regime_id].to(weights.device)
                    combined_weights = combined_weights * regime_w
                
                # Передаем веса внутрь лосса (он теперь корректно взвешивает каждый пример)
                loss_cls = self.criterion(logits, y, sample_weights=combined_weights)
            else:
                # Single horizon - обычный loss
                loss_cls_raw = self.criterion(logits, y)
                if self.use_time_weighting or self.use_regime_weighting:
                    # Комбинируем временные веса и веса режимов
                    combined_weights = weights
                    if self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
                        regime_w = self.regime_weights[regime_id].to(weights.device)
                        combined_weights = combined_weights * regime_w
                    loss_cls = (loss_cls_raw * combined_weights).mean()
                else:
                    loss_cls = loss_cls_raw
            
        # 2. Лосс волатильности (MSE) - одинаков для обоих режимов
        loss_vol_raw = self.vol_criterion(vol_pred, vol_target)
        if self.use_time_weighting or self.use_regime_weighting:
            combined_weights = weights
            if self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
                regime_w = self.regime_weights[regime_id].to(weights.device)
                combined_weights = combined_weights * regime_w
            loss_vol = (loss_vol_raw * combined_weights).mean()
        else:
            loss_vol = loss_vol_raw
        
        # Задача 238: Добавляем Curvature Regularization Penalty
        if self.use_curvature_reg:
            from .lit_model import compute_curvature_penalty
            reg_loss = compute_curvature_penalty(
                self.model, 
                x, 
                logits, 
                lambda_=self.curvature_lambda,
                regime_id=regime_id  # Передача текущего режима
            )
            # Логируем reg_loss для мониторинга
            self.log("train_loss_reg", reg_loss, on_step=False, on_epoch=True)
        else:
            reg_loss = 0.0
            
        # 3. Комбинированный Multi-Task Loss (с весами неопределенности)
        # Задача 304 (Fix): Используем формулу Kendall et al. (0.5 * log_var)
        precision_cls = torch.exp(-self.log_var_cls)
        precision_vol = torch.exp(-self.log_var_vol)
        
        loss = precision_cls * loss_cls + 0.5 * self.log_var_cls + \
               precision_vol * loss_vol + 0.5 * self.log_var_vol + \
               reg_loss
        
        # Задача 305-2: Защита от NaN в градиентах
        if not torch.isfinite(loss):
            self.zero_grad()
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        # Логирование
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train_loss_vol", loss_vol, on_step=False, on_epoch=True)
        self.log("weight_cls", precision_cls, on_step=False, on_epoch=True)
        self.log("weight_vol", precision_vol, on_step=False, on_epoch=True)
        
        # Логируем текущий learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        # Получаем текущий momentum из optimizer
        current_momentum = self.optimizers().param_groups[0].get('momentum', 0.0)
        self.log("momentum", current_momentum, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            x, y, vol_target, _, regime_id = batch
        else:
            x, y, vol_target, _ = batch
            regime_id = None
        
        logits, vol_pred = self(x, regime_id=regime_id)
        vol_pred = vol_pred.squeeze(-1)
        
        # Лоссы без весов для валидации
        if self.is_multi_horizon:
            # Multi-horizon: вычисляем loss с маскированием
            # Используем MultiHorizonLoss без sample_weights
            loss_cls = self.criterion(logits, y, sample_weights=None)
        else:
            # Single horizon
            loss_cls = nn.functional.cross_entropy(logits, y)
        
        loss_vol = nn.functional.mse_loss(vol_pred, vol_target)
        mae_vol = nn.functional.l1_loss(vol_pred, vol_target)
        
        # Предсказания (argmax)
        if self.is_multi_horizon:
            # logits: (B, H, 3) -> preds: (B, H)
            preds = torch.argmax(logits, dim=2)
        else:
            # logits: (B, 3) -> preds: (B,)
            preds = torch.argmax(logits, dim=1)
        
        # Накапливаем данные для sklearn
        self.val_y_true.append(y.detach().cpu().numpy())
        
        # Накапливаем regime_ids для логирования метрик по режимам
        if regime_id is not None:
            self.val_regime_ids.append(regime_id.detach().cpu().numpy())
        self.val_y_pred.append(preds.detach().cpu().numpy())
        self.val_logits.append(logits.detach().cpu())
        self.val_vol_true.append(vol_target.detach().cpu().numpy())
        self.val_vol_pred.append(vol_pred.detach().cpu().numpy())
        
        # Логируем основные метрики
        self.log("val_loss", loss_cls, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mse_vol", loss_vol, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae_vol", mae_vol, on_step=False, on_epoch=True)
        
        # Обновляем метрики (только для single horizon)
        if not self.is_multi_horizon:
            # Логируем основные метрики на каждой эпохе (автоматически обновляют объекты)
            self.log("val_mcc", self.mcc(logits, y), prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_f1_macro", self.f1_macro(logits, y), on_step=False, on_epoch=True)
            
            # Обновляем матрицу ошибок и метрики по классам (без логирования на каждом шаге)
            self.conf_matrix.update(logits, y)
            self.precision_per_class.update(logits, y)
            self.recall_per_class.update(logits, y)
        
        return loss_cls + loss_vol

    def on_validation_epoch_end(self):
        # 1. Объединяем накопленные результаты
        y_true = np.concatenate(self.val_y_true)
        y_pred = np.concatenate(self.val_y_pred)
        logits = torch.cat(self.val_logits, dim=0)
        
        # 2. Вычисляем метрики в зависимости от режима (single/multi-horizon)
        if self.is_multi_horizon:
            # Multi-Horizon метрики (Задача 160)
            from .utils import compute_multi_horizon_metrics
            
            # y_true: (n_samples, num_horizons), y_pred: (n_samples, num_horizons)
            metrics = compute_multi_horizon_metrics(y_true, y_pred, self.num_horizons)
            
            # Логируем метрики для каждого горизонта
            for name, value in metrics.items():
                self.log(f"val_{name}", value, logger=True)
            
            # Вычисляем время эпохи
            import time
            epoch_time = time.time() - self.epoch_start_time if hasattr(self, 'epoch_start_time') else 0
            epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s"
            
            # Выводим в консоль
            print(f"\nEpoch {self.current_epoch} ({epoch_time_str}) Multi-Horizon Validation:")
            for h in range(self.num_horizons):
                mcc_h = metrics.get(f"mcc_h{h}", 0.0)
                f1_h = metrics.get(f"f1_h{h}", 0.0)
                samples_h = metrics.get(f"samples_h{h}", 0)
                print(f"  Horizon {h}: MCC={mcc_h:.4f}, F1={f1_h:.4f}, Samples={samples_h}")
            
            # Вычисляем средний MCC по всем горизонтам для мониторинга
            avg_mcc = np.mean([metrics.get(f"mcc_h{h}", 0.0) for h in range(self.num_horizons)])
            self.log("val_mcc", avg_mcc, logger=True, prog_bar=True)
            
            # Калибровка и PR-кривые не поддерживаются для multi-horizon (пропускаем)
            # Можно добавить поддержку позже, если нужно
            
        else:
            # Single Horizon метрики (обычный режим)
            # Передаем веса для логирования (извлекая из criterion)
            class_weights = self.criterion.weight.cpu().numpy() if hasattr(self.criterion, 'weight') and self.criterion.weight is not None else None
            metrics = compute_metrics(y_true, y_pred, class_weights=class_weights)
            
            # Вычисляем метрики калибровки (ECE и MCE)
            y_true_tensor = torch.from_numpy(y_true).long()
            
            # Задача 303-9: Проверка на NaN перед расчетами
            if not torch.isfinite(logits).all():
                print("\n" + "!" * 80)
                print("⚠️  CRITICAL WARNING: NaN or Inf detected in model logits during validation!")
                print("   This indicates extreme numerical instability (exploding gradients).")
                print("   Metrics and visualizations for this epoch will be unreliable.")
                print("!" * 80 + "\n")

            ece, mce, bin_data = self.calibration_metrics.calculate(logits, y_true_tensor)
            
            # Логируем все метрики
            for name, value in metrics.items():
                # Добавляем префикс val_ для единообразия
                self.log(f"val_{name}", value, logger=True)
            
            # Логируем метрики калибровки
            self.log("val_ece", ece, logger=True)
            self.log("val_mce", mce, logger=True)
            
            # Вычисляем время эпохи
            import time
            epoch_time = time.time() - self.epoch_start_time if hasattr(self, 'epoch_start_time') else 0
            epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s"
                
            # Отдельно выводим в консоль ключевые показатели
            print(f"\nEpoch {self.current_epoch} ({epoch_time_str}) Validation: MCC={metrics['mcc']:.4f}, "
                  f"Macro-F1={metrics['f1_macro']:.4f}, "
                  f"ECE={ece:.4f}, MCE={mce:.4f}")
            
            # Сохраняем Reliability Diagram каждые 5 эпох
            if self.current_epoch % 5 == 0:
                # Получаем symbol из trainer (если доступен)
                symbol = getattr(self.trainer, 'symbol', 'UNKNOWN')
                base_path = Path(__file__).parent.parent.parent
                reports_dir = base_path / "reports" / symbol
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = reports_dir / f"reliability_diagram_epoch_{self.current_epoch}.png"
                plot_reliability_diagram(bin_data, ece, mce, str(save_path))
            
            # Логируем поклассовые метрики из torchmetrics
            prec = self.precision_per_class.compute()
            rec = self.recall_per_class.compute()
            
            self.log("val_prec_flat", prec[0])
            self.log("val_rec_flat", rec[0])
            self.log("val_prec_up", prec[1])
            self.log("val_rec_up", rec[1])
            self.log("val_prec_down", prec[2])
            self.log("val_rec_down", rec[2])
            
            # TensorBoard визуализация (Задача 158)
            if self.logger and hasattr(self.logger, 'experiment'):
                writer = self.logger.experiment
                
                # Confusion Matrix через add_figure (каждые 5 эпох)
                if self.current_epoch % 5 == 0:
                    from .utils import plot_confusion_matrix_tensorboard, plot_pr_curves_tensorboard
                    
                    class_names = ["Flat", "Up", "Down"]
                    plot_confusion_matrix_tensorboard(
                        y_true, y_pred, class_names, writer, self.current_epoch
                    )
                    
                    # PR-кривые (используем softmax вероятности из logits)
                    y_pred_probs = torch.softmax(logits, dim=1).numpy()
                    plot_pr_curves_tensorboard(
                        y_true, y_pred_probs, class_names, writer, self.current_epoch
                    )
        
        # 3. Метрики регрессии волатильности (общие для обоих режимов)
        y_vol_true = np.concatenate(self.val_vol_true)
        y_vol_pred = np.concatenate(self.val_vol_pred)
        vol_mse = np.mean((y_vol_true - y_vol_pred)**2)
        vol_mae = np.mean(np.abs(y_vol_true - y_vol_pred))
        
        self.log("val_vol_mse", vol_mse, logger=True)
        self.log("val_vol_mae", vol_mae, logger=True)
        print(f"  Vol-MSE={vol_mse:.6f}, Vol-MAE={vol_mae:.6f}")
        
        # 4. Логируем метрики по режимам (если доступны) - только для single horizon
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
        
        # 5. TensorBoard визуализация (общая для обоих режимов)
        if self.logger and hasattr(self.logger, 'experiment'):
            writer = self.logger.experiment
            
            # Логируем градиенты (каждую эпоху)
            from .utils import log_gradient_norms
            log_gradient_norms(self.model, writer, self.current_epoch)
            
            # Логируем embeddings для TensorBoard Projector (каждые 10 эпох)
            if self.current_epoch % 10 == 0:
                from .utils import log_embeddings
                
                # Получаем параметры из hparams
                tb_embedding_samples = self.hparams.get("tb_embedding_samples", 1000)
                
                # Используем validation dataloader
                val_dataloader = self.trainer.val_dataloaders
                if val_dataloader is not None:
                    log_embeddings(
                        self.model, val_dataloader, writer, 
                        self.current_epoch, max_samples=tb_embedding_samples
                    )
        
        # 7. Сброс накопленных данных и метрик
        self.val_y_true.clear()
        self.val_y_pred.clear()
        self.val_logits.clear()
        self.val_vol_true.clear()
        self.val_vol_pred.clear()
        self.val_regime_ids.clear()
        self.acc.reset()
        self.f1_macro.reset()
        self.mcc.reset()
        self.precision_per_class.reset()
        self.recall_per_class.reset()
        self.conf_matrix.reset()

    def configure_optimizers(self):
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-5)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Получаем параметры scheduler из hparams
        scheduler_type = self.hparams.get("scheduler", "plateau")
        
        if scheduler_type == "none":
            # Без scheduler - константный LR
            return optimizer
        
        elif scheduler_type == "onecycle":
            # OneCycleLR с циклическим momentum
            # Используем estimated_stepping_batches для автоматического расчета total_steps
            total_steps = self.trainer.estimated_stepping_batches
            div_factor = self.hparams.get("div_factor", 25.0)
            final_div_factor = self.hparams.get("final_div_factor", 10000.0)
            pct_start = self.hparams.get("pct_start", 0.3)
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                anneal_strategy='cos',
                cycle_momentum=True,  # Циклический momentum для AdamW
                base_momentum=0.85,
                max_momentum=0.95
            )
            
            # Логируем LR и momentum на каждом шаге
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # OneCycleLR работает на уровне шагов
                    "frequency": 1
                }
            }
        
        elif scheduler_type == "cosine":
            # CosineAnnealingWithWarmup через SequentialLR
            # Рассчитываем количество шагов для warmup (10% от общего числа шагов)
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(0.1 * total_steps)
            cosine_steps = total_steps - warmup_steps
            
            # Warmup: линейное увеличение от lr/div_factor до lr
            div_factor = self.hparams.get("div_factor", 25.0)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0/div_factor,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            # Cosine annealing после warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=lr / 10000.0
            )
            
            # Комбинируем через SequentialLR
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        
        elif scheduler_type == "step":
            # StepLR - простое снижение каждые N эпох
            step_size = self.hparams.get("step_size", 10)
            gamma = self.hparams.get("gamma", 0.5)
            
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        else:  # "plateau" (default)
            # ReduceLROnPlateau - адаптивное снижение при стагнации
            factor = self.hparams.get("plateau_factor", 0.5)
            patience = self.hparams.get("plateau_patience", 5)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=factor,
                patience=patience
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mcc",
                    "interval": "epoch"
                }
            }
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """
        Переопределяем метод для поддержки Adaptive Gradient Clipping (AGC).
        
        Этот метод вызывается автоматически после backward() и перед optimizer.step().
        Поддерживает три режима:
        1. 'none' - без клиппинга
        2. 'norm' - стандартный gradient clipping по глобальной норме
        3. 'agc' - адаптивный послойный клиппинг (AGC)
        """
        clip_mode = self.hparams.get("clip_mode", "none")
        clip_val = self.hparams.get("clip_val", 1.0)
        
        if clip_mode == "none":
            # Без клиппинга
            return
        
        elif clip_mode == "norm":
            # Стандартный gradient clipping по глобальной норме
            # Используем встроенный механизм Lightning
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm="norm"
            )
        
        elif clip_mode == "agc":
            # Adaptive Gradient Clipping (AGC)
            from .utils import adaptive_gradient_clipping, log_grad_stats
            
            # Применяем AGC
            clip_stats = adaptive_gradient_clipping(
                self.model,
                clip_factor=clip_val,
                eps=1e-6
            )
            
            # Логируем статистику градиентов каждые 100 шагов
            if self.global_step % 100 == 0:
                grad_stats = log_grad_stats(
                    self.model,
                    clip_stats=clip_stats,
                    logger=self.logger.experiment if self.logger else None,
                    global_step=self.global_step
                )
                
                # Выводим в консоль для мониторинга
                print(f"\n[Step {self.global_step}] Gradient Stats:")
                print(f"  Clipped: {clip_stats['clipped_pct']:.1f}% ({clip_stats['clipped_count']}/{clip_stats['total_count']})")
                print(f"  Max Ratio (All): {clip_stats['max_ratio']:.4f}")
                print(f"  Max Ratio (Attention): {clip_stats['max_ratio_attention']:.4f}")
                print(f"  Global Grad Norm: {grad_stats['global_grad_norm']:.4f}")

def enable_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()

def objective_seq_len_search(trial, args, base_path, data_path, df, 
                              in_channels, past_returns_lags, num_horizons, horizon_weights, 
                              weights, normalizer, regime_detector, regime_weights, num_regimes, cache_dir=None):
    """
    Optuna objective для поиска оптимальной seq_len (Задача 055).
    
    Параметры:
    - trial: Optuna Trial объект
    - args: аргументы командной строки
    - base_path, data_path: пути к данным
    - df: исходный DataFrame или LazyFrame с данными
    - in_channels: количество входных каналов
    - past_returns_lags: лаги past returns
    - num_horizons: количество горизонтов
    - horizon_weights: веса горизонтов
    - weights: веса классов
    - normalizer: нормализатор
    - regime_detector: детектор режимов
    - regime_weights: веса режимов
    - num_regimes: количество режимов
    - cache_dir: директория кэша для memmap режима
    
    Возвращает:
    - val_mcc: MCC на валидационном наборе
    """
    # Предлагаем seq_len для поиска (Задача 055, пункт 3)
    seq_len = trial.suggest_int("seq_len", 10, 100, step=10)
    print(f"\n[Optuna Trial] Testing seq_len={seq_len}")
    
    # Предлагаем параметры scheduler через Optuna (Задача 093)
    scheduler = trial.suggest_categorical("scheduler", ["onecycle", "plateau", "cosine", "step", "none"])
    
    # Параметры для OneCycleLR
    div_factor = trial.suggest_float("div_factor", 10.0, 40.0, log=True)
    final_div_factor = trial.suggest_float("final_div_factor", 1000.0, 10000.0, log=True)
    pct_start = trial.suggest_float("pct_start", 0.1, 0.5)
    
    # Параметры для ReduceLROnPlateau
    plateau_factor = trial.suggest_float("plateau_factor", 0.1, 0.9)
    plateau_patience = trial.suggest_int("plateau_patience", 2, 10)
    
    # Параметры для StepLR
    step_size = trial.suggest_int("step_size", 5, 30)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    
    print(f"[Optuna Trial] Scheduler: {scheduler}, div_factor={div_factor:.2f}, final_div_factor={final_div_factor:.0f}, pct_start={pct_start:.2f}")
    
    # Пересоздаем датасеты с новой seq_len
    # ВАЖНО: Нужно пересоздать датасеты, так как seq_len влияет на размер окна
    try:
        # Подготовка параметров для временного взвешивания
        time_weighting_params = {}
        if args.use_time_weighting:
            time_weighting_params = {
                'half_life_hours': args.half_life_hours,
                'min_weight': args.min_sample_weight,
                'class_weights': weights
            }
        else:
            time_weighting_params = {
                'half_life_hours': 24.0,
                'min_weight': 1.0,
                'class_weights': None
            }
        
        # Пересоздаем датасет с новой seq_len
        if args.data_mode == "streaming":
            trial_dataset = LOBDataset(
                df,
                seq_len=seq_len,
                n_past_returns=len(past_returns_lags),
                data_mode="streaming",
                is_train=False,
                augment_prob=args.augment_prob,
                use_symmetric_flip=args.use_symmetric_flip,
                volume_jitter_range=args.volume_jitter_range,
                aug_seed=args.aug_seed,
                regime_detector=regime_detector,
                regime_window=1000,
                scaler_type=args.scaler_type,
                winsor_limits=tuple([float(x.strip()) for x in args.winsor_limits.split(",")]),
                **time_weighting_params
            )
        elif args.data_mode == "memmap":
            trial_dataset = LOBDataset(
                df,
                seq_len=seq_len,
                n_past_returns=len(past_returns_lags),
                data_mode="memmap",
                cache_dir=cache_dir,
                is_train=False,
                augment_prob=args.augment_prob,
                use_symmetric_flip=args.use_symmetric_flip,
                volume_jitter_range=args.volume_jitter_range,
                aug_seed=args.aug_seed,
                regime_detector=regime_detector,
                regime_window=1000,
                scaler_type=args.scaler_type,
                winsor_limits=tuple([float(x.strip()) for x in args.winsor_limits.split(",")]),
                **time_weighting_params
            )
        else:
            # Memory mode (по умолчанию)
            trial_dataset = LOBDataset(
                df,
                seq_len=seq_len,
                n_past_returns=len(past_returns_lags),
                data_mode="memory",
                is_train=False,
                augment_prob=args.augment_prob,
                use_symmetric_flip=args.use_symmetric_flip,
                volume_jitter_range=args.volume_jitter_range,
                aug_seed=args.aug_seed,
                regime_detector=regime_detector,
                regime_window=1000,
                scaler_type=args.scaler_type,
                winsor_limits=tuple([float(x.strip()) for x in args.winsor_limits.split(",")]),
                **time_weighting_params
            )
        
        # Разделяем на train/val
        total_len = len(trial_dataset)
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        
        trial_train_ds, trial_val_ds = random_split(trial_dataset, [train_size, val_size])
        
        # Создаем DataLoaders
        num_workers = 2 if args.data_mode == "streaming" else 4
        worker_init_fn = _streaming_worker_init_fn if args.data_mode == "streaming" else None
        
        trial_train_loader = DataLoader(
            trial_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=worker_init_fn
        )
        trial_val_loader = DataLoader(
            trial_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=worker_init_fn
        )
        
    except Exception as e:
        print(f"[Optuna Trial] Error creating dataset with seq_len={seq_len}: {e}")
        raise TrialPruned()
    
    # Создаем модель с текущей seq_len
    from .lit_model import LiTConfig
    
    trial_config = LiTConfig(
        seq_len=seq_len,
        in_channels=in_channels,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        multi_task=True,
        num_horizons=num_horizons,
        use_horizon_embedding=args.use_horizon_embedding
    )
    
    trial_model = LiTModule(
        seq_len=trial_config.seq_len,
        lr=1e-4,
        class_weights=weights,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        activation=trial_config.activation,
        use_time_weighting=args.use_time_weighting,
        use_regime_weighting=(regime_detector is not None),
        regime_weights=regime_weights,
        in_channels=trial_config.in_channels,
        past_returns_lags=past_returns_lags,
        d_model=trial_config.d_model,
        nhead=trial_config.nhead,
        num_layers=trial_config.num_layers,
        dropout=trial_config.dropout,
        multi_task=trial_config.multi_task,
        num_regimes=num_regimes,
        regime_embedding_dim=16,
        num_horizons=num_horizons,
        horizon_weights=horizon_weights,
        use_horizon_embedding=args.use_horizon_embedding,
        scheduler=scheduler,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        pct_start=pct_start,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience,
        step_size=step_size,
        gamma=gamma,
        weight_decay=args.weight_decay,
        clip_mode=args.clip_mode,
        clip_val=args.clip_val,
        tb_hist_freq=args.tb_hist_freq,
        tb_embedding_samples=args.tb_embedding_samples,
        use_curvature_reg=args.use_curvature_reg,
        curvature_lambda=args.curvature_lambda,
        input_noise_std=args.input_noise_std,
        scaler_type=args.scaler_type,
        winsor_limits=list(tuple([float(x.strip()) for x in args.winsor_limits.split(",")])) if args.winsor_limits else None
    )
    
    # Создаем Trainer с EarlyStopping (Задача 055, пункт 3)
    trial_checkpoint_callback = ModelCheckpoint(
        dirpath=base_path / "bots" / args.symbol / "models" / "optuna_checkpoints" / f"seq_len_{seq_len}",
        filename="lit-{epoch:02d}-{val_mcc:.4f}",
        save_top_k=1,
        monitor="val_mcc",
        mode="max"
    )
    
    trial_callbacks = [
        EarlyStopping(monitor="val_mcc", patience=5, mode="max"),  # Ранняя остановка для каждого trial
        trial_checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    trial_logger = TensorBoardLogger(
        f"runs/{args.symbol}/optuna",
        name=f"seq_len_{seq_len}"
    )
    
    trial_trainer = pl.Trainer(
        max_epochs=min(20, args.epochs),  # Ограничиваем эпохи для быстрого поиска
        callbacks=trial_callbacks,
        logger=trial_logger,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        enable_progress_bar=False,  # Отключаем прогресс-бар для чистоты вывода
        log_every_n_steps=100,      # Задача 304: Уменьшаем шаг логирования
        gradient_clip_val=0.5       # Задача 304: Защита от NaN
    )
    
    # Обучаем модель
    try:
        trial_trainer.fit(trial_model, trial_train_loader, trial_val_loader)
    except Exception as e:
        print(f"[Optuna Trial] Training failed for seq_len={seq_len}: {e}")
        raise TrialPruned()
    
    # Вычисляем MCC на валидационном наборе
    trial_model.eval()
    all_preds = []
    all_labels = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_model.to(device)
    
    with torch.no_grad():
        for batch in trial_val_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            regime_id = batch[4].to(device) if len(batch) > 4 else None
            
            # Вызываем forward() модели
            output = trial_model(x, regime_id=regime_id)
            
            # Обрабатываем вывод (может быть кортеж или тензор)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Для multi-horizon берем первый горизонт
            if logits.dim() == 3:
                logits = logits[:, 0, :]  # (batch, 3)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    
    val_preds = torch.cat(all_preds).numpy()
    val_labels_tensor = torch.cat(all_labels)
    
    # Обрабатываем multi-horizon метки
    if val_labels_tensor.dim() == 2:
        # Multi-horizon: берем первый горизонт
        val_labels = val_labels_tensor[:, 0].numpy()
    else:
        # Single horizon
        val_labels = val_labels_tensor.numpy()
    
    val_mcc = matthews_corrcoef(val_labels, val_preds)
    
    # Вычисляем ECE для Optuna (Задача 122, пункт 3)
    # Собираем логиты для валидационного набора
    all_logits = []
    trial_model.eval()
    with torch.no_grad():
        for batch in trial_val_loader:
            x = batch[0].to(device)
            regime_id = batch[4].to(device) if len(batch) > 4 else None
            
            output = trial_model(x, regime_id=regime_id)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            if logits.dim() == 3:
                logits = logits[:, 0, :]
            
            all_logits.append(logits.cpu())
    
    val_logits_tensor = torch.cat(all_logits)
    val_labels_tensor_for_ece = torch.from_numpy(val_labels).long()
    
    # Вычисляем ECE через CalibrationMetrics
    calibration_metrics = CalibrationMetrics(n_bins=15)
    ece, _, _ = calibration_metrics.calculate(val_logits_tensor, val_labels_tensor_for_ece)
    
    # Целевая функция: максимизируем MCC, минимизируем ECE
    # Так как Optuna максимизирует, используем: MCC - (ECE * 0.5)
    score = val_mcc - (ece * 0.5)
    
    print(f"[Optuna Trial] seq_len={seq_len}, val_mcc={val_mcc:.4f}, ece={ece:.4f}, score={score:.4f}")
    
    return score

def update_model_metadata(base_path, symbol, args, winsor_limits, norm_params_path):
    """
    Обновляет или создает metadata.json с параметрами нормализации (Задача 240/056).
    """
    metadata_path = base_path / "bots" / symbol / "models" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Загружаем существующие метаданные или создаем новые
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "metadata_version": "1.1.0",
            "model_name": "LiT",
            "export_timestamp": datetime.datetime.now(UTC).isoformat() + "Z",
        }
    
    # Загружаем сохраненные параметры нормализации
    if norm_params_path.exists():
        with open(norm_params_path, 'r') as f:
            norm_data = json.load(f)
        
        # Извлекаем параметры для метаданных
        if isinstance(norm_data, dict) and "params" in norm_data:
            params = norm_data["params"]
        else:
            params = norm_data
            
        metadata["normalization"] = {
            "scaler_type": args.scaler_type,
            "winsor_limits": winsor_limits,
            "params": params
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[{symbol}] Metadata updated with normalization params at {metadata_path}")

def train():
    parser = argparse.ArgumentParser(description="Train LiT model on LOB data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to train on")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--horizon", type=int, default=100, help="Prediction horizon for labels (single horizon, deprecated)")
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated list of horizons for multi-horizon prediction (e.g., '10,50,100')")
    parser.add_argument("--horizon_weights", type=str, default=None, help="Comma-separated list of weights for each horizon (e.g., '0.4,0.3,0.3')")
    parser.add_argument("--use_horizon_embedding", action="store_true", help="Use Horizon Embedding instead of separate heads")
    parser.add_argument("--threshold", type=float, default=0.0005, help="Статический порог доходности (0.0005 = 0.05%)")
    parser.add_argument("--class_weight_smooth", type=float, default=1.0, help="Smoothing for class weights calculation")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for CrossEntropyLoss")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["ce", "focal"], help="Loss function type")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    parser.add_argument("--past_returns_lags", type=str, default="10,50,100", help="Comma-separated list of lags for past returns (e.g., '10,50,100')")
    parser.add_argument("--activation", type=str, default="gelu_exact", choices=["relu", "gelu_exact", "gelu_tanh", "silu"], help="Activation function type")
    
    # Параметры архитектуры модели (для режима train - teacher модель)
    parser.add_argument("--d_model", type=int, default=64, help="Model embedding dimension (d_model)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Параметры загрузки данных (Задача 094)
    parser.add_argument("--data_mode", type=str, default="memory", choices=["memory", "streaming", "memmap"], help="Data loading mode: memory (default), streaming (lazy), or memmap (binary cache)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for memmap mode (required for memmap)")
    
    # Параметры LR Scheduler
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["onecycle", "plateau", "cosine", "step", "none"], help="Learning rate scheduler type")
    parser.add_argument("--div_factor", type=float, default=25.0, help="Initial LR divisor for OneCycle/Cosine warmup")
    parser.add_argument("--final_div_factor", type=float, default=10000.0, help="Final LR divisor for OneCycle")
    parser.add_argument("--pct_start", type=float, default=0.3, help="Percentage of cycle spent increasing LR in OneCycle")
    parser.add_argument("--plateau_factor", type=float, default=0.5, help="Factor for ReduceLROnPlateau")
    parser.add_argument("--plateau_patience", type=int, default=5, help="Patience for ReduceLROnPlateau")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for StepLR scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer")
    
    # Параметры Adaptive Gradient Clipping (Задача 154)
    parser.add_argument("--clip_mode", type=str, default="none", choices=["none", "norm", "agc"], help="Gradient clipping mode: none (no clipping), norm (global norm), agc (adaptive per-layer)")
    parser.add_argument("--clip_val", type=float, default=0.01, help="Clipping threshold (for norm: usually 1.0, for agc: 0.01-0.1)")
    
    # Параметры временного взвешивания (Задача 123)
    parser.add_argument("--use_time_weighting", action="store_true", help="Enable time-decay weighting")
    parser.add_argument("--half_life_hours", type=float, default=24.0, help="Weight decay half-life in hours")
    parser.add_argument("--min_sample_weight", type=float, default=0.1, help="Minimum weight for old samples")
    
    # Параметры аугментации данных (Задача 124)
    parser.add_argument("--augment_prob", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--use_symmetric_flip", action="store_true", help="Enable Bid/Ask flipping with label reversal")
    parser.add_argument("--volume_jitter_range", type=float, default=0.1, help="Max relative volume change (e.g. 0.1 for +/- 10%)")
    parser.add_argument("--aug_seed", type=int, default=42, help="Seed for reproducible augmentation")
    
    # Задача 307.3: Отключаем оверсемплинг данных (оставляем только веса в FocalLoss для стабильности I/O)
    parser.add_argument("--balance_method", type=str, default="none", choices=["none", "smote", "bgmm", "adasyn"], help="Dataset balancing method (deprecated in 307)")
    parser.add_argument("--balance_ratio", type=float, default=0.5, help="Target ratio for minority classes relative to majority class")
    
    # Параметры Knowledge Distillation (Задача 151)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "distill", "cv"], help="Training mode: train (normal), distill (knowledge distillation), or cv (purged k-fold cross-validation)")
    parser.add_argument("--teacher_path", type=str, default=None, help="Path to teacher model checkpoint (required for distill mode)")
    parser.add_argument("--alpha", type=float, default=0.9, help="Weight for soft loss in distillation (0.9 recommended for LOB data)")
    parser.add_argument("--temperature", type=float, default=3.0, help="Temperature for softening logits in distillation")
    parser.add_argument("--student_d_model", type=int, default=64, help="Student model d_model (embedding dimension)")
    parser.add_argument("--student_nhead", type=int, default=4, help="Student model number of attention heads")
    parser.add_argument("--student_num_layers", type=int, default=2, help="Student model number of transformer layers")
    
    # Параметры Purged K-Fold Cross-Validation (Задача 153)
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation (cv mode)")
    parser.add_argument("--purge_buffer_events", type=int, default=100, help="Number of events to purge before validation fold")
    parser.add_argument("--embargo_buffer_events", type=int, default=50, help="Number of events to embargo after validation fold")
    
    # Параметры Optuna Pruning (Задача 156) - используются только если передан trial объект
    parser.add_argument("--pruner_type", type=str, default="median", choices=["median", "hyperband", "patience"], 
                        help="Pruner type for Optuna: median (baseline), hyperband (resource budgeting), patience (noise tolerance)")
    parser.add_argument("--min_resource", type=int, default=1, help="Minimum resource (epochs) for HyperbandPruner")
    parser.add_argument("--max_resource", type=int, default=20, help="Maximum resource (epochs) for HyperbandPruner")
    parser.add_argument("--n_startup_trials", type=int, default=20, help="Number of startup trials before pruning starts (min 20 for LOB data)")
    parser.add_argument("--n_warmup_steps", type=int, default=25, help="Number of warmup steps (epochs) before first pruning check (min 25 for transformers)")
    parser.add_argument("--patience", type=int, default=3, help="Patience for PatientPruner (steps without improvement)")
    
    # Параметры TensorBoard визуализации (Задача 158)
    parser.add_argument("--tb_dir", type=str, default=None, help="TensorBoard log directory (default: runs/SYMBOL/)")
    parser.add_argument("--tb_hist_freq", type=int, default=10, help="Frequency of writing full histograms (every N epochs)")
    parser.add_argument("--tb_embedding_samples", type=int, default=1000, help="Max samples for TensorBoard Projector (default: 1000)")
    
    # Параметры Model Pruning (Задача 159)
    parser.add_argument("--prune_mode", type=str, default="none", choices=["none", "unstructured", "structured_2_4"], 
                        help="Pruning mode: none (no pruning), unstructured (magnitude-based), structured_2_4 (2:4 sparsity for NVIDIA)")
    parser.add_argument("--prune_amount", type=float, default=0.5, help="Target sparsity level (0.0-0.6 recommended for LOB, default: 0.5)")
    parser.add_argument("--prune_iterations", type=int, default=3, help="Number of prune-and-finetune iterations (default: 3)")
    parser.add_argument("--prune_finetune_epochs", type=int, default=2, help="Epochs of fine-tuning after each pruning iteration (default: 2)")
    
    # Параметры Micro-Trades Imbalance (Задача 236)
    parser.add_argument("--trade_imb_windows", type=str, nargs="+", default=["1s", "5s", "15s", "60s"], help="Windows for trade imbalance aggregation")
    parser.add_argument("--trade_imb_agg", type=str, default="vol", choices=["vol", "count"], help="Aggregation type for imbalance: vol (volume) or count (number of trades)")
    parser.add_argument("--trade_noise_filter_pct", type=float, default=0.05, help="Noise filter percentage (trades smaller than this % of median size are excluded)")
    
    # Параметры Curvature Regularization (Задача 238)
    parser.add_argument("--use_curvature_reg", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable curvature regularization penalty")
    parser.add_argument("--curvature_lambda", type=float, default=1e-4, help="Curvature penalty coefficient (recommended: 1e-4 to 1e-3)")
    parser.add_argument("--input_noise_std", type=float, default=0.005, help="Standard deviation for input noise injection during training")
    
    # Параметры Robust Scaling (Задача 240, 306)
    parser.add_argument("--scaler_type", type=str, default="robust", choices=["zscore", "robust", "winsor_robust"], help="Scaler type: robust (default, median/IQR), zscore, or winsor_robust")
    parser.add_argument("--winsor_limits", type=str, default="0.01,0.99", help="Winsorization limits as comma-separated floats (e.g., '0.01,0.99' for 1st and 99th percentiles)")
    
    # Параметры Optuna поиска seq_len (Задача 055)
    parser.add_argument("--optuna_seq_len_search", action="store_true", help="Enable Optuna hyperparameter search for seq_len")
    parser.add_argument("--optuna_n_trials", type=int, default=10, help="Number of Optuna trials for seq_len search (default: 10)")
    parser.add_argument("--optuna_pruner", type=str, default="median", choices=["median", "hyperband", "patient"], help="Optuna pruner type")
    
    args = parser.parse_args()
    
    # Парсим winsor_limits из строки (Задача 240)
    try:
        winsor_limits_list = [float(x.strip()) for x in args.winsor_limits.split(",")]
        if len(winsor_limits_list) != 2:
            raise ValueError(f"winsor_limits must have exactly 2 values, got {len(winsor_limits_list)}")
        winsor_limits = tuple(winsor_limits_list)
    except ValueError as e:
        raise ValueError(f"Invalid --winsor_limits format: {e}. Expected comma-separated floats like '0.01,0.99'")
    
    print(f"Scaler configuration: type={args.scaler_type}, winsor_limits={winsor_limits}")
    
    # Валидация аргументов прунинга (Задача 159)
    if args.prune_mode != "none":
        if args.prune_amount < 0.0 or args.prune_amount > 0.6:
            raise ValueError(f"--prune_amount должен быть в диапазоне [0.0, 0.6] для LOB данных, получено: {args.prune_amount}")
        if args.prune_iterations < 1:
            raise ValueError(f"--prune_iterations должен быть >= 1, получено: {args.prune_iterations}")
        if args.prune_finetune_epochs < 1:
            raise ValueError(f"--prune_finetune_epochs должен быть >= 1, получено: {args.prune_finetune_epochs}")
        if args.mode == "cv":
            raise ValueError("Pruning не поддерживается в режиме cross-validation (cv). Используйте режим 'train' или 'distill'.")
    
    # Парсим лаги из строки
    past_returns_lags = [int(x.strip()) for x in args.past_returns_lags.split(",")]
    n_past_returns = len(past_returns_lags)
    
    # В плане 306 итоговая структура x_final всегда 6 каналов
    in_channels = 6
    
    print(f"Using past returns lags: {past_returns_lags}")
    print(f"Total input channels: {in_channels} (Price, Vol, Imb, OFI, VIB, PastRet)")
    print(f"Data loading mode: {args.data_mode}")

    # 1. Фиксируем seed для воспроизводимости
    pl.seed_everything(42)

    # 2. Настройка путей
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "bots" / args.symbol / "data" / "raw"
    norm_params_path = base_path / "bots" / args.symbol / "models" / "norm_params.json"
    checkpoint_dir = base_path / "bots" / args.symbol / "models" / "checkpoints"
    
    # Путь для кэша (если используется memmap)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = base_path / "bots" / args.symbol / "models" / "cache"

    # 3. Валидация ресурсов (Задача 094)
    if args.data_mode == "memory":
        # Проверяем доступную RAM
        mem = psutil.virtual_memory()
        available_ram_gb = mem.available / (1024 ** 3)
        
        # Оцениваем размер датасета (грубая оценка)
        pattern = f"{args.symbol}_*.parquet"
        files = list(data_path.glob(pattern))
        if files:
            # Считаем размер файлов на диске
            total_size_gb = sum(f.stat().st_size for f in files) / (1024 ** 3)
            
            # ИСПРАВЛЕНИЕ: Более точная оценка с учетом скользящего окна
            # Parquet сжат примерно в 3-5 раз, поэтому в памяти будет больше
            # Дополнительно учитываем seq_len для скользящего окна (дублирование данных)
            # Формула: compressed_size * decompression_factor * window_overhead
            decompression_factor = 4  # Parquet -> RAM
            window_overhead = args.seq_len / 10  # Скользящее окно увеличивает размер
            estimated_ram_gb = total_size_gb * decompression_factor * (1 + window_overhead / 100)
            
            if estimated_ram_gb > available_ram_gb * 0.7:
                print(f"\n⚠️  WARNING: Dataset size (~{estimated_ram_gb:.2f} GB) may exceed available RAM ({available_ram_gb:.2f} GB)")
                print(f"   Estimated breakdown:")
                print(f"   - Compressed Parquet: {total_size_gb:.2f} GB")
                print(f"   - Decompressed in RAM: {total_size_gb * decompression_factor:.2f} GB")
                print(f"   - With sliding window (seq_len={args.seq_len}): {estimated_ram_gb:.2f} GB")
                print(f"   Consider using --data_mode streaming or --data_mode memmap for large datasets")
                print(f"   Continuing with 'memory' mode as requested...\n")

    # 4. Загрузка и подготовка данных
    print(f"Loading data for {args.symbol} from {data_path}...")
    loader = LOBDataLoader(str(data_path), args.symbol)
    
    if args.data_mode == "streaming":
        # Для streaming режима используем lazy загрузку
        df = loader.load_data(lazy=True)
    else:
        # Для memory и memmap загружаем в память
        df = loader.load_data(lazy=False)

    # Задача 236: Загрузка публичных сделок и расчет trade imbalance — ОТЛОЖЕНО
    # Пока Trades не используем, всё обучение проходит только на Orderbook. 
    # В будущем, если нужно будет Trades, раскомментируем.
    """
    print("Loading trades data...")
    df_trades = loader.load_trades(lazy=False)
    
    if not df_trades.is_empty():
        print("Computing trade imbalance features...")
        from .dataset import compute_trade_imbalance
        
        # Параметры из args (Задача 236)
        trade_imb_windows = args.trade_imb_windows
        trade_imb_agg = args.trade_imb_agg
        trade_noise_filter_pct = args.trade_noise_filter_pct
        
        if args.data_mode == "streaming":
            # Для LazyFrame сначала собираем в память для compute_trade_imbalance
            df_collected = df.collect()
            df_collected = compute_trade_imbalance(
                df_collected, 
                df_trades, 
                windows=trade_imb_windows,
                agg_type=trade_imb_agg,
                noise_filter_pct=trade_noise_filter_pct
            )
            df = df_collected.lazy()
        else:
            df = compute_trade_imbalance(
                df, 
                df_trades, 
                windows=trade_imb_windows,
                agg_type=trade_imb_agg,
                noise_filter_pct=trade_noise_filter_pct
            )
        print(f"Added {len(trade_imb_windows)} trade imbalance features")
    else:
        print("No trades data found, skipping trade imbalance features")
    """

    # Генерация признаков
    print("Engineering features...")
    fe = FeatureEngineer(n_levels=50)
    
    if args.data_mode == "streaming":
        # Для LazyFrame применяем трансформации лениво
        df = fe.transform(df)
    else:
        df = fe.transform(df)

    # Разметка
    print("Adding labels...")
    
    # Определяем horizons (multi-horizon или single)
    if args.horizons:
        horizons = [int(h.strip()) for h in args.horizons.split(',')]
        print(f"Using multi-horizon prediction: {horizons}")
        num_horizons = len(horizons)
        
        # Парсим horizon_weights
        if args.horizon_weights:
            horizon_weights = [float(w.strip()) for w in args.horizon_weights.split(',')]
            if len(horizon_weights) != num_horizons:
                raise ValueError(f"Number of horizon_weights ({len(horizon_weights)}) must match number of horizons ({num_horizons})")
            # Нормализуем веса (сумма = 1.0)
            total_weight = sum(horizon_weights)
            horizon_weights = [w / total_weight for w in horizon_weights]
            print(f"Using horizon weights (normalized): {horizon_weights}")
        else:
            # Равные веса по умолчанию
            horizon_weights = [1.0 / num_horizons] * num_horizons
            print(f"Using equal horizon weights: {horizon_weights}")
    else:
        horizons = args.horizon
        num_horizons = 1
        horizon_weights = None
        print(f"Using single horizon prediction: {horizons}")
    
    labeler = Labeler(
        horizon=horizons, 
        threshold=args.threshold, 
        dynamic_threshold=False  # Выключаем авто-подбор, переходим на ручное управление
    )
    
    if args.data_mode == "streaming":
        df = labeler.add_labels(df)
    else:
        df = labeler.add_labels(df)

    # 5. Инициализация Normalizer (fit будет позже на train set)
    print("Initializing normalizer...")
    normalizer = Normalizer(norm_params_path)
    
    # 5.5. Обучение RegimeDetector (Задача 155)
    regime_detector = None
    regime_weights = None
    num_regimes = 0
    
    # --- ВРЕМЕННО ОТКЛЮЧЕНО (ЗАДАЧА 155 ПРИОСТАНОВЛЕНА) ---
    # if args.data_mode != "streaming":  # Regime detection требует полных данных в памяти
    #     print("\n[Regime Detection] Training HMM for market regime identification...")
    #     from .regime import RegimeDetector, optimize_n_components_optuna
    #     from .dataset import compute_regime_features
    #     
    #     # Вычисляем признаки режима
    #     regime_features = compute_regime_features(df, window=1000)
    #     
    #     # Оптимизируем количество состояний через Optuna (опционально)
    #     try:
    #         best_n_components, best_score = optimize_n_components_optuna(
    #             regime_features, 
    #             min_components=2, 
    #             max_components=6, 
    #             n_trials=10
    #         )
    #         print(f"[Regime Detection] Optimal number of regimes: {best_n_components} (Silhouette Score: {best_score:.4f})")
    #     except Exception as e:
    #         print(f"[Regime Detection] Optuna optimization failed: {e}. Using default n_components=3")
    #         best_n_components = 3
    #     
    #     # Обучаем RegimeDetector
    #     regime_detector = RegimeDetector(n_components=best_n_components)
    #     regime_detector.fit(regime_features)
    #     
    #     # Получаем распределение режимов
    #     regime_distribution = regime_detector.get_regime_distribution(regime_features)
    #     print(f"[Regime Detection] Regime distribution: {regime_distribution}")
    #     
    #     # Вычисляем веса режимов (обратно пропорционально частоте)
    #     total_samples = len(regime_features)
    #     regime_weights = total_samples / (regime_distribution + 1e-8)
    #     regime_weights = regime_weights / regime_weights.sum() * best_n_components
    #     print(f"[Regime Detection] Regime weights: {regime_weights}")
    #     
    #     num_regimes = best_n_components
    #     
    #     # Сохраняем параметры HMM
    #     regime_config_path = base_path / "bots" / args.symbol / "model" / "regime_config.json"
    #     regime_detector.save(str(regime_config_path))
    #     print(f"[Regime Detection] Saved regime config to {regime_config_path}")
    # else:
    #     print("[Regime Detection] Skipped for streaming mode (requires full data in memory)")
    # ---------------------------------------------------

    
    # 6. Создание Dataset и хронологическое разделение (70/15/15)
    print(f"Creating dataset in '{args.data_mode}' mode (raw features)...")
    
    # Подготовка параметров для временного взвешивания
    time_weighting_params = {}
    if args.use_time_weighting:
        time_weighting_params = {
            'half_life_hours': args.half_life_hours,
            'min_weight': args.min_sample_weight,
            'class_weights': weights  # Передаем веса классов для интеграции
        }
        print(f"Time weighting enabled: half_life={args.half_life_hours}h, min_weight={args.min_sample_weight}")
    else:
        # Если временное взвешивание отключено, все веса = 1.0
        time_weighting_params = {
            'half_life_hours': 24.0,  # Значение по умолчанию (не используется)
            'min_weight': 1.0,  # Все веса = 1.0
            'class_weights': None  # Не используем веса классов в датасете
        }
    
    if args.data_mode == "streaming":
        # Для streaming создаем один датасет и делим через Subset
        full_dataset = LOBDataset(
            df, 
            seq_len=args.seq_len, 
            n_past_returns=n_past_returns,
            past_returns_lags=past_returns_lags,  # Задача 091
            data_mode="streaming",
            is_train=False,  # Будет переопределено для train_ds
            augment_prob=args.augment_prob,
            use_symmetric_flip=args.use_symmetric_flip,
            volume_jitter_range=args.volume_jitter_range,
            aug_seed=args.aug_seed,
            regime_detector=regime_detector,
            regime_window=1000,
            scaler_type=args.scaler_type,  # Задача 240
            winsor_limits=winsor_limits,  # Задача 240
            **time_weighting_params
        )
    elif args.data_mode == "memmap":
        # Для memmap создаем кэш
        full_dataset = LOBDataset(
            df,
            seq_len=args.seq_len,
            n_past_returns=n_past_returns,
            past_returns_lags=past_returns_lags,  # Задача 091
            data_mode="memmap",
            cache_dir=cache_dir,
            is_train=False,  # Будет переопределено для train_ds
            augment_prob=args.augment_prob,
            use_symmetric_flip=args.use_symmetric_flip,
            volume_jitter_range=args.volume_jitter_range,
            aug_seed=args.aug_seed,
            regime_detector=regime_detector,
            regime_window=1000,
            scaler_type=args.scaler_type,  # Задача 240
            winsor_limits=winsor_limits,  # Задача 240
            **time_weighting_params
        )
    else:
        # Memory mode (по умолчанию)
        full_dataset = LOBDataset(
            df, 
            seq_len=args.seq_len, 
            n_past_returns=n_past_returns,
            past_returns_lags=past_returns_lags,  # Задача 091
            data_mode="memory",
            is_train=False,  # Будет переопределено для train_ds
            augment_prob=args.augment_prob,
            use_symmetric_flip=args.use_symmetric_flip,
            volume_jitter_range=args.volume_jitter_range,
            aug_seed=args.aug_seed,
            regime_detector=regime_detector,
            regime_window=1000,
            scaler_type=args.scaler_type,  # Задача 240
            winsor_limits=winsor_limits,  # Задача 240
            **time_weighting_params
        )
    
    # Задача 306.2.4: Остановка обучения при обнаружении NaN во входных данных
    if args.data_mode != "streaming":
        if np.isnan(full_dataset.features).any():
            raise ValueError("КРИТИЧНО: Входящие features содержат NaN строки для запуска обучения!")
    
    # Проверка данных на NaN перед обучением (Sample-based check)
    print("\nПроверка данных на NaN (sampling)...")
    nan_check_samples = min(100, len(full_dataset))
    nan_found = False
    
    for i in range(0, nan_check_samples, 10):
        try:
            sample = full_dataset[i]
            x, y, vol_target, weight = sample[:4]  # Первые 4 элемента
            
            if torch.isnan(x).any():
                print(f"⚠️  WARNING: NaN обнаружен в признаках (x) на индексе {i}")
                nan_found = True
            if torch.isnan(torch.tensor(y)).any():
                print(f"⚠️  WARNING: NaN обнаружен в метках (y) на индексе {i}")
                nan_found = True
            if torch.isnan(torch.tensor(vol_target)).any():
                print(f"⚠️  WARNING: NaN обнаружен в целевой волатильности (vol_target) на индексе {i}")
                nan_found = True
        except Exception as e:
            print(f"⚠️  WARNING: Ошибка при проверке индекса {i}: {e}")
            nan_found = True
    
    if nan_found:
        print("\n" + "!" * 80)
        print("⚠️  CRITICAL WARNING: Обнаружены NaN значения в данных!")
        print("   Это может привести к нестабильности обучения и NaN в метриках.")
        print("   Рекомендации:")
        print("   1. Проверьте качество исходных Parquet файлов")
        print("   2. Проверьте параметры нормализации (scaler_type, winsor_limits)")
        print("   3. Проверьте параметры feature engineering")
        print("!" * 80 + "\n")
    else:
        print(f"✓ Проверка завершена: NaN не обнаружены в {nan_check_samples} проверенных примерах")
    
    # Хронологическое разделение 70/15/15 (Train/Val/Test)
    total_len = len(full_dataset)
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size

    # Хронологические индексы (0-70%, 70-85%, 85-100%)
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_len))

    from torch.utils.data import Subset

    # Используем TrainSubset для безопасной аугментации в обучении
    train_ds = TrainSubset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)

    # Верификация разделения
    print(f"\nChronological split verification:")
    print(f"  Train: indices {train_indices[0]}-{train_indices[-1]} ({len(train_ds)} samples, {len(train_ds)/total_len*100:.1f}%)")
    print(f"  Val:   indices {val_indices[0]}-{val_indices[-1]} ({len(val_ds)} samples, {len(val_ds)/total_len*100:.1f}%)")
    print(f"  Test:  indices {test_indices[0]}-{test_indices[-1]} ({len(test_ds)} samples, {len(test_ds)/total_len*100:.1f}%)")
    
    # 7. Оверсэмплинг и нормализация тренировочного набора (Задача 127, оптимизация 303)
    # Задача 307: Отключаем оверсемплинг (SMOTE/BGMM), так как он конфликтует с весами классов и I/O
    if False: # args.balance_method != "none":
        if args.data_mode == "streaming":
            print("\n⚠️  WARNING: Oversampling is not supported in 'streaming' mode. Skipping balancing.")
            # Для streaming режима все равно нужен fit
            sample_df = df.head(100000).collect(engine="streaming")
            normalizer.fit(sample_df, winsor_limits=winsor_limits)
            normalizer.save(scaler_type=args.scaler_type, winsor_limits=winsor_limits)
            update_model_metadata(base_path, args.symbol, args, winsor_limits, norm_params_path)
        else:
            print(f"\nApplying oversampling to training set ({args.balance_method}, ratio={args.balance_ratio})...")
            print("Using batch processing to optimize memory usage...")
            
            train_indices = train_ds.indices
            seq_len = full_dataset.seq_len
            
            # Шаг 1: Глобальный расчет sampling_strategy
            print("Step 1/4: Computing global sampling strategy...")
            train_labels_all = full_dataset.labels[train_indices + seq_len - 1]
            global_counts = np.bincount(train_labels_all)
            if len(global_counts) < 3:
                full_counts = np.zeros(3, dtype=int)
                full_counts[:len(global_counts)] = global_counts
                global_counts = full_counts
            
            maj_class = np.argmax(global_counts)
            target_count = int(global_counts[maj_class] * args.balance_ratio)
            sampling_strategy = {
                1: max(global_counts[1], target_count), 
                2: max(global_counts[2], target_count)
            }
            print(f"Global class distribution: {global_counts}, Target strategy: {sampling_strategy}")
            
            # Шаг 2: Fit нормализатора на 2D сырых данных
            print("Step 2/4: Fitting normalizer on raw 2D training data...")
            train_features_2d = full_dataset.features[train_indices]
            # Задача 306.4.4: Используем список колонок строго из датасета
            feat_cols = full_dataset.feat_cols.copy()
            
            print(f"Features dimension check: {train_features_2d.shape[1]} vs {len(feat_cols)}")
            normalizer.fit(train_features_2d, feature_names=feat_cols, winsor_limits=winsor_limits)
            normalizer.save(scaler_type=args.scaler_type, winsor_limits=winsor_limits)
            update_model_metadata(base_path, args.symbol, args, winsor_limits, norm_params_path)
            print(f"✓ Normalizer fitted on {len(train_features_2d)} samples")
            
            # Шаг 3: Батчевая обработка с записью в файл
            print("Step 3/4: Batch processing (normalize -> balance -> write to disk)...")
            BATCH_SIZE = 50000
            n_features = train_features_2d.shape[1]
            
            # Создаем временные файлы для записи
            feat_bin_path = os.path.join(base_path, args.symbol, "model", "balanced_features.bin")
            lab_bin_path = os.path.join(base_path, args.symbol, "model", "balanced_labels.bin")
            
            total_balanced_samples = 0
            with open(feat_bin_path, 'wb') as f_feat, open(lab_bin_path, 'wb') as f_lab:
                for i in range(0, len(train_indices), BATCH_SIZE):
                    batch_indices = train_indices[i : i + BATCH_SIZE]
                    
                    # А) Сборка 3D батча + метки батча
                    batch_3d_list = []
                    for idx in batch_indices:
                        window = full_dataset.features[idx : idx + seq_len]
                        batch_3d_list.append(window)
                    batch_3d = np.stack(batch_3d_list, axis=0)
                    batch_labels = full_dataset.labels[batch_indices + seq_len - 1]
                    
                    # Б) Нормализация батча
                    batch_3d_norm = normalizer.transform(batch_3d)
                    
                    # В) Балансировка батча с глобальной стратегией
                    b_feat, b_lab = balance_dataset(
                        batch_3d_norm, 
                        batch_labels, 
                        method=args.balance_method, 
                        sampling_strategy=sampling_strategy
                    )
                    
                    # Г) Запись в файл (append)
                    f_feat.write(b_feat.astype('float32').tobytes())
                    f_lab.write(b_lab.astype('int64').tobytes())
                    total_balanced_samples += len(b_lab)
                    
                    print(f"  Processed batch {i//BATCH_SIZE + 1}: {len(batch_indices)} -> {len(b_lab)} samples (total: {total_balanced_samples})")
            
            print(f"✓ Batch processing complete: {len(train_indices)} -> {total_balanced_samples} samples")
            
            # Шаг 4: Подключаем memmap к результату
            print("Step 4/4: Creating memmap dataset...")
            features_res = np.memmap(feat_bin_path, dtype='float32', mode='r', 
                                     shape=(total_balanced_samples, seq_len, n_features))
            labels_res = np.memmap(lab_bin_path, dtype='int64', mode='r', 
                                   shape=(total_balanced_samples,))
            
            # Создаем новый тренировочный датасет
            class BalancedTrainDataset(Dataset):
                def __init__(self, features, labels, original_ds):
                    self.features = features  # memmap array
                    self.labels = labels      # memmap array
                    self.original_ds = original_ds
                    self.is_train = True 
                    
                def __len__(self):
                    return len(self.labels)
                    
                def __getitem__(self, idx):
                    # Признаки уже нормализованы в features (memmap)
                    x = torch.from_numpy(self.features[idx].copy()).float()
                    y = torch.from_numpy(np.array(self.labels[idx])).long()
                    
                    # Применяем аугментацию если нужно
                    if self.is_train and torch.rand(1).item() < self.original_ds.augment_prob:
                        x_aug, y_aug = x.clone(), y
                        if self.original_ds.use_symmetric_flip and torch.rand(1).item() < 0.5:
                             x_aug, y_aug = apply_symmetric_flip(
                                 x_aug, y_aug, 
                                 self.original_ds.price_cols, 
                                 self.original_ds.ask_cols, 
                                 self.original_ds.bid_cols
                             )
                        if self.original_ds.volume_jitter_range > 0:
                             x_aug = apply_volume_jitter(
                                 x_aug, 
                                 self.original_ds.volume_jitter_range, 
                                 self.original_ds.vol_cols, 
                                 self.original_ds.generator
                             )
                        # Проверка консистентности: Best Bid < Best Ask
                        if x_aug[0, 2] < x_aug[0, 0]:
                            x, y = x_aug, y_aug
                    
                    # Решейп и бродкаст (Error A - исправлено по образу LOBDataset)
                    # Используем внутреннюю логику оригинального датасета для сборки 6 каналов
                    # Параметры v, w, regime_id для сбалансированного датасета фиктивны
                    x_final, _, _, _, _ = self.original_ds._process_sample(
                        x.numpy(), y.item(), 0.0, 1.0, 0
                    )
                    
                    return x_final, y, torch.tensor(0.0).float(), torch.tensor(1.0).float(), torch.tensor(0).long()
            
            # Заменяем train_ds
            train_ds = BalancedTrainDataset(features_res, labels_res, full_dataset)
            print(f"✓ Training set balanced and normalized: {len(train_labels_all)} -> {len(train_ds)} samples")

    else:
        # Если балансировка не используется, обучаем нормализатор на обычном тренировочном наборе
        if args.data_mode != "streaming":
            print("\nFitting normalizer on original training set...")
            train_indices = train_ds.indices
            
            # Оптимизация памяти: обучаем на 2D данных вместо 3D окон
            train_features_2d = full_dataset.features[train_indices]
            
            # Задача 306.4.4: Используем список колонок строго из датасета
            feat_cols = full_dataset.feat_cols.copy()
            
            print(f"Features dimension check: {train_features_2d.shape[1]} vs {len(feat_cols)}")
            normalizer.fit(train_features_2d, feature_names=feat_cols, winsor_limits=winsor_limits)
            normalizer.save(scaler_type=args.scaler_type, winsor_limits=winsor_limits)
            update_model_metadata(base_path, args.symbol, args, winsor_limits, norm_params_path)
            print(f"✓ Normalizer fitted on {len(train_features_2d)} samples")
        else:
            # Для streaming обучаем на сэмпе
            sample_df = df.head(100000).collect(engine="streaming")
            normalizer.fit(sample_df, winsor_limits=winsor_limits)
            normalizer.save(scaler_type=args.scaler_type, winsor_limits=winsor_limits)
            update_model_metadata(base_path, args.symbol, args, winsor_limits, norm_params_path)

    # 8. Финальная нормализация всех данных
    # Применяем параметры Z-score, вычисленные на train set, ко всему набору
    if args.data_mode != "streaming":
        print("Applying normalization to all data subsets (preventing leakage)...")
        # Нормализуем исходные данные (features)
        full_dataset.features = normalizer.transform(full_dataset.features)
        # Примечание: если была балансировка, train_ds уже заменен на BalancedTrainDataset, 
        # который работает со своей копией данных. Валидация и тест берут данные из full_dataset.
    else:
        # Для streaming нормализация уже "встроена" в LazyFrame (нужно убедиться в этом)
        # В текущей реализации streaming в train.py нормализация применяется к LazyFrame df
        df = normalizer.transform(df)
        # Но нам нужно пересоздать датасет, если мы изменили df после его создания?
        # В Lightning/streaming это сложнее. Для простоты считаем, что для streaming 
        # нормализация применяется один раз.
        pass

    print(f"Dataset split (Chronological): Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    if args.use_symmetric_flip or args.volume_jitter_range > 0:
        print(f"Augmentation enabled for training: flip={args.use_symmetric_flip}, jitter={args.volume_jitter_range}, prob={args.augment_prob}")

    # 8. DataLoaders
    # Для streaming режима используем меньше воркеров для thread safety
    num_workers = 2 if args.data_mode == "streaming" else 4
    
    # Функция инициализации воркеров для streaming режима
    worker_init_fn = _streaming_worker_init_fn if args.data_mode == "streaming" else None
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn
    )

    # 8. Расчет весов классов на основе тренировочного набора
    print("Calculating class weights from training set...")
    
    # ИСПРАВЛЕНИЕ: Используем get_class_distribution вместо цикла по всем элементам
    if args.data_mode == "streaming":
        # Для streaming режима нужно считать только на тренировочной части
        # Создаем временный LazyFrame только для тренировочной части
        train_lazy_df = full_dataset.lazy_df.slice(0, train_size + full_dataset.seq_len - 1)
        
        # Считаем распределение классов через Polars
        label_counts = (
            train_lazy_df
            .select(pl_pol.col("label"))
            .slice(full_dataset.seq_len - 1, train_size)
            .group_by("label")
            .agg(pl_pol.len().alias("count"))
            .collect(engine="streaming")
        )
        
        counts = np.zeros(3, dtype=np.int64)
        for row in label_counts.iter_rows():
            label, count = row
            if 0 <= label < 3:
                counts[int(label)] = count
    else:
        # Для memory и memmap режимов используем прямой доступ к меткам
        train_labels = full_dataset.labels[train_ds.indices]
        classes, counts_list = np.unique(train_labels, return_counts=True)
        
        counts = np.zeros(3, dtype=np.int64)
        for cls, count in zip(classes, counts_list):
            if 0 <= cls < 3:
                counts[int(cls)] = count
    
    total_samples = np.sum(counts)
    smoothing = args.class_weight_smooth
    n_classes = 3
    
    # Формула: weights = total / (n_classes * (counts + smoothing))
    weights = total_samples / (n_classes * (counts + smoothing))
    # Нормализуем
    weights = weights / np.mean(weights)
    
    print(f"Effective class weights: [Flat: {weights[0]:.2f}, Up: {weights[1]:.2f}, Down: {weights[2]:.2f}]")

    # 9. Инициализация модели
    print(f"Initializing model with loss: {args.loss_type} (gamma={args.focal_gamma if args.loss_type == 'focal' else 'N/A'}), activation: {args.activation}, scheduler: {args.scheduler}")
    print(f"Gradient clipping: mode={args.clip_mode}, threshold={args.clip_val}")
    
    # Если временное взвешивание включено, не передаем class_weights в модель (они уже в датасете)
    model_class_weights = None if args.use_time_weighting else weights
    
    # Knowledge Distillation режим
    teacher_model = None
    if args.mode == "distill":
        if args.teacher_path is None:
            raise ValueError("--teacher_path is required for distillation mode")
        
        print(f"\n=== Knowledge Distillation Mode ===")
        print(f"Loading teacher model from: {args.teacher_path}")
        
        # Загружаем teacher модель
        teacher_module = LiTModule.load_from_checkpoint(args.teacher_path)
        teacher_model = teacher_module.model
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        
        # Выводим информацию о teacher
        from .utils import count_parameters
        from .lit_model import LiTConfig
        
        teacher_params = count_parameters(teacher_model)
        print(f"Teacher model parameters: {teacher_params:,}")
        print(f"Teacher architecture: d_model={teacher_module.hparams.get('d_model', 64)}, "
              f"nhead={teacher_module.hparams.get('nhead', 4)}, "
              f"num_layers={teacher_module.hparams.get('num_layers', 2)}")
        
        # Создаем student конфигурацию
        print(f"\nCreating student model:")
        student_config = LiTConfig(
            seq_len=args.seq_len,
            in_channels=in_channels,
            d_model=args.student_d_model,
            nhead=args.student_nhead,
            num_layers=args.student_num_layers,
            dropout=0.1,
            activation=args.activation,
            multi_task=True,
            num_horizons=num_horizons,
            use_horizon_embedding=args.use_horizon_embedding
        )
        print(f"Student architecture: d_model={student_config.d_model}, "
              f"nhead={student_config.nhead}, "
              f"num_layers={student_config.num_layers}")
        
        # Создаем student модель через LiTModule с параметрами из config
        model = LiTModule(
            seq_len=student_config.seq_len,
            lr=1e-4,
            class_weights=None,  # Не используем class weights для distillation
            label_smoothing=0.0,  # Не используем label smoothing для distillation
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            activation=student_config.activation,
            use_time_weighting=args.use_time_weighting,
            teacher_model=teacher_model,
            alpha=args.alpha,
            temperature=args.temperature,
            in_channels=student_config.in_channels,
            past_returns_lags=past_returns_lags,
            d_model=student_config.d_model,
            nhead=student_config.nhead,
            num_layers=student_config.num_layers,
            dropout=student_config.dropout,
            multi_task=student_config.multi_task,
            num_horizons=num_horizons,
            horizon_weights=horizon_weights,
            use_horizon_embedding=args.use_horizon_embedding,
            # Параметры scheduler
            scheduler=args.scheduler,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            pct_start=args.pct_start,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            step_size=args.step_size,
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            # Параметры gradient clipping (Задача 154)
            clip_mode=args.clip_mode,
            clip_val=args.clip_val,
            # Параметры TensorBoard (Задача 158)
            tb_hist_freq=args.tb_hist_freq,
            tb_embedding_samples=args.tb_embedding_samples,
            # Параметры Curvature Regularization (Задача 238)
            use_curvature_reg=args.use_curvature_reg,
            curvature_lambda=args.curvature_lambda,
            input_noise_std=args.input_noise_std,
            # Параметры Robust Scaling (Задача 240)
            scaler_type=args.scaler_type,
            winsor_limits=list(winsor_limits) if winsor_limits else None
        )
        
        student_params = count_parameters(model.model)
        compression_ratio = teacher_params / student_params
        print(f"Student model parameters: {student_params:,}")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Distillation parameters: alpha={args.alpha}, temperature={args.temperature}")
        print("=" * 40 + "\n")
    else:
        # Обычный режим обучения - создаем teacher конфигурацию
        from .lit_model import LiTConfig
        
        # Для teacher используем параметры из командной строки
        teacher_config = LiTConfig(
            seq_len=args.seq_len,
            in_channels=in_channels,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            multi_task=True,
            num_horizons=num_horizons,
            use_horizon_embedding=args.use_horizon_embedding
        )
        
        print(f"\nCreating model with configuration:")
        print(f"Architecture: d_model={teacher_config.d_model}, "
              f"nhead={teacher_config.nhead}, "
              f"num_layers={teacher_config.num_layers}, "
              f"dropout={teacher_config.dropout}")
        
        model = LiTModule(
            seq_len=teacher_config.seq_len,
            lr=1e-4,
            class_weights=model_class_weights,
            label_smoothing=args.label_smoothing,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            activation=teacher_config.activation,
            use_time_weighting=args.use_time_weighting,
            use_regime_weighting=(regime_detector is not None),
            regime_weights=regime_weights,
            in_channels=teacher_config.in_channels,
            past_returns_lags=past_returns_lags,
            d_model=teacher_config.d_model,
            nhead=teacher_config.nhead,
            num_layers=teacher_config.num_layers,
            dropout=teacher_config.dropout,
            multi_task=teacher_config.multi_task,
            num_regimes=num_regimes,
            regime_embedding_dim=16,
            num_horizons=num_horizons,
            horizon_weights=horizon_weights,
            use_horizon_embedding=args.use_horizon_embedding,
            # Параметры scheduler
            scheduler=args.scheduler,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            pct_start=args.pct_start,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            step_size=args.step_size,
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            # Параметры gradient clipping (Задача 154)
            clip_mode=args.clip_mode,
            clip_val=args.clip_val,
            # Параметры TensorBoard (Задача 158)
            tb_hist_freq=args.tb_hist_freq,
            tb_embedding_samples=args.tb_embedding_samples,
            # Параметры Curvature Regularization (Задача 238)
            use_curvature_reg=args.use_curvature_reg,
            curvature_lambda=args.curvature_lambda,
            input_noise_std=args.input_noise_std,
            # Параметры Robust Scaling (Задача 240)
            scaler_type=args.scaler_type,
            winsor_limits=list(winsor_limits) if winsor_limits else None
        )

    # 10. Callbacks и Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="lit-{epoch:02d}-{val_mcc:.4f}",
        save_top_k=3,
        monitor="val_mcc",
        mode="max"
    )
    callbacks = [
        EarlyStopping(monitor="val_mcc", patience=15, mode="max"),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Настройка TensorBoard logger с пользовательской директорией
    tb_dir = args.tb_dir if args.tb_dir else f"runs/{args.symbol}"
    
    # Автоматическая очистка старых логов (опционально)
    from .utils import cleanup_old_tensorboard_logs
    cleanup_old_tensorboard_logs(tb_dir, max_runs=50)
    
    logger = TensorBoardLogger(tb_dir, name="lit_training")
    
    # Настраиваем Custom Scalars Layout для структурированного дашборда
    from .utils import setup_custom_scalars_layout
    setup_custom_scalars_layout(logger.experiment)

    # Логируем начальные гиперпараметры в начале запуска (Задача 158)
    from .utils import log_hparams
    hparams_dict = {
        'lr': 1e-4,
        'd_model': args.d_model if args.mode != 'distill' else args.student_d_model,
        'nhead': args.nhead if args.mode != 'distill' else args.student_nhead,
        'num_layers': args.num_layers if args.mode != 'distill' else args.student_num_layers,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'activation': args.activation,
        'scheduler': args.scheduler,
        'loss_type': args.loss_type,
        'label_smoothing': args.label_smoothing,
    }
    
    # Инициализируем с пустыми метриками (будут обновлены в конце)
    log_hparams(logger.experiment, hparams_dict, {})

    # 11. Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision="32",              # Задача 305-2: Временная мера для диагностики
        log_every_n_steps=100,      # Задача 304: Уменьшаем шаг логирования
        gradient_clip_val=0.5,      # Задача 304: Защита от NaN
        gradient_clip_algorithm="norm", # Задача 305-2: Явный алгоритм клиппинга
        enable_progress_bar=False   # Отключаем прогресс-бар, чтобы не было повторяющихся логов
    )
    
    # Добавляем symbol в trainer для доступа из LiTModule
    trainer.symbol = args.symbol

    # 11.5. Optuna поиск seq_len (Задача 055)
    if args.optuna_seq_len_search:
        print("\n" + "=" * 70)
        print("OPTUNA HYPERPARAMETER SEARCH FOR seq_len")
        print("=" * 70)
        print(f"Number of trials: {args.optuna_n_trials}")
        print(f"Pruner type: {args.optuna_pruner}")
        print("=" * 70 + "\n")
        
        # Создаем Optuna study
        if args.optuna_pruner == "hyperband":
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=min(20, args.epochs),
                reduction_factor=3
            )
        elif args.optuna_pruner == "patient":
            pruner = PatientPruner(patience=3)
        else:
            pruner = MedianPruner()
        
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Запускаем поиск
        study.optimize(
            lambda trial: objective_seq_len_search(
                trial, args, base_path, data_path, df,
                in_channels, past_returns_lags, num_horizons, horizon_weights,
                weights, normalizer, regime_detector, regime_weights, num_regimes, cache_dir
            ),
            n_trials=args.optuna_n_trials,
            show_progress_bar=True
        )
        
        # Получаем лучший seq_len
        best_trial = study.best_trial
        best_seq_len = best_trial.params["seq_len"]
        best_mcc = best_trial.value
        
        print(f"\n{'='*70}")
        print(f"OPTUNA SEARCH COMPLETED")
        print(f"{'='*70}")
        print(f"Best seq_len: {best_seq_len}")
        print(f"Best MCC: {best_mcc:.4f}")
        print(f"{'='*70}\n")
        
        # Сохраняем best_seq_len в конфиг эксперимента (Задача 055, пункт 3)
        config_path = base_path / "bots" / args.symbol / "models" / "optuna_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        optuna_config = {
            "best_seq_len": int(best_seq_len),
            "best_mcc": float(best_mcc),
            "n_trials": args.optuna_n_trials,
            "pruner_type": args.optuna_pruner,
            "all_trials": [
                {
                    "seq_len": int(trial.params["seq_len"]),
                    "mcc": float(trial.value) if trial.value is not None else None,
                    "state": str(trial.state)
                }
                for trial in study.trials
            ]
        }
        
        with open(config_path, 'w') as f:
            json.dump(optuna_config, f, indent=2)
        
        print(f"Optuna config saved to: {config_path}")
        
        # Обновляем args.seq_len на best_seq_len для дальнейшего обучения
        print(f"\nUpdating seq_len from {args.seq_len} to {best_seq_len} for training")
        args.seq_len = best_seq_len
        
        # Пересоздаем датасеты с best_seq_len
        print("Recreating datasets with best seq_len...")
        
        # Подготовка параметров для временного взвешивания
        time_weighting_params_final = {}
        if args.use_time_weighting:
            time_weighting_params_final = {
                'half_life_hours': args.half_life_hours,
                'min_weight': args.min_sample_weight,
                'class_weights': weights
            }
        else:
            time_weighting_params_final = {
                'half_life_hours': 24.0,
                'min_weight': 1.0,
                'class_weights': None
            }
        
        if args.data_mode == "streaming":
            full_dataset = LOBDataset(
                df,
                seq_len=args.seq_len,
                n_past_returns=n_past_returns,
                past_returns_lags=past_returns_lags,  # Задача 091
                data_mode="streaming",
                is_train=False,
                augment_prob=args.augment_prob,
                use_symmetric_flip=args.use_symmetric_flip,
                volume_jitter_range=args.volume_jitter_range,
                aug_seed=args.aug_seed,
                regime_detector=regime_detector,
                regime_window=1000,
                scaler_type=args.scaler_type,
                winsor_limits=winsor_limits,
                **time_weighting_params_final
            )
        elif args.data_mode == "memmap":
            full_dataset = LOBDataset(
                df,
                seq_len=args.seq_len,
                n_past_returns=n_past_returns,
                past_returns_lags=past_returns_lags,  # Задача 091
                data_mode="memmap",
                cache_dir=cache_dir,
                is_train=False,
                augment_prob=args.augment_prob,
                use_symmetric_flip=args.use_symmetric_flip,
                volume_jitter_range=args.volume_jitter_range,
                aug_seed=args.aug_seed,
                regime_detector=regime_detector,
                regime_window=1000,
                scaler_type=args.scaler_type,
                winsor_limits=winsor_limits,
                **time_weighting_params_final
            )
        else:
            full_dataset = LOBDataset(
                df,
                seq_len=args.seq_len,
                n_past_returns=n_past_returns,
                past_returns_lags=past_returns_lags,  # Задача 091
                data_mode="memory",
                is_train=False,
                augment_prob=args.augment_prob,
                use_symmetric_flip=args.use_symmetric_flip,
                volume_jitter_range=args.volume_jitter_range,
                aug_seed=args.aug_seed,
                regime_detector=regime_detector,
                regime_window=1000,
                scaler_type=args.scaler_type,
                winsor_limits=winsor_limits,
                **time_weighting_params_final
            )
        
        # Пересоздаем train/val разделение (80/20)
        total_len = len(full_dataset)
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        
        # Пересоздаем DataLoaders
        worker_init_fn = _streaming_worker_init_fn if args.data_mode == "streaming" else None
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=worker_init_fn
        )
        
        print("Datasets recreated successfully\n")

    # 12. Обучение
    if args.mode == "cv":
        # ============================================================================
        # Purged K-Fold Cross-Validation Mode (Задача 153)
        # ============================================================================
        print("\n" + "=" * 70)
        print("PURGED K-FOLD CROSS-VALIDATION MODE")
        print("=" * 70)
        
        from .utils import PurgedKFold
        from torch.utils.data import Subset
        
        # Инициализируем PurgedKFold
        cv = PurgedKFold(
            n_splits=args.n_splits,
            purge_buffer_events=args.purge_buffer_events,
            embargo_buffer_events=args.embargo_buffer_events
        )
        
        # Получаем timestamps из датасета для проверки сортировки
        timestamps = full_dataset.get_timestamps()
        
        # Используем только train+val данные для CV (test оставляем для финального holdout)
        cv_size = train_size + val_size
        cv_indices = np.arange(cv_size)
        cv_timestamps = timestamps[:cv_size]
        
        # Получаем метки для статистики
        if args.data_mode != "streaming":
            cv_labels = full_dataset.labels[:cv_size]
        else:
            # Для streaming загружаем метки
            cv_labels_df = full_dataset.lazy_df.select(pl_pol.col("label")).slice(full_dataset.seq_len - 1, cv_size).collect(engine="streaming")
            cv_labels = cv_labels_df.to_series().to_numpy()
        
        # Список для сбора MCC по фолдам
        fold_mccs = []
        fold_results = []
        
        print(f"\nCV Configuration:")
        print(f"  - Number of folds: {args.n_splits}")
        print(f"  - Purge buffer: {args.purge_buffer_events} events")
        print(f"  - Embargo buffer: {args.embargo_buffer_events} events")
        print(f"  - Total CV samples: {cv_size}")
        print(f"  - Holdout test samples: {len(test_ds)}")
        print()
        
        # Цикл по фолдам
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(cv_indices, cv_labels, cv_timestamps)):
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx + 1}/{args.n_splits}")
            print(f"{'='*70}")
            print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            
            # Создаем Subset для train и val
            fold_train_ds = Subset(full_dataset, train_idx)
            fold_val_ds = Subset(full_dataset, val_idx)
            
            # Создаем DataLoaders для фолда
            num_workers = 2 if args.data_mode == "streaming" else 4
            worker_init_fn = _streaming_worker_init_fn if args.data_mode == "streaming" else None
            
            fold_train_loader = DataLoader(
                fold_train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
                worker_init_fn=worker_init_fn
            )
            
            fold_val_loader = DataLoader(
                fold_val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )
            
            # ВАЖНО: Полная инициализация модели с fresh weights для каждого фолда
            from .lit_model import LiTConfig
            
            fold_config = LiTConfig(
                seq_len=args.seq_len,
                in_channels=in_channels,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dropout=args.dropout,
                activation=args.activation,
                multi_task=True,
                num_horizons=num_horizons,
                use_horizon_embedding=args.use_horizon_embedding
            )
            
            fold_model = LiTModule(
                seq_len=fold_config.seq_len,
                lr=1e-4,
                class_weights=model_class_weights,
                label_smoothing=args.label_smoothing,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                activation=fold_config.activation,
                use_time_weighting=args.use_time_weighting,
                in_channels=fold_config.in_channels,
                past_returns_lags=past_returns_lags,
                d_model=fold_config.d_model,
                nhead=fold_config.nhead,
                num_layers=fold_config.num_layers,
                num_horizons=num_horizons,
                horizon_weights=horizon_weights,
                use_horizon_embedding=args.use_horizon_embedding,
                dropout=fold_config.dropout,
                multi_task=fold_config.multi_task,
                scheduler=args.scheduler,
                div_factor=args.div_factor,
                final_div_factor=args.final_div_factor,
                pct_start=args.pct_start,
                plateau_factor=args.plateau_factor,
                plateau_patience=args.plateau_patience,
                step_size=args.step_size,
                gamma=args.gamma,
                weight_decay=args.weight_decay,
                # Параметры gradient clipping (Задача 154)
                clip_mode=args.clip_mode,
                clip_val=args.clip_val,
                # Параметры Curvature Regularization (Задача 238)
                use_curvature_reg=args.use_curvature_reg,
                curvature_lambda=args.curvature_lambda,
                input_noise_std=args.input_noise_std,
                # Параметры Robust Scaling (Задача 240)
                scaler_type=args.scaler_type,
                winsor_limits=list(winsor_limits) if winsor_limits else None
            )
            
            # Создаем callbacks для фолда (независимый EarlyStopping)
            fold_checkpoint_dir = checkpoint_dir / f"fold_{fold_idx + 1}"
            fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            fold_checkpoint_callback = ModelCheckpoint(
                dirpath=fold_checkpoint_dir,
                filename="lit-{epoch:02d}-{val_mcc:.4f}",
                save_top_k=1,
                monitor="val_mcc",
                mode="max"
            )
            
            fold_callbacks = [
                EarlyStopping(monitor="val_mcc", patience=15, mode="max"),
                fold_checkpoint_callback,
                LearningRateMonitor(logging_interval="epoch")
            ]
            
            fold_logger = TensorBoardLogger("tb_logs", name=f"lit_{args.symbol}_fold{fold_idx + 1}")
            
            # Создаем trainer для фолда
            fold_trainer = pl.Trainer(
                max_epochs=args.epochs,
                callbacks=fold_callbacks,
                logger=fold_logger,
                accelerator="auto",
                devices=1,
                precision="16-mixed" if torch.cuda.is_available() else 32,
                enable_progress_bar=False,  # Отключаем прогресс-бар
                log_every_n_steps=100,      # Задача 304: Уменьшаем шаг логирования
                gradient_clip_val=0.5       # Задача 304: Защита от NaN
            )
            
            fold_trainer.symbol = args.symbol
            
            # Обучаем модель на фолде
            print(f"\nTraining fold {fold_idx + 1}...")
            fold_trainer.fit(fold_model, fold_train_loader, fold_val_loader)
            
            # Оцениваем на валидационном фолде
            best_fold_model_path = fold_checkpoint_callback.best_model_path
            if best_fold_model_path:
                print(f"\nEvaluating fold {fold_idx + 1}...")
                best_fold_model = LiTModule.load_from_checkpoint(best_fold_model_path)
                best_fold_model.eval()
                best_fold_model.freeze()
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                best_fold_model.to(device)
                
                # Предсказания на валидационном фолде
                fold_y_true = []
                fold_y_pred = []
                
                with torch.no_grad():
                    for batch in fold_val_loader:
                        if args.balance_method != "none" or args.use_time_weighting:
                            x, y, _, _ = batch
                        else:
                            x, y, _, _ = batch
                        
                        x = x.to(device)
                        logits, _ = best_fold_model(x)
                        preds = torch.argmax(logits, dim=1)
                        
                        fold_y_true.extend(y.cpu().numpy())
                        fold_y_pred.extend(preds.cpu().numpy())
                
                # Вычисляем MCC для фолда
                fold_mcc = matthews_corrcoef(fold_y_true, fold_y_pred)
                fold_mccs.append(fold_mcc)
                
                # Сохраняем результаты фолда
                fold_results.append({
                    'fold': fold_idx + 1,
                    'mcc': fold_mcc,
                    'train_size': len(train_idx),
                    'val_size': len(val_idx),
                    'best_model_path': best_fold_model_path
                })
                
                print(f"\nFold {fold_idx + 1} Results:")
                print(f"  - MCC: {fold_mcc:.4f}")
                print(f"  - Best model: {best_fold_model_path}")
        
        # Агрегируем результаты по всем фолдам
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*70}")
        
        mean_mcc = np.mean(fold_mccs)
        std_mcc = np.std(fold_mccs)
        
        print(f"\nPer-Fold MCC:")
        for result in fold_results:
            print(f"  Fold {result['fold']}: {result['mcc']:.4f}")
        
        print(f"\nAggregated Metrics:")
        print(f"  - Mean MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")
        print(f"  - Min MCC: {np.min(fold_mccs):.4f}")
        print(f"  - Max MCC: {np.max(fold_mccs):.4f}")
        
        # Сохраняем результаты CV в JSON
        import json
        cv_results = {
            'n_splits': args.n_splits,
            'purge_buffer_events': args.purge_buffer_events,
            'embargo_buffer_events': args.embargo_buffer_events,
            'mean_mcc': float(mean_mcc),
            'std_mcc': float(std_mcc),
            'min_mcc': float(np.min(fold_mccs)),
            'max_mcc': float(np.max(fold_mccs)),
            'folds': [
                {
                    'fold': r['fold'],
                    'mcc': float(r['mcc']),
                    'train_size': int(r['train_size']),
                    'val_size': int(r['val_size'])
                }
                for r in fold_results
            ]
        }
        
        cv_results_path = base_path / "bots" / args.symbol / "models" / "cv_results.json"
        cv_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cv_results_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        print(f"\nCV results saved to: {cv_results_path}")
        
        # Финальная оценка на holdout test set (используем лучшую модель из лучшего фолда)
        best_fold_idx = np.argmax(fold_mccs)
        best_fold_result = fold_results[best_fold_idx]
        
        print(f"\n{'='*70}")
        print(f"HOLDOUT TEST EVALUATION (using best fold: {best_fold_result['fold']})")
        print(f"{'='*70}")
        
        best_model_path = best_fold_result['best_model_path']
        best_model = LiTModule.load_from_checkpoint(best_model_path)
        best_model.eval()
        best_model.freeze()
        best_model.to(device)
        
        # Предсказания на holdout test set
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                if args.balance_method != "none" or args.use_time_weighting:
                    x, y, _, _ = batch
                else:
                    x, y, _, _ = batch
                
                x = x.to(device)
                logits, _ = best_model(x)
                preds = torch.argmax(logits, dim=1)
                
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        # Вычисляем метрики на holdout
        holdout_mcc = matthews_corrcoef(y_true, y_pred)
        
        print(f"\nHoldout Test Results:")
        print(f"  - MCC: {holdout_mcc:.4f}")
        
        # Сравниваем CV и Holdout
        mcc_diff = abs(mean_mcc - holdout_mcc)
        mcc_diff_pct = (mcc_diff / mean_mcc) * 100 if mean_mcc != 0 else 0
        
        print(f"\nCV vs Holdout Comparison:")
        print(f"  - CV Mean MCC: {mean_mcc:.4f}")
        print(f"  - Holdout MCC: {holdout_mcc:.4f}")
        print(f"  - Absolute Difference: {mcc_diff:.4f}")
        print(f"  - Relative Difference: {mcc_diff_pct:.2f}%")
        
        if mcc_diff_pct > 15:
            print(f"\n⚠️  WARNING: MCC difference > 15% detected!")
            print(f"  This may indicate Data Drift or feature instability.")
            print(f"  Consider:")
            print(f"    - Reviewing feature engineering")
            print(f"    - Checking for regime changes in market data")
            print(f"    - Increasing purge/embargo buffers")
        
        print(f"\n{'='*70}")
        print("Cross-validation completed successfully!")
        print(f"{'='*70}\n")
        
    else:
        # Обычный режим обучения (train или distill)
        print("Starting training...")
        
        # Гиперпараметры уже логированы в начале (Задача 158)
        # Выводим финальное значение label_smoothing (с учетом логики взаимного исключения)
        effective_label_smoothing = 0.0 if args.loss_type == "focal" else args.label_smoothing
        print(f"Label smoothing configuration:")
        print(f"  - Requested: {args.label_smoothing}")
        print(f"  - Effective (after Focal Loss check): {effective_label_smoothing}")
        if args.loss_type == "focal":
            print(f"  - Note: Label smoothing disabled because Focal Loss is used (they are alternatives)")
        print()
        
        # --- Sanity Check (Задача 304) ---
        print("Performing sanity check on data...")
        try:
            batch = next(iter(train_loader))
            x_check = batch[0] # [B, 3, 50] или [B, C, S, 50] в зависимости от реализации LOBPatching
            if not torch.isfinite(x_check).all():
                raise ValueError("NaN or Inf detected in input features before training! Check feature engineering and normalization.")
            print(f"Sanity check passed. Input shape: {x_check.shape}, range: [{x_check.min():.4f}, {x_check.max():.4f}]")
        except Exception as e:
            print(f"Sanity check failed: {e}")
            if not args.optuna_seq_len_search: # Не падаем в Optuna, если это временная ошибка
                raise e
        # -------------------------------
        
        trainer.fit(model, train_loader, val_loader)
        
        # ============================================================================
        # MC Dropout for Uncertainty Estimation (Задача 125)
        # ============================================================================
        print(f"\n{'='*60}")
        print("MC DROPOUT UNCERTAINTY ESTIMATION")
        print(f"{'='*60}\n")
        
        try:
            from .utils import calculate_uncertainty
            
            # Переключаем модель в режим MC Dropout
            model.model.apply(enable_dropout)
            model.model.eval()  # Остальные слои остаются в eval режиме
            
            # Выполняем warm-up для стабилизации CUDA/JIT
            print("Warming up model...")
            with torch.no_grad():
                dummy_input = torch.randn(1, args.seq_len, in_channels, 50, device=next(model.parameters()).device)
                for _ in range(5):
                    _ = model(dummy_input)
            print("Warm-up completed.\n")
            
            # Выполняем N прогонов MC Dropout для сбора логитов
            n_mc_passes = 20
            print(f"Performing {n_mc_passes} MC Dropout passes on validation set...")
            
            mc_logits_list = []
            val_labels_list = []
            
            for mc_pass in tqdm(range(n_mc_passes), desc="MC Passes"):
                pass_logits = []
                pass_labels = []
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Pass {mc_pass+1}/{n_mc_passes}", leave=False):
                        if len(batch) == 5:
                            x, y, _, _, regime_id = batch
                        else:
                            x, y, _, _ = batch
                            regime_id = None
                        
                        x = x.to(next(model.parameters()).device)
                        y = y.to(next(model.parameters()).device)
                        
                        if regime_id is not None:
                            regime_id = regime_id.to(next(model.parameters()).device)
                        
                        logits, _ = model(x, regime_id=regime_id)
                        pass_logits.append(logits.cpu())
                        pass_labels.append(y.cpu())
                
                mc_logits_list.append(torch.cat(pass_logits, dim=0))
                if mc_pass == 0:
                    val_labels_list = torch.cat(pass_labels, dim=0)
            
            # Объединяем логиты: (n_passes, batch_size, num_classes)
            mc_logits = torch.stack(mc_logits_list, dim=0)
            
            # Вычисляем неопределенность
            mean_probs, entropy, mutual_info = calculate_uncertainty(mc_logits)
            
            # Логируем статистику неопределенности
            print(f"\n{'='*60}")
            print("Uncertainty Statistics:")
            print(f"  Entropy - Mean: {entropy.mean().item():.4f}, Std: {entropy.std().item():.4f}")
            print(f"           Min: {entropy.min().item():.4f}, Max: {entropy.max().item():.4f}")
            print(f"  MI      - Mean: {mutual_info.mean().item():.4f}, Std: {mutual_info.std().item():.4f}")
            print(f"           Min: {mutual_info.min().item():.4f}, Max: {mutual_info.max().item():.4f}")
            print(f"{'='*60}\n")
            
            # Сохраняем MC Dropout логиты и неопределенность для использования в evaluate_uncertainty.py
            uncertainty_data = {
                'mc_logits': mc_logits,
                'entropy': entropy,
                'mutual_info': mutual_info,
                'val_labels': val_labels_list,
                'mean_probs': mean_probs
            }
            
            # Сохраняем в checkpoint директорию для дальнейшего анализа
            uncertainty_path = checkpoint_dir / "mc_dropout_uncertainty.pt"
            torch.save(uncertainty_data, uncertainty_path)
            print(f"MC Dropout uncertainty data saved to: {uncertainty_path}\n")
            
        except Exception as e:
            print(f"Warning: MC Dropout uncertainty estimation failed: {str(e)}")
            print("Continuing with model pruning...\n")
        
        # ============================================================================
        # Model Pruning (Задача 159)
        # ============================================================================
        if args.prune_mode != "none":
            from .utils import (
                apply_iterative_pruning, 
                apply_structured_pruning_2_4,
                remove_pruning_reparametrization,
                calculate_sparsity,
                save_pruned_model,
                log_pruning_progress,
                print_pruning_warning
            )
            
            print(f"\n{'='*60}")
            print(f"STARTING MODEL PRUNING")
            print(f"{'='*60}")
            print(f"Mode: {args.prune_mode}")
            print(f"Target Sparsity: {args.prune_amount:.2%}")
            print(f"Iterations: {args.prune_iterations}")
            print(f"Fine-tune Epochs per Iteration: {args.prune_finetune_epochs}")
            print(f"{'='*60}\n")
            
            # Выводим предупреждение для unstructured pruning
            if args.prune_mode == "unstructured":
                print_pruning_warning()
            
            # Загружаем лучшую модель для прунинга
            if checkpoint_callback.best_model_path:
                print(f"Loading best model for pruning: {checkpoint_callback.best_model_path}")
                model_module = LiTModule.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    map_location="cpu"
                )
                model = model_module.model
            
            # Вычисляем baseline MCC для сравнения
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Computing baseline MCC"):
                    # Батч всегда содержит 5 элементов: x, y, vol, weight, regime_id
                    x = batch[0].to(model.device)
                    y = batch[1].to(model.device)
                    regime_id = batch[4].to(model.device) if len(batch) > 4 else None
                    
                    logits_cls, _ = model(x, regime_id=regime_id)
                    preds = torch.argmax(logits_cls, dim=1)
                    
                    all_preds.append(preds.cpu())
                    all_labels.append(y.cpu())
            
            baseline_preds = torch.cat(all_preds).numpy()
            baseline_labels = torch.cat(all_labels).numpy()
            baseline_mcc = matthews_corrcoef(baseline_labels, baseline_preds)
            
            print(f"\nBaseline MCC (before pruning): {baseline_mcc:.4f}")
            
            # Итеративный прунинг
            sparsifier = None
            
            for iteration in range(1, args.prune_iterations + 1):
                # Вычисляем текущую долю прунинга (линейное увеличение)
                current_amount = (iteration / args.prune_iterations) * args.prune_amount
                
                # Применяем прунинг
                if args.prune_mode == "unstructured":
                    prune_stats = apply_iterative_pruning(model, current_amount, prune_mode='unstructured')
                elif args.prune_mode == "structured_2_4":
                    if iteration == 1:  # Применяем 2:4 только один раз
                        sparsifier = apply_structured_pruning_2_4(model)
                        if sparsifier is None:
                            print("⚠️  Structured 2:4 pruning failed. Skipping pruning.")
                            break
                
                # Вычисляем текущую разреженность
                sparsity_stats = calculate_sparsity(model, detailed=False)
                
                # Fine-tuning после прунинга
                print(f"\nFine-tuning for {args.prune_finetune_epochs} epochs...")
                
                # Создаем новый trainer для fine-tuning
                finetune_trainer = pl.Trainer(
                    max_epochs=args.prune_finetune_epochs,
                    accelerator="auto",
                    devices=1,
                    logger=logger,
                    callbacks=[checkpoint_callback],
                    enable_progress_bar=False,  # Отключаем прогресс-бар
                    gradient_clip_val=0.5,      # Задача 304: Защита от NaN
                    deterministic=False,
                    log_every_n_steps=100       # Задача 304: Уменьшаем шаг логирования
                )
                
                finetune_trainer.fit(model, train_loader, val_loader)
                
                # Оцениваем MCC после fine-tuning
                model.eval()
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Evaluating iteration {iteration}"):
                        # Батч всегда содержит 5 элементов: x, y, vol, weight, regime_id
                        x = batch[0].to(model.device)
                        y = batch[1].to(model.device)
                        regime_id = batch[4].to(model.device) if len(batch) > 4 else None
                        
                        logits_cls, _ = model(x, regime_id=regime_id)
                        preds = torch.argmax(logits_cls, dim=1)
                        
                        all_preds.append(preds.cpu())
                        all_labels.append(y.cpu())
                
                current_preds = torch.cat(all_preds).numpy()
                current_labels = torch.cat(all_labels).numpy()
                current_mcc = matthews_corrcoef(current_labels, current_preds)
                
                # Логируем прогресс
                log_pruning_progress(
                    iteration, 
                    args.prune_iterations, 
                    current_amount, 
                    args.prune_amount,
                    sparsity_stats, 
                    current_mcc, 
                    baseline_mcc
                )
            
            # Удаляем параметризации прунинга
            if args.prune_mode == "unstructured":
                remove_pruning_reparametrization(model)
            elif args.prune_mode == "structured_2_4" and sparsifier is not None:
                sparsifier.squash_mask()
                print("✓ Squashed 2:4 sparsity masks")
                
                # Конвертируем в SparseSemiStructuredTensor для ускорения
                from .utils import convert_to_sparse_semi_structured
                convert_to_sparse_semi_structured(model)
            
            # Финальная статистика
            final_sparsity_stats = calculate_sparsity(model, detailed=True)
            
            print(f"\n{'='*60}")
            print(f"PRUNING COMPLETED")
            print(f"{'='*60}")
            print(f"Final Sparsity: {final_sparsity_stats['global_sparsity']:.2%}")
            print(f"Final MCC: {current_mcc:.4f}")
            print(f"MCC Drop: {baseline_mcc - current_mcc:.4f} ({((baseline_mcc - current_mcc) / baseline_mcc * 100):.2f}%)")
            print(f"{'='*60}\n")
            
            # Сохраняем разреженную модель
            pruned_model_path = base_path / "bots" / args.symbol / "models" / f"pruned_{args.prune_mode}.pt"
            save_pruned_model(model, pruned_model_path, final_sparsity_stats, baseline_mcc)
            
            # Обновляем checkpoint_callback для использования разреженной модели
            checkpoint_callback.best_model_path = str(pruned_model_path)
        
        # Логируем итоговые метрики после обучения
        best_val_mcc = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0.0
        metrics_dict = {
            'hparam/best_val_mcc': best_val_mcc,
        }
        log_hparams(logger.experiment, hparams_dict, metrics_dict)

    # 13. Финальное тестирование на Holdout (Test) выборке
    # Только для режимов train и distill (для cv уже выполнено выше)
    if args.mode != "cv":
        print("\nStarting final Holdout evaluation...")
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            best_model = LiTModule.load_from_checkpoint(best_model_path)
            best_model.eval()
            best_model.freeze()
            
            # Предсказания на тестовом наборе
            y_true = []
            y_pred = []
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            best_model.to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for batch in tqdm(test_loader, desc="Testing"):
                    if args.balance_method != "none" or args.use_time_weighting:
                        x, y, vol_target, _ = batch  # Игнорируем веса на тесте
                    else:
                        x, y, vol_target, _ = batch
                    
                    logits, _ = best_model(x.to(device))
                    preds = torch.argmax(logits, dim=1)
                    y_true.append(y.numpy())
                    y_pred.append(preds.cpu().numpy())
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        # Сохранение матриц ошибок
        class_names = ["Flat", "Up", "Down"]
        save_confusion_matrices(y_true, y_pred, class_names, base_path / "bots" / args.symbol / "model")
        
        # Отчет в консоль
        print("\nHoldout Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # 14. Сравнение Teacher vs Student (только для distillation режима)
        if args.mode == "distill":
            print("\n" + "=" * 60)
            print("KNOWLEDGE DISTILLATION: Teacher vs Student Comparison")
            print("=" * 60)
            
            from .utils import measure_latency, count_parameters
            from sklearn.metrics import matthews_corrcoef
            
            # Вычисляем MCC для student
            student_mcc = matthews_corrcoef(y_true, y_pred)
            
            # Вычисляем MCC для teacher
            teacher_model.to(device)
            teacher_model.eval()
            y_pred_teacher = []
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    for batch in tqdm(test_loader, desc="Testing Teacher"):
                        if args.balance_method != "none" or args.use_time_weighting:
                            x, y, vol_target, _ = batch
                        else:
                            x, y, vol_target, _ = batch
                        
                        teacher_logits, _ = teacher_model(x.to(device))
                        preds = torch.argmax(teacher_logits, dim=1)
                        y_pred_teacher.append(preds.cpu().numpy())
            
            y_pred_teacher = np.concatenate(y_pred_teacher)
            teacher_mcc = matthews_corrcoef(y_true, y_pred_teacher)
            
            # Замеряем латентность
            # Создаем тестовый батч
            sample_batch = next(iter(test_loader))
            if args.balance_method != "none" or args.use_time_weighting:
                sample_x, _, _, _ = sample_batch
            else:
                sample_x, _, _, _ = sample_batch
            
            teacher_latency = measure_latency(teacher_model, sample_x, device=str(device), warmup_runs=10, test_runs=100)
            student_latency = measure_latency(best_model.model, sample_x, device=str(device), warmup_runs=10, test_runs=100)
            
            # Подсчитываем параметры
            teacher_params = count_parameters(teacher_model)
            student_params = count_parameters(best_model.model)
            
            # Вычисляем метрики сравнения
            speedup = teacher_latency / student_latency
            compression_ratio = teacher_params / student_params
            mcc_retention = (student_mcc / teacher_mcc) * 100 if teacher_mcc != 0 else 0
            
            # Выводим таблицу сравнения
            print(f"\n{'Metric':<25} {'Teacher':<15} {'Student':<15} {'Improvement':<15}")
            print("-" * 70)
            print(f"{'MCC':<25} {teacher_mcc:<15.4f} {student_mcc:<15.4f} {mcc_retention:<15.2f}%")
            print(f"{'Latency (ms)':<25} {teacher_latency:<15.2f} {student_latency:<15.2f} {speedup:<15.2f}x")
            print(f"{'Parameters':<25} {teacher_params:<15,} {student_params:<15,} {compression_ratio:<15.2f}x")
            print("-" * 70)
            
            # Сохраняем метрики в JSON
            import json
            metrics_dict = {
                "teacher": {
                    "mcc": float(teacher_mcc),
                    "latency_ms": float(teacher_latency),
                    "parameters": int(teacher_params)
                },
                "student": {
                    "mcc": float(student_mcc),
                    "latency_ms": float(student_latency),
                    "parameters": int(student_params)
                },
                "comparison": {
                    "speedup": float(speedup),
                    "compression_ratio": float(compression_ratio),
                    "mcc_retention_percent": float(mcc_retention)
                },
                "distillation_params": {
                    "alpha": args.alpha,
                    "temperature": args.temperature
                }
            }
            
            metrics_path = base_path / "bots" / args.symbol / "models" / "distillation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            print(f"\nDistillation metrics saved to: {metrics_path}")
            print("\n" + "=" * 60)
        
        print("\nEvaluation completed. Run 'python evaluate.py --checkpoint PATH' for uncertainty and interpretability analysis.")

    else:
        print("No best model found, skipping evaluation.")
    
    # 14. Автоматическое сохранение модели (Задача 151, пункты 1 и 6)
    # Только для режимов train и distill
    if args.mode != "cv" and best_model_path:
        import shutil
        
        # Определяем целевой путь в зависимости от режима
        if args.mode == "distill":
            # Student модель сохраняем как lit.pt для экспорта в ONNX
            target_path = base_path / "bots" / args.symbol / "models" / "lit.pt"
            model_type = "Student"
        else:
            # Teacher модель сохраняем как teacher_lit.pt для дистилляции
            target_path = base_path / "bots" / args.symbol / "models" / "teacher_lit.pt"
            model_type = "Teacher"
        
        # Создаем директорию если не существует
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Копируем лучший checkpoint
        shutil.copy2(best_model_path, target_path)
        
        print(f"\n{'='*60}")
        print(f"✓ {model_type} model automatically saved to:")
        print(f"  {target_path}")
        print(f"  Source: {best_model_path}")
        
        if args.mode == "train":
            print(f"\nNext step: Use this teacher for distillation:")
            print(f"  python -m python_lab.src.train \\")
            print(f"    --symbol {args.symbol} \\")
            print(f"    --mode distill \\")
            print(f"    --teacher_path {target_path}")
        else:
            print(f"\nNext step: Export to ONNX:")
            print(f"  python -m python_lab.scripts.export_onnx \\")
            print(f"    --checkpoint {target_path}")
        
        print(f"{'='*60}\n")
    elif args.mode == "cv":
        print(f"\n{'='*60}")
        print("CV mode completed. Best fold models saved in:")
        print(f"  {checkpoint_dir}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    train()
