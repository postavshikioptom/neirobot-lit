"""
train_model_factory.py — Сборка teacher/student модели для train.py.
Вынесено из train.py в рамках задачи 322.6.
"""
from dataclasses import dataclass

from .train_module import LiTModule


@dataclass
class BuiltTrainingModel:
    module: LiTModule
    teacher_model: object  # nn.Module | None
    model_class_weights: object  # np.ndarray | None


def build_training_module(
    args,
    *,
    in_channels,
    past_returns_lags,
    num_horizons,
    horizon_weights,
    model_class_weights,
    regime_detector,
    regime_weights,
    num_regimes,
    winsor_limits,
    label_columns,
    class_weight_metadata,
) -> BuiltTrainingModel:
    """
    Собирает LiTModule для режимов train или distill.
    LiTConfig остаётся в lit_model.py — здесь только orchestration.
    """
    from .lit_model import LiTConfig

    teacher_model = None

    if args.mode == "distill":
        if args.teacher_path is None:
            raise ValueError("--teacher_path is required for distillation mode")

        print(f"\n=== Knowledge Distillation Mode ===")
        print(f"Loading teacher model from: {args.teacher_path}")

        teacher_module = LiTModule.load_from_checkpoint(args.teacher_path)
        teacher_model = teacher_module.model
        teacher_model.eval()
        teacher_model.requires_grad_(False)

        from .utils import count_parameters
        teacher_params = count_parameters(teacher_model)
        print(f"Teacher model parameters: {teacher_params:,}")
        print(f"Teacher architecture: d_model={teacher_module.hparams.get('d_model', 64)}, "
              f"nhead={teacher_module.hparams.get('nhead', 4)}, "
              f"num_layers={teacher_module.hparams.get('num_layers', 2)}")

        print(f"\nCreating student model:")
        student_config = LiTConfig(
            seq_len=args.seq_len,
            in_channels=in_channels,
            d_model=args.student_d_model,
            nhead=args.student_nhead,
            num_layers=args.student_num_layers,
            dropout=0.1,
            activation=args.activation,
            multi_task=args.multi_task,
            num_horizons=num_horizons,
            use_horizon_embedding=args.use_horizon_embedding,
        )
        print(f"Student architecture: d_model={student_config.d_model}, "
              f"nhead={student_config.nhead}, "
              f"num_layers={student_config.num_layers}")

        module = LiTModule(
            seq_len=student_config.seq_len,
            lr=1e-4,
            class_weights=None,
            label_smoothing=0.0,
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
            cls_loss_weight=args.cls_loss_weight,
            vol_loss_weight=args.vol_loss_weight,
            num_horizons=num_horizons,
            horizon_weights=horizon_weights,
            use_horizon_embedding=args.use_horizon_embedding,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            scheduler=args.scheduler,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            pct_start=args.pct_start,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            step_size=args.step_size,
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            clip_mode=args.clip_mode,
            clip_val=args.clip_val,
            tb_hist_freq=args.tb_hist_freq,
            tb_embedding_samples=args.tb_embedding_samples,
            use_curvature_reg=args.use_curvature_reg,
            curvature_lambda=args.curvature_lambda,
            input_noise_std=args.input_noise_std,
            scaler_type=args.scaler_type,
            winsor_limits=list(winsor_limits) if winsor_limits else None,
            metric_contract=args.metric_contract,
            metric_log_prefix=args.metric_log_prefix,
            metric_directional_base=args.metric_directional_base,
            decision_rule=args.decision_rule,
            decision_confidence=args.decision_confidence,
            decision_hold_threshold=args.decision_hold_threshold,
            flat_prob_threshold=args.flat_prob_threshold,
            up_prob_threshold=args.up_prob_threshold,
            down_prob_threshold=args.down_prob_threshold,
            margin_threshold=args.margin_threshold,
            report_fee_bps=args.report_fee_bps,
            report_slippage_bps=args.report_slippage_bps,
            report_half_spread_bps=args.report_half_spread_bps,
            label_mode=args.label_mode,
            time_mode=args.time_mode,
            model_label_columns=list(label_columns),
            class_weight_metadata=dict(class_weight_metadata),
        )

        student_params = count_parameters(module.model)
        compression_ratio = teacher_params / student_params
        print(f"Student model parameters: {student_params:,}")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Distillation parameters: alpha={args.alpha}, temperature={args.temperature}")
        print("=" * 40 + "\n")

    else:
        # Обычный режим train
        teacher_config = LiTConfig(
            seq_len=args.seq_len,
            in_channels=in_channels,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            multi_task=args.multi_task,
            num_horizons=num_horizons,
            use_horizon_embedding=args.use_horizon_embedding,
        )

        print(f"\nCreating model with configuration:")
        print(f"Architecture: d_model={teacher_config.d_model}, "
              f"nhead={teacher_config.nhead}, "
              f"num_layers={teacher_config.num_layers}, "
              f"dropout={teacher_config.dropout}")

        module = LiTModule(
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
            cls_loss_weight=args.cls_loss_weight,
            vol_loss_weight=args.vol_loss_weight,
            num_regimes=num_regimes,
            regime_embedding_dim=16,
            num_horizons=num_horizons,
            horizon_weights=horizon_weights,
            use_horizon_embedding=args.use_horizon_embedding,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            scheduler=args.scheduler,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            pct_start=args.pct_start,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            step_size=args.step_size,
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            clip_mode=args.clip_mode,
            clip_val=args.clip_val,
            tb_hist_freq=args.tb_hist_freq,
            tb_embedding_samples=args.tb_embedding_samples,
            use_curvature_reg=args.use_curvature_reg,
            curvature_lambda=args.curvature_lambda,
            input_noise_std=args.input_noise_std,
            scaler_type=args.scaler_type,
            winsor_limits=list(winsor_limits) if winsor_limits else None,
            metric_contract=args.metric_contract,
            metric_log_prefix=args.metric_log_prefix,
            metric_directional_base=args.metric_directional_base,
            decision_rule=args.decision_rule,
            decision_confidence=args.decision_confidence,
            decision_hold_threshold=args.decision_hold_threshold,
            flat_prob_threshold=args.flat_prob_threshold,
            up_prob_threshold=args.up_prob_threshold,
            down_prob_threshold=args.down_prob_threshold,
            margin_threshold=args.margin_threshold,
            report_fee_bps=args.report_fee_bps,
            report_slippage_bps=args.report_slippage_bps,
            report_half_spread_bps=args.report_half_spread_bps,
            label_mode=args.label_mode,
            time_mode=args.time_mode,
            model_label_columns=list(label_columns),
            class_weight_metadata=dict(class_weight_metadata),
        )

    return BuiltTrainingModel(
        module=module,
        teacher_model=teacher_model,
        model_class_weights=model_class_weights,
    )
