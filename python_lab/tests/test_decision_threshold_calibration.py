import numpy as np

from python_lab.src.utils import (
    apply_decision_rule,
    calibrate_decision_thresholds_for_target_coverage,
)


def _build_synthetic_inputs(n: int = 100):
    directional_conf = np.linspace(0.2, 0.95, n, dtype=np.float64)
    probs = np.zeros((n, 3), dtype=np.float64)
    probs[:, 0] = 1.0 - directional_conf
    probs[: n // 2, 1] = directional_conf[: n // 2]
    probs[n // 2 :, 2] = directional_conf[n // 2 :]

    y_true = np.zeros(n, dtype=np.int64)
    y_true[: n // 2] = 1
    y_true[n // 2 :] = 2

    logits = np.log(np.clip(probs, 1e-8, 1.0))
    f_ret = np.where(y_true == 1, 0.002, -0.002).astype(np.float64)
    imbalance = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    return probs, y_true, logits, f_ret, imbalance


def test_threshold_calibration_hits_target_coverage_band():
    probs, y_true, logits, f_ret, imbalance = _build_synthetic_inputs()
    base_params = {
        "decision_confidence": 0.5,
        "decision_hold_threshold": 0.6,
        "flat_prob_threshold": 0.34,
        "up_prob_threshold": 0.34,
        "down_prob_threshold": 0.34,
        "margin_threshold": 0.0,
    }

    result = calibrate_decision_thresholds_for_target_coverage(
        probs=probs,
        y_true=y_true,
        logits=logits,
        f_ret=f_ret,
        imbalance=imbalance,
        rule="flat_bias",
        base_params=base_params,
        target_coverage=0.40,
        target_tolerance=0.05,
        min_coverage=0.18,
        max_coverage=0.75,
        quantiles=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        opt_metric="net_edge_total",
        directional_base="predicted",
        fee_bps=0.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
    )

    assert result["used_fallback"] is False
    selected_coverage = float(result["selected_metrics"]["coverage_directional"])
    assert 0.35 <= selected_coverage <= 0.45


def test_threshold_calibration_fallback_when_no_candidate():
    probs, y_true, logits, f_ret, imbalance = _build_synthetic_inputs()
    base_params = {
        "decision_confidence": 0.5,
        "decision_hold_threshold": 0.6,
        "flat_prob_threshold": 0.34,
        "up_prob_threshold": 0.34,
        "down_prob_threshold": 0.34,
        "margin_threshold": 0.0,
    }

    result = calibrate_decision_thresholds_for_target_coverage(
        probs=probs,
        y_true=y_true,
        logits=logits,
        f_ret=f_ret,
        imbalance=imbalance,
        rule="flat_bias",
        base_params=base_params,
        target_coverage=0.99,
        target_tolerance=0.005,
        min_coverage=0.99,
        max_coverage=1.0,
        quantiles=[0.50, 0.60, 0.70, 0.80, 0.90],
        opt_metric="net_edge_total",
        directional_base="predicted",
        fee_bps=0.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
    )

    assert result["used_fallback"] is True
    assert result["selected_thresholds"]["decision_confidence"] == base_params["decision_confidence"]
    assert result["selected_thresholds"]["decision_hold_threshold"] == base_params["decision_hold_threshold"]


def test_apply_decision_rule_threshold_overrides_backward_compatible():
    probs = np.array(
        [
            [0.20, 0.70, 0.10],
            [0.15, 0.40, 0.45],
            [0.80, 0.10, 0.10],
        ],
        dtype=np.float64,
    )
    kwargs = {
        "decision_confidence": 0.5,
        "decision_hold_threshold": 0.6,
        "flat_prob_threshold": 0.34,
        "up_prob_threshold": 0.34,
        "down_prob_threshold": 0.34,
        "margin_threshold": 0.0,
    }

    pred_legacy = apply_decision_rule(probs, "flat_bias", **kwargs)
    pred_none_overrides = apply_decision_rule(probs, "flat_bias", **kwargs, threshold_overrides=None)
    pred_with_overrides = apply_decision_rule(
        probs,
        "flat_bias",
        **kwargs,
        threshold_overrides={"decision_confidence": 0.75, "decision_hold_threshold": 0.75},
    )

    assert np.array_equal(pred_legacy, pred_none_overrides)
    assert not np.array_equal(pred_legacy, pred_with_overrides)
