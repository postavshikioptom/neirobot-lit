"""
train_data.py — Data orchestration для train.py.
Вынесено из train.py в рамках задачи 322.5.

Порядок инициализации (нельзя менять):
  LOBDataLoader -> FeatureEngineer -> Labeler -> LOBDataset
  -> chronological split -> normalizer.fit(train only) -> DataLoaders -> class weights
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import LOBDataset, LOBDataLoader
from .features import FeatureEngineer
from .labels import Labeler
from .normalization import Normalizer, symlog_transform
from .train_module import TrainSubset
from .train_runtime import build_dataloader_kwargs


DEFAULT_SWEEP_HORIZONS = [10, 20, 50, 100]
DEFAULT_SWEEP_THRESHOLDS = [0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0010, 0.0015]
SWEEP_MIN_TRADE_SHARE = 0.05
SWEEP_MAX_TRADE_SHARE = 0.60
SWEEP_MIN_DIRECTION_SHARE = 0.02
SWEEP_MAX_FLAT_SHARE = 0.90


@dataclass
class SweepBaselineRow:
    horizon: int
    threshold: float
    share_flat: float
    share_up: float
    share_down: float
    trade_share: float
    row_time_seconds: float
    event_time_seconds: float | None
    median_spread_bps: float
    threshold_bps: float
    threshold_to_spread_ratio: float
    subspread_target: bool
    sample_count: int
    is_shortlisted: bool = False
    shortlist_rank: int | None = None
    mini_train_mcc: float | None = None
    mini_train_coverage_directional: float | None = None
    mini_train_net_edge_total: float | None = None


@dataclass
class SweepDynamicReference:
    horizon: int
    threshold_mode: str
    share_flat: float
    share_up: float
    share_down: float
    unsafe_reference: bool
    note: str


@dataclass
class SweepBaselineArtifacts:
    grid: list[SweepBaselineRow]
    shortlist: list[dict]
    dynamic_threshold_reference: SweepDynamicReference
    generated_at_utc: str
    use_event_rows: bool
    symbol: str
    horizons: list[int]
    thresholds: list[float]


@dataclass
class PreparedTrainingData:
    df: object
    normalizer: object
    full_dataset: object
    train_ds: object
    val_ds: object
    test_ds: object
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_weights: np.ndarray
    past_returns_lags: list
    in_channels: int
    num_horizons: int
    horizon_weights: object  # list[float] | None
    regime_detector: object
    regime_weights: object
    num_regimes: int
    # Вспомогательные поля для пересоздания датасетов (Optuna)
    n_past_returns: int = 0
    time_weighting_params: dict = field(default_factory=dict)
    label_columns: list[str] = field(default_factory=list)
    class_weight_metadata: dict = field(default_factory=dict)
    split_artifacts: dict = field(default_factory=dict)
    effective_threshold_summary: dict = field(default_factory=dict)


def _parse_past_returns_lags(raw: str) -> list:
    """Парсит строку лагов past returns в список int."""
    return [int(x.strip()) for x in raw.split(',')]


def parse_int_sweep(raw_value: str | None, default_values: list[int]) -> list[int]:
    """Parse comma-separated int sweep config or return default grid."""
    if raw_value is None or raw_value.strip() == "":
        return list(default_values)
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_float_sweep(raw_value: str | None, default_values: list[float]) -> list[float]:
    """Parse comma-separated float sweep config or return default grid."""
    if raw_value is None or raw_value.strip() == "":
        return list(default_values)
    return [float(part.strip()) for part in raw_value.split(",") if part.strip()]


def resolve_sweep_grid(args) -> tuple[list[int], list[float]]:
    horizons = parse_int_sweep(getattr(args, "horizon_sweep", None), DEFAULT_SWEEP_HORIZONS)
    thresholds = parse_float_sweep(getattr(args, "threshold_sweep", None), DEFAULT_SWEEP_THRESHOLDS)
    if bool(getattr(args, "narrow_threshold_sweep", False)):
        thresholds = build_narrow_threshold_candidates(
            center=float(getattr(args, "threshold", 0.0005)),
            span=float(getattr(args, "threshold_sweep_span", 0.0002)),
            step=float(getattr(args, "threshold_sweep_step", 0.0001)),
        )
    return horizons, thresholds


def build_narrow_threshold_candidates(*, center: float, span: float, step: float) -> list[float]:
    """Build local threshold candidates around center to avoid broad search."""
    if span <= 0.0:
        return [float(center)]
    half_span = span / 2.0
    left = max(0.0, center - half_span)
    right = max(left, center + half_span)
    values = np.arange(left, right + (0.5 * step), step, dtype=np.float64)
    candidates = sorted({float(np.round(item, 8)) for item in values if item >= 0.0})
    return candidates or [float(center)]


def is_sweep_mode(args) -> bool:
    return (
        getattr(args, "horizon_sweep", None) is not None
        or getattr(args, "threshold_sweep", None) is not None
        or getattr(args, "sweep_baseline_path", None) is not None
    )


def clone_args_with_overrides(args, **overrides):
    values = vars(args).copy()
    values.update(overrides)
    return type(args)(**values)


def select_event_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only rows where feat_update_id changes, preserving chronological order."""
    if "feat_update_id" not in df.columns:
        return df
    return df.filter(pl.col("feat_update_id").diff().fill_null(1).ne(0))


def load_feature_frame(args, paths, *, use_event_rows: bool = False) -> pl.DataFrame:
    """Load raw LOB data and run feature engineering once for sweep/train orchestration."""
    print(f"Loading data for {args.symbol} from {paths.data_path}...")
    loader = LOBDataLoader(str(paths.data_path), args.symbol)
    df = loader.load_data(lazy=False)

    print("Engineering features...")
    fe = FeatureEngineer(n_levels=50)
    feature_df = fe.transform(df)

    if use_event_rows:
        feature_df = select_event_rows(feature_df)

    return feature_df


def build_labeled_frame(feature_df: pl.DataFrame, *, horizon: int | list[int], threshold: float,
                        dynamic_threshold: bool = False, args=None) -> pl.DataFrame:
    """Apply Labeler on an already engineered feature frame."""
    if getattr(args, "label_mode", "legacy_mid_return") == "execution_mid_return" and dynamic_threshold:
        raise ValueError(
            "dynamic_threshold разрешен только для legacy/debug режима. "
            "Сочетание execution_mid_return + dynamic_threshold запрещено."
        )
    labeler = Labeler(
        horizon=horizon,
        threshold=threshold,
        dynamic_threshold=dynamic_threshold,
        label_mode=getattr(args, "label_mode", "legacy_mid_return"),
        time_mode=getattr(args, "time_mode", "row"),
        event_time_column=getattr(args, "event_time_column", "feat_update_id"),
        cost_floor_bps=getattr(args, "cost_floor_bps", 0.0),
        fee_bps=getattr(args, "fee_bps", 0.0),
        slippage_bps=getattr(args, "slippage_bps", 0.0),
        use_spread_floor=getattr(args, "use_spread_floor", False),
    )
    return labeler.add_labels(feature_df)


def _label_contract_from_args(args) -> dict:
    return {
        "label_mode": getattr(args, "label_mode", "legacy_mid_return"),
        "time_mode": getattr(args, "time_mode", "row"),
        "event_time_column": getattr(args, "event_time_column", "feat_update_id"),
        "dynamic_threshold": bool(getattr(args, "dynamic_threshold", False)),
        "threshold": float(getattr(args, "threshold", 0.0005)),
        "cost_floor_bps": float(getattr(args, "cost_floor_bps", 0.0)),
        "fee_bps": float(getattr(args, "fee_bps", 0.0)),
        "slippage_bps": float(getattr(args, "slippage_bps", 0.0)),
        "use_spread_floor": bool(getattr(args, "use_spread_floor", False)),
        "effective_threshold_summary": _effective_threshold_summary_from_args(args),
    }


def _effective_threshold_summary_from_args(args) -> dict:
    static_threshold = float(getattr(args, "threshold", 0.0005))
    cost_floor_effective = (
        float(getattr(args, "cost_floor_bps", 0.0))
        + 2.0 * float(getattr(args, "fee_bps", 0.0))
        + float(getattr(args, "slippage_bps", 0.0))
    ) / 10000.0
    return {
        "static_threshold": static_threshold,
        "cost_floor_effective": cost_floor_effective,
        "effective_floor_without_spread": max(static_threshold, cost_floor_effective),
        "use_spread_floor": bool(getattr(args, "use_spread_floor", False)),
    }


def _resolve_dataset_label_columns(full_dataset) -> list[str]:
    label_cols = list(getattr(full_dataset, "label_cols", []) or [])
    return label_cols or ["label"]


def _resolve_train_label_indices(full_dataset, train_ds) -> np.ndarray:
    dataset_indices = np.asarray(train_ds.indices, dtype=np.int64)
    label_indices = dataset_indices + int(full_dataset.seq_len) - 1
    if label_indices.size and label_indices[-1] >= len(full_dataset.labels):
        raise ValueError(
            "Train split label indices exceed dataset.labels length. "
            "Cannot compute class weights safely."
        )
    return label_indices


def _extract_label_shares(labeled_df: pl.DataFrame, label_col: str = "label") -> tuple[float, float, float, int]:
    valid = labeled_df.filter(pl.col(label_col) != -100)
    sample_count = valid.height
    if sample_count == 0:
        return 0.0, 0.0, 0.0, 0

    counts = valid.select(
        [
            (pl.col(label_col) == 0).sum().alias("flat"),
            (pl.col(label_col) == 1).sum().alias("up"),
            (pl.col(label_col) == 2).sum().alias("down"),
        ]
    ).row(0)
    share_flat = float(counts[0] / sample_count)
    share_up = float(counts[1] / sample_count)
    share_down = float(counts[2] / sample_count)
    return share_flat, share_up, share_down, sample_count


def _median_step_seconds(df: pl.DataFrame) -> float | None:
    if "timestamp_ms" not in df.columns or df.height < 2:
        return None
    diff_series = df.select(pl.col("timestamp_ms").diff().drop_nulls().alias("delta_ms")).get_column("delta_ms")
    if diff_series.len() == 0:
        return None
    return float(diff_series.median() / 1000.0)


def _median_spread_bps(df: pl.DataFrame) -> float:
    spread_expr = (((pl.col("feat_ask_p_0") - pl.col("feat_bid_p_0")) / pl.col("mid_price")) * 10000.0).alias("spread_bps")
    return float(df.select(spread_expr.median().alias("median_spread_bps")).item())


def collect_sweep_baseline(feature_df: pl.DataFrame, *, horizons: list[int], thresholds: list[float],
                           use_event_rows: bool = False, args=None) -> SweepBaselineArtifacts:
    """Collect reproducible baseline grid for horizon x threshold."""
    working_df = select_event_rows(feature_df) if use_event_rows else feature_df
    event_df = select_event_rows(feature_df)
    row_step_seconds = _median_step_seconds(working_df) or 0.0
    event_step_seconds = _median_step_seconds(event_df)
    median_spread_bps = _median_spread_bps(working_df)

    grid: list[SweepBaselineRow] = []
    for horizon in horizons:
        for threshold in thresholds:
            labeled_df = build_labeled_frame(working_df, horizon=horizon, threshold=threshold, args=args)
            share_flat, share_up, share_down, sample_count = _extract_label_shares(labeled_df)
            threshold_bps = float(threshold * 10000.0)
            ratio = float(threshold_bps / median_spread_bps) if median_spread_bps > 0 else 0.0
            grid.append(
                SweepBaselineRow(
                    horizon=horizon,
                    threshold=float(threshold),
                    share_flat=share_flat,
                    share_up=share_up,
                    share_down=share_down,
                    trade_share=share_up + share_down,
                    row_time_seconds=float(horizon * row_step_seconds),
                    event_time_seconds=(float(horizon * event_step_seconds) if event_step_seconds is not None else None),
                    median_spread_bps=median_spread_bps,
                    threshold_bps=threshold_bps,
                    threshold_to_spread_ratio=ratio,
                    subspread_target=bool(ratio < 1.0),
                    sample_count=sample_count,
                )
            )

    dynamic_df = build_labeled_frame(working_df, horizon=100, threshold=0.0005, dynamic_threshold=True)
    dyn_flat, dyn_up, dyn_down, _ = _extract_label_shares(dynamic_df)
    dynamic_reference = SweepDynamicReference(
        horizon=100,
        threshold_mode="rolling_std_x_0_5",
        share_flat=dyn_flat,
        share_up=dyn_up,
        share_down=dyn_down,
        unsafe_reference=True,
        note="Unsafe reference only: dynamic threshold must not replace static grid default.",
    )

    shortlisted = shortlist_sweep_candidates(grid, 0)
    return SweepBaselineArtifacts(
        grid=grid,
        shortlist=[
            {
                "candidate_id": _candidate_id(row.horizon, row.threshold),
                "candidate_rank": row.shortlist_rank,
                "horizon": row.horizon,
                "threshold": row.threshold,
                "trade_share": row.trade_share,
                "share_flat": row.share_flat,
                "share_up": row.share_up,
                "share_down": row.share_down,
                "threshold_to_spread_ratio": row.threshold_to_spread_ratio,
                "mini_train_mcc": row.mini_train_mcc,
                "mini_train_coverage_directional": row.mini_train_coverage_directional,
                "mini_train_net_edge_total": row.mini_train_net_edge_total,
            }
            for row in shortlisted
        ],
        dynamic_threshold_reference=dynamic_reference,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        use_event_rows=use_event_rows,
        symbol="unknown",
        horizons=list(horizons),
        thresholds=list(thresholds),
    )


def shortlist_sweep_candidates(grid: list[SweepBaselineRow], topk: int) -> list[SweepBaselineRow]:
    """Pick top-k static-grid candidates with enough directional share and no subspread target."""
    if topk <= 0:
        return []

    filtered = [
        row for row in grid
        if not row.subspread_target
        and SWEEP_MIN_TRADE_SHARE <= row.trade_share <= SWEEP_MAX_TRADE_SHARE
        and row.share_flat <= SWEEP_MAX_FLAT_SHARE
        and row.share_up >= SWEEP_MIN_DIRECTION_SHARE
        and row.share_down >= SWEEP_MIN_DIRECTION_SHARE
    ]

    filtered.sort(
        key=lambda row: (
            abs(row.share_up - row.share_down),
            -row.trade_share,
            row.threshold_to_spread_ratio,
            row.horizon,
            row.threshold,
        )
    )
    shortlisted = filtered[:topk]
    for rank, row in enumerate(shortlisted, start=1):
        row.is_shortlisted = True
        row.shortlist_rank = rank
    return shortlisted


def export_sweep_baseline(paths, args, artifacts: SweepBaselineArtifacts) -> tuple[Path, Path]:
    """Write sweep baseline to CSV and JSON."""
    base_path = Path(args.sweep_baseline_path) if args.sweep_baseline_path else paths.base_path / "docs" / "sweep_baseline"
    if not base_path.is_absolute():
        base_path = paths.base_path / base_path
    base_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".json")
    rows_dict = [asdict(row) for row in artifacts.grid]

    pl.DataFrame(rows_dict).write_csv(csv_path)
    json_path.write_text(json.dumps(asdict(artifacts), indent=2, ensure_ascii=False), encoding="utf-8")
    baselines_path = Path(__file__).parent / "baselines.md"
    baselines_path.write_text(render_baselines_markdown(artifacts), encoding="utf-8")
    train_logs_path = paths.base_path / "docs" / "train_logs.md"
    with train_logs_path.open("a", encoding="utf-8") as handle:
        handle.write(render_sweep_train_log_entry(artifacts, csv_path, json_path))
    return csv_path, json_path


def render_baselines_markdown(artifacts: SweepBaselineArtifacts) -> str:
    """Render baselines.md content from sweep artifacts."""
    lines = [
        "# Baseline Sweep",
        "",
        f"- Generated at (UTC): `{artifacts.generated_at_utc}`",
        f"- Event rows mode: `{str(artifacts.use_event_rows).lower()}`",
        "",
        "## Static Grid",
        "",
        "| horizon | threshold | flat% | up% | down% | trade% | row_time_seconds | event_time_seconds | threshold_bps | median_spread_bps | threshold_to_spread_ratio | subspread_target | mini_train_mcc | mini_train_coverage_directional | mini_train_net_edge_total |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(artifacts.grid, key=lambda item: (item.horizon, item.threshold)):
        lines.append(
            "| {h} | {thr:.4f} | {sf:.4f} | {su:.4f} | {sd:.4f} | {ts:.4f} | {rt:.4f} | {et} | {tb:.2f} | {ms:.2f} | {ratio:.4f} | {sub} | {mcc} | {cov} | {edge} |".format(
                h=row.horizon,
                thr=row.threshold,
                sf=row.share_flat * 100.0,
                su=row.share_up * 100.0,
                sd=row.share_down * 100.0,
                ts=row.trade_share * 100.0,
                rt=row.row_time_seconds,
                et=f"{row.event_time_seconds:.4f}" if row.event_time_seconds is not None else "n/a",
                tb=row.threshold_bps,
                ms=row.median_spread_bps,
                ratio=row.threshold_to_spread_ratio,
                sub=str(row.subspread_target).lower(),
                mcc=f"{row.mini_train_mcc:.4f}" if row.mini_train_mcc is not None else "n/a",
                cov=f"{row.mini_train_coverage_directional:.4f}" if row.mini_train_coverage_directional is not None else "n/a",
                edge=f"{row.mini_train_net_edge_total:.6f}" if row.mini_train_net_edge_total is not None else "n/a",
            )
        )

    dyn = artifacts.dynamic_threshold_reference
    lines.extend(
        [
            "",
            "## Dynamic Threshold Reference",
            "",
            f"- horizon: `{dyn.horizon}`",
            f"- threshold_mode: `{dyn.threshold_mode}`",
            f"- unsafe_reference: `{str(dyn.unsafe_reference).lower()}`",
            f"- share_flat / share_up / share_down: `{dyn.share_flat * 100.0:.2f} / {dyn.share_up * 100.0:.2f} / {dyn.share_down * 100.0:.2f}`",
            f"- note: {dyn.note}",
            "",
            "## Shortlist",
            "",
        ]
    )

    shortlist = [row for row in artifacts.grid if row.is_shortlisted]
    if shortlist:
        for row in sorted(shortlist, key=lambda item: item.shortlist_rank or 0):
            lines.append(
                f"- rank {row.shortlist_rank}: h={row.horizon}, thr={row.threshold:.4f}, "
                f"trade_share={row.trade_share * 100.0:.2f}%, mcc={row.mini_train_mcc if row.mini_train_mcc is not None else 'n/a'}, "
                f"coverage={row.mini_train_coverage_directional if row.mini_train_coverage_directional is not None else 'n/a'}, "
                f"net_edge={row.mini_train_net_edge_total if row.mini_train_net_edge_total is not None else 'n/a'}"
            )
    else:
        lines.append("- shortlist is empty")

    return "\n".join(lines) + "\n"


def render_sweep_train_log_entry(artifacts: SweepBaselineArtifacts, csv_path: Path, json_path: Path) -> str:
    dyn = artifacts.dynamic_threshold_reference
    lines = [
        "",
        f"## Sweep baseline 326 | Дата: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        f"- Rows: {len(artifacts.grid)}",
        f"- Event rows only: {str(artifacts.use_event_rows).lower()}",
        f"- Dynamic threshold unsafe reference (h=100): {dyn.share_flat * 100.0:.2f} / {dyn.share_up * 100.0:.2f} / {dyn.share_down * 100.0:.2f}",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{json_path}`",
    ]
    shortlist = [row for row in artifacts.grid if row.is_shortlisted]
    if shortlist:
        lines.append("- Shortlist:")
        for row in sorted(shortlist, key=lambda item: item.shortlist_rank or 0):
            lines.append(
                f"  - #{row.shortlist_rank}: h={row.horizon}, thr={row.threshold:.4f}, "
                f"trade={row.trade_share * 100.0:.2f}%, mini_mcc={row.mini_train_mcc if row.mini_train_mcc is not None else 'n/a'}"
            )
    return "\n".join(lines) + "\n"


def _parse_sweep_values(raw_value, cast, default_values):
    """Парсит CSV-сетку из CLI или возвращает дефолтные значения."""
    if raw_value is None:
        return list(default_values)

    parts = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    if not parts:
        return list(default_values)
    return [cast(part) for part in parts]


def load_training_dataframe(args, paths):
    """Загружает исходный parquet и прогоняет его через FeatureEngineer."""
    print(f"Loading data for {args.symbol} from {paths.data_path}...")
    loader = LOBDataLoader(str(paths.data_path), args.symbol)
    df = loader.load_data(lazy=False)

    print("Engineering features...")
    fe = FeatureEngineer(n_levels=50)
    return fe.transform(df)


def _detect_update_id_column(df) -> str | None:
    for col_name in ("feat_update_id", "last_update_id"):
        if col_name in df.columns:
            return col_name
    return None


def _build_event_rows(df):
    """Оставляет только строки, где update_id меняется."""
    update_col = _detect_update_id_column(df)
    if update_col is None or df.height == 0:
        return None

    update_ids = df[update_col].to_numpy()
    event_mask = np.ones(len(update_ids), dtype=bool)
    event_mask[1:] = update_ids[1:] != update_ids[:-1]
    return df.filter(pl.Series("event_mask", event_mask))


def _compute_class_shares(labels: np.ndarray) -> tuple[float, float, float]:
    valid = labels[labels != -100]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0

    total = float(len(valid))
    return (
        float(np.sum(valid == 0) / total),
        float(np.sum(valid == 1) / total),
        float(np.sum(valid == 2) / total),
    )


def _compute_horizon_time_seconds(df, horizon: int) -> float | None:
    if df is None or "timestamp_ms" not in df.columns or df.height <= horizon:
        return None

    timestamps = df["timestamp_ms"].to_numpy()
    deltas = (timestamps[horizon:] - timestamps[:-horizon]) / 1000.0
    return float(np.median(deltas)) if len(deltas) > 0 else None


def _compute_median_spread_bps(df) -> float:
    spread_bps = (df["feat_ask_p_0"].to_numpy() - df["feat_bid_p_0"].to_numpy()) * 10000.0
    return float(np.median(spread_bps))


def _candidate_id(horizon: int, threshold: float) -> str:
    return f"h{horizon}_thr{threshold:.4f}".replace(".", "p")


def collect_single_sweep_point(df, horizon: int, threshold: float, use_event_rows: bool = False, event_df=None) -> SweepBaselineRow:
    """Собирает одну точку sweep для статического threshold."""
    event_df = event_df if event_df is not None else _build_event_rows(df)
    labels_df = event_df if use_event_rows and event_df is not None else df

    labeler = Labeler(horizon=horizon, threshold=threshold, dynamic_threshold=False)
    labeled_df = labeler.add_labels(labels_df.lazy()).collect()
    labels = labeled_df["label"].to_numpy()

    share_flat, share_up, share_down = _compute_class_shares(labels)
    trade_share = float(share_up + share_down)
    median_spread_bps = _compute_median_spread_bps(df)
    threshold_bps = float(threshold * 10000.0)
    threshold_to_spread_ratio = float(threshold_bps / median_spread_bps) if median_spread_bps > 0 else 0.0

    return SweepBaselineRow(
        horizon=int(horizon),
        threshold=float(threshold),
        share_flat=share_flat,
        share_up=share_up,
        share_down=share_down,
        trade_share=trade_share,
        row_time_seconds=float(_compute_horizon_time_seconds(df, horizon) or 0.0),
        event_time_seconds=_compute_horizon_time_seconds(event_df, horizon),
        median_spread_bps=median_spread_bps,
        threshold_bps=threshold_bps,
        threshold_to_spread_ratio=threshold_to_spread_ratio,
        subspread_target=bool(threshold_to_spread_ratio < 1.0),
        sample_count=int(len(labels[labels != -100])),
    )


def collect_dynamic_threshold_reference(df, horizon: int = 100) -> SweepDynamicReference:
    """Фиксирует dynamic threshold отдельно от static grid."""
    labeler = Labeler(horizon=horizon, threshold=0.0005, dynamic_threshold=True, window=1000, K=0.5)
    labeled_df = labeler.add_labels(df.lazy()).collect()
    labels = labeled_df["label"].to_numpy()
    share_flat, share_up, share_down = _compute_class_shares(labels)
    return SweepDynamicReference(
        horizon=int(horizon),
        threshold_mode="rolling_std_x_0.5",
        share_flat=share_flat,
        share_up=share_up,
        share_down=share_down,
        unsafe_reference=True,
        note="Unsafe reference only: rolling_std * 0.5 не должен становиться новым default.",
    )


def _shortlist_score(row: SweepBaselineRow) -> float:
    return (
        abs(row.trade_share - 0.25)
        + abs(row.share_up - row.share_down)
        + max(0.0, row.share_flat - 0.75)
    )


def shortlist_sweep_candidates(grid_rows: list[SweepBaselineRow], topk: int) -> list[SweepBaselineRow]:
    """Отбирает shortlist top-k по явным фильтрам задачи."""
    if topk <= 0:
        return []

    filtered = [
        row for row in grid_rows
        if not row.subspread_target
        and SWEEP_MIN_TRADE_SHARE <= row.trade_share <= SWEEP_MAX_TRADE_SHARE
        and row.share_up >= SWEEP_MIN_DIRECTION_SHARE
        and row.share_down >= SWEEP_MIN_DIRECTION_SHARE
    ]

    ranked = sorted(filtered, key=lambda row: (_shortlist_score(row), row.horizon, row.threshold))
    shortlist = []
    for rank, row in enumerate(ranked[:topk], start=1):
        row.is_shortlisted = True
        row.shortlist_rank = rank
        shortlist.append(row)
    return shortlist


def _collect_sweep_baseline_from_args(args, paths) -> SweepBaselineArtifacts:
    """Legacy helper kept for compatibility; main runner uses collect_sweep_baseline(feature_df, ...)."""
    horizons = _parse_sweep_values(args.horizon_sweep, int, DEFAULT_SWEEP_HORIZONS)
    thresholds = _parse_sweep_values(args.threshold_sweep, float, DEFAULT_SWEEP_THRESHOLDS)
    df = load_training_dataframe(args, paths)
    event_df = _build_event_rows(df)

    grid_rows = []
    for horizon in horizons:
        for threshold in thresholds:
            grid_rows.append(
                collect_single_sweep_point(
                    df,
                    horizon=horizon,
                    threshold=threshold,
                    use_event_rows=args.sweep_use_event_rows,
                    event_df=event_df,
                )
            )

    shortlisted = shortlist_sweep_candidates(grid_rows, args.sweep_train_topk)
    return SweepBaselineArtifacts(
        grid=grid_rows,
        shortlist=[
            {
                "candidate_id": _candidate_id(row.horizon, row.threshold),
                "candidate_rank": row.shortlist_rank,
                "horizon": row.horizon,
                "threshold": row.threshold,
                "trade_share": row.trade_share,
                "share_flat": row.share_flat,
                "share_up": row.share_up,
                "share_down": row.share_down,
                "threshold_to_spread_ratio": row.threshold_to_spread_ratio,
                "mini_train_mcc": row.mini_train_mcc,
                "mini_train_coverage_directional": row.mini_train_coverage_directional,
                "mini_train_net_edge_total": row.mini_train_net_edge_total,
            }
            for row in shortlisted
        ],
        dynamic_threshold_reference=collect_dynamic_threshold_reference(df, horizon=100),
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        use_event_rows=bool(args.sweep_use_event_rows),
        symbol=args.symbol,
        horizons=list(horizons),
        thresholds=list(thresholds),
    )


def _resolve_sweep_output_paths(base_path: Path, baseline_path: str | None) -> tuple[Path, Path, Path]:
    json_path = Path(baseline_path) if baseline_path else base_path / "docs" / "sweep_baseline.json"
    if not json_path.is_absolute():
        json_path = base_path / json_path
    if json_path.suffix.lower() != ".json":
        json_path = json_path.with_suffix(".json")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = json_path.with_suffix(".csv")
    if baseline_path:
        baselines_md_path = json_path.parent / "baselines.md"
    else:
        baselines_md_path = Path(__file__).parent / "baselines.md"
    return json_path, csv_path, baselines_md_path


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_optional(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _build_baselines_markdown(artifacts: SweepBaselineArtifacts) -> str:
    lines = [
        "# Baseline Sweep",
        "",
        f"- generated_at_utc: `{artifacts.generated_at_utc}`",
        f"- use_event_rows_for_labels: `{str(artifacts.use_event_rows).lower()}`",
        "",
        "## Static Grid",
        "",
        "| horizon | threshold | flat | up | down | trade_share | row_time_s | event_time_s | spread_bps | thr_bps | thr/spread | subspread | shortlist | mini_mcc | mini_cov | mini_edge |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|",
    ]

    for row in artifacts.grid:
        lines.append(
            f"| {row.horizon} | {row.threshold:.4f} | {_format_pct(row.share_flat)} | {_format_pct(row.share_up)} | {_format_pct(row.share_down)} | "
            f"{_format_pct(row.trade_share)} | {row.row_time_seconds:.3f} | {_format_optional(row.event_time_seconds, 3)} | "
            f"{row.median_spread_bps:.2f} | {row.threshold_bps:.2f} | {row.threshold_to_spread_ratio:.3f} | "
            f"{str(row.subspread_target).lower()} | {str(row.is_shortlisted).lower()} | {_format_optional(row.mini_train_mcc)} | "
            f"{_format_optional(row.mini_train_coverage_directional)} | {_format_optional(row.mini_train_net_edge_total)} |"
        )

    lines.extend([
        "",
        "## Dynamic Threshold Reference",
        "",
        f"- horizon: `{artifacts.dynamic_threshold_reference.horizon}`",
        f"- share_flat/share_up/share_down: `{_format_pct(artifacts.dynamic_threshold_reference.share_flat)} / {_format_pct(artifacts.dynamic_threshold_reference.share_up)} / {_format_pct(artifacts.dynamic_threshold_reference.share_down)}`",
        f"- note: {artifacts.dynamic_threshold_reference.note}",
        "",
        "## Shortlist For 327",
        "",
    ])

    shortlisted = [row for row in artifacts.grid if row.is_shortlisted]
    if shortlisted:
        for row in sorted(shortlisted, key=lambda item: item.shortlist_rank or 999):
            lines.append(
                f"- #{row.shortlist_rank} `{_candidate_id(row.horizon, row.threshold)}`: "
                f"h={row.horizon}, thr={row.threshold:.4f}, trade_share={_format_pct(row.trade_share)}, "
                f"balance={_format_pct(row.share_flat)}/{_format_pct(row.share_up)}/{_format_pct(row.share_down)}, "
                f"thr/spread={row.threshold_to_spread_ratio:.3f}, mini_mcc={_format_optional(row.mini_train_mcc)}"
            )
    else:
        lines.append("- shortlist пуст: ни одна точка не прошла фильтры.")

    return "\n".join(lines) + "\n"


def persist_sweep_baseline(artifacts: SweepBaselineArtifacts, base_path: Path, baseline_path: str | None = None) -> dict:
    """Пишет sweep baseline в CSV, JSON и markdown-отчеты."""
    json_path, csv_path, baselines_md_path = _resolve_sweep_output_paths(base_path, baseline_path)
    train_logs_path = base_path / "docs" / "train_logs.md"

    grid_payload = [asdict(row) for row in artifacts.grid]
    pl.DataFrame(grid_payload).write_csv(csv_path)
    json_path.write_text(json.dumps(asdict(artifacts), ensure_ascii=False, indent=2), encoding="utf-8")
    baselines_md_path.write_text(_build_baselines_markdown(artifacts), encoding="utf-8")

    existing_train_logs = train_logs_path.read_text(encoding="utf-8") if train_logs_path.exists() else ""
    marker = "## Задача 326 | Baseline sweep"
    if marker not in existing_train_logs:
        shortlist_ids = [f"`{_candidate_id(row.horizon, row.threshold)}`" for row in artifacts.grid if row.is_shortlisted]
        block = (
            "\n## Задача 326 | Baseline sweep\n\n"
            f"- Артефакты: `{csv_path.name}`, `{json_path.name}`, `{baselines_md_path.name}`\n"
            f"- Grid points: `{len(artifacts.grid)}`\n"
            f"- Shortlist: {', '.join(shortlist_ids) if shortlist_ids else 'empty'}\n"
        )
        train_logs_path.write_text(existing_train_logs.rstrip() + block + "\n", encoding="utf-8")

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "baselines_md_path": str(baselines_md_path),
        "train_logs_path": str(train_logs_path),
    }


def build_full_dataset(df, args, past_returns_lags, winsor_limits, normalizer,
                        regime_detector, time_weighting_params):
    """Создаёт LOBDataset в memory режиме."""
    n_past_returns = len(past_returns_lags)
    return LOBDataset(
        df,
        seq_len=args.seq_len,
        n_past_returns=n_past_returns,
        past_returns_lags=past_returns_lags,
        data_mode="memory",
        is_train=False,  # val/test работают в eval mode; train через TrainSubset
        augment_prob=args.augment_prob,
        use_symmetric_flip=args.use_symmetric_flip,
        volume_jitter_range=args.volume_jitter_range,
        aug_seed=args.aug_seed,
        regime_detector=regime_detector,
        regime_window=1000,
        scaler_type=args.scaler_type,
        winsor_limits=winsor_limits,
        scale_multiplier=args.scale_multiplier,
        normalizer=normalizer,
        **time_weighting_params
    )


def split_dataset_chronologically(full_dataset):
    """Хронологическое разделение 70/15/15."""
    total_len = len(full_dataset)
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_len))

    train_ds = TrainSubset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)
    return train_ds, val_ds, test_ds, train_indices, val_indices, test_indices


def _resolve_max_horizon(horizons) -> int:
    if isinstance(horizons, (list, tuple)):
        return max(int(h) for h in horizons) if horizons else 1
    return int(horizons)


def _resolve_max_lag(args) -> int:
    lags = _parse_past_returns_lags(getattr(args, "past_returns_lags", "10,50,100"))
    return max(lags) if lags else 0


def _log_txy_lengths(stage: str, *, timestamps_len: int | None, x_len: int | None, y_len: int | None,
                     extra: str | None = None) -> None:
    msg = f"[TXY] {stage}: timestamps={timestamps_len}, X={x_len}, y={y_len}"
    if extra:
        msg += f" | {extra}"
    print(msg)


def _resolve_sequence_lengths(full_dataset) -> tuple[int, int]:
    seq_len = int(getattr(full_dataset, "seq_len", 1))
    max_lag = _resolve_max_lag_from_dataset(full_dataset)
    x_raw = getattr(full_dataset, "x_raw", None)
    labels = getattr(full_dataset, "labels", None)

    if x_raw is None:
        x_len = len(full_dataset)
    else:
        x_len = max(0, (len(x_raw) - seq_len + 1) - max_lag)

    if labels is None:
        y_len = len(full_dataset)
    else:
        y_len = max(0, (len(labels) - seq_len + 1) - max_lag)

    return x_len, y_len


def _resolve_max_lag_from_dataset(full_dataset) -> int:
    lags = getattr(full_dataset, "past_returns_lags", None)
    if lags:
        return max(int(lag) for lag in lags)
    return 0


def _aligned_timestamps_for_dataset(full_dataset) -> np.ndarray:
    timestamps = np.asarray(full_dataset.get_timestamps(), dtype=np.int64)
    total_len = len(full_dataset)
    if timestamps.size == total_len:
        return timestamps

    seq_len = int(getattr(full_dataset, "seq_len", 1))
    max_lag = _resolve_max_lag_from_dataset(full_dataset)
    offset = max(0, seq_len - 1) + max_lag
    expected_raw = total_len + offset
    if timestamps.size == expected_raw:
        return timestamps[offset:]

    raise ValueError(
        "Timestamps length mismatch: "
        f"timestamps={timestamps.size}, dataset={total_len}, seq_len={seq_len}, max_lag={max_lag}"
    )


def _assert_txy_invariant(stage: str, full_dataset, timestamps: np.ndarray) -> None:
    x_len, y_len = _resolve_sequence_lengths(full_dataset)
    if not (timestamps.size == x_len == y_len):
        raise ValueError(
            f"TXY invariant failed ({stage}): timestamps={timestamps.size}, X={x_len}, y={y_len}"
        )


def _safe_median_step_ms(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return 0.0
    deltas = np.diff(timestamps)
    if deltas.size == 0:
        return 0.0
    return float(np.median(deltas))


def _format_duration_hours(duration_ms: int) -> str:
    hours = float(duration_ms) / 1000.0 / 3600.0
    if hours >= 24.0:
        return f"{hours / 24.0:.2f}d ({hours:.2f}h)"
    return f"{hours:.2f}h"


def _build_holdout_masks(timestamps: np.ndarray, holdout_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    last_ts = int(timestamps[-1])
    test_start_ts = last_ts - holdout_ms + 1
    val_start_ts = test_start_ts - holdout_ms

    val_mask = (timestamps >= val_start_ts) & (timestamps < test_start_ts)
    test_mask = timestamps >= test_start_ts
    train_mask = timestamps < val_start_ts
    return train_mask, val_mask, test_mask, val_start_ts, test_start_ts


def _resolve_effective_holdout_ms(
    timestamps: np.ndarray,
    requested_holdout_ms: int,
    *,
    effective_purge_events: int,
) -> tuple[int, dict]:
    if timestamps.size < 2:
        return requested_holdout_ms, {
            "auto_adjusted": False,
            "aligned_span_ms": 0,
            "balanced_cap_ms": 0,
            "requested_holdout_ms": requested_holdout_ms,
        }

    span_ms = max(0, int(timestamps[-1]) - int(timestamps[0]))
    balanced_cap_ms = span_ms // 3
    effective_holdout_ms = requested_holdout_ms
    auto_adjusted = False

    min_window_ms = max(1, int(round(_safe_median_step_ms(timestamps))))
    if balanced_cap_ms > 0 and requested_holdout_ms > balanced_cap_ms:
        effective_holdout_ms = max(min_window_ms, balanced_cap_ms)
        auto_adjusted = effective_holdout_ms != requested_holdout_ms

    return effective_holdout_ms, {
        "auto_adjusted": auto_adjusted,
        "aligned_span_ms": span_ms,
        "balanced_cap_ms": balanced_cap_ms,
        "requested_holdout_ms": requested_holdout_ms,
        "effective_purge_events": int(effective_purge_events),
        "min_window_ms": int(min_window_ms),
    }


def _indices_range(indices: list[int], timestamps: np.ndarray) -> dict:
    if not indices:
        return {"start_idx": None, "end_idx": None, "start_ts": None, "end_ts": None}
    return {
        "start_idx": int(indices[0]),
        "end_idx": int(indices[-1]),
        "start_ts": int(timestamps[indices[0]]),
        "end_ts": int(timestamps[indices[-1]]),
    }


def _build_split_artifacts(
    strategy: str,
    timestamps: np.ndarray,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    effective_purge_events: int,
    embargo_events: int,
) -> dict:
    return {
        "strategy": strategy,
        "effective_purge_events": int(effective_purge_events),
        "embargo_events": int(embargo_events),
        "train_indices_count": int(len(train_indices)),
        "val_indices_count": int(len(val_indices)),
        "test_indices_count": int(len(test_indices)),
        "train_range": _indices_range(train_indices, timestamps),
        "val_range": _indices_range(val_indices, timestamps),
        "test_range": _indices_range(test_indices, timestamps),
    }


def split_dataset_purged_holdout(full_dataset, args, horizons):
    timestamps = _aligned_timestamps_for_dataset(full_dataset)
    total_len = len(full_dataset)
    x_len, y_len = _resolve_sequence_lengths(full_dataset)
    _log_txy_lengths("purge/before", timestamps_len=timestamps.size, x_len=x_len, y_len=y_len)
    if total_len < 10:
        raise ValueError("Dataset is too small for purged holdout split.")

    max_horizon = _resolve_max_horizon(horizons)
    max_lag = _resolve_max_lag(args)
    effective_purge_events = max(
        int(getattr(args, "seq_len", 1)),
        int(max_horizon),
        int(max_lag),
        int(getattr(args, "purge_buffer_events", 0)),
    )

    day_ms = 24 * 60 * 60 * 1000
    requested_holdout_days = float(getattr(args, "holdout_days", 1.0))
    requested_holdout_ms = int(requested_holdout_days * day_ms)
    holdout_ms, holdout_meta = _resolve_effective_holdout_ms(
        timestamps,
        requested_holdout_ms,
        effective_purge_events=effective_purge_events,
    )
    if holdout_meta["auto_adjusted"]:
        print(
            "[PURGED_HOLDOUT] Requested holdout window is too large for aligned data span. "
            f"requested={requested_holdout_days:.4f}d ({_format_duration_hours(requested_holdout_ms)}), "
            f"aligned_span={_format_duration_hours(holdout_meta['aligned_span_ms'])}, "
            f"effective_holdout={_format_duration_hours(holdout_ms)}, "
            f"effective_purge_events={effective_purge_events}"
        )

    train_mask, val_mask, test_mask, val_start_ts, test_start_ts = _build_holdout_masks(timestamps, holdout_ms)

    base_train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()

    if len(val_indices) == 0 or len(test_indices) == 0 or len(base_train_indices) == 0:
        suggested_holdout_days = holdout_meta["balanced_cap_ms"] / day_ms if holdout_meta["balanced_cap_ms"] > 0 else 0.0
        raise ValueError(
            "Purged holdout split produced empty train/val/test. "
            f"aligned_span={_format_duration_hours(holdout_meta['aligned_span_ms'])}, "
            f"requested_holdout={requested_holdout_days:.4f}d, "
            f"suggested_holdout_days<={suggested_holdout_days:.4f}. "
            "Increase dataset or reduce holdout_days."
        )

    median_step_ms = _safe_median_step_ms(timestamps)
    embargo_seconds = int(getattr(args, "embargo_seconds", 0))
    embargo_events_from_seconds = int(np.ceil((embargo_seconds * 1000.0) / median_step_ms)) if median_step_ms > 0 else 0
    embargo_events = max(int(getattr(args, "embargo_buffer_events", 0)), embargo_events_from_seconds)

    blocked = np.zeros(total_len, dtype=bool)
    boundary_specs = [
        (val_indices[0], val_indices[-1]),
        (test_indices[0], test_indices[-1]),
    ]
    for start_idx, end_idx in boundary_specs:
        left = max(0, start_idx - effective_purge_events)
        right = min(total_len, end_idx + 1 + embargo_events)
        blocked[left:right] = True

        if embargo_seconds > 0:
            start_ts = int(timestamps[start_idx])
            end_ts = int(timestamps[end_idx])
            blocked |= (timestamps >= (start_ts - embargo_seconds * 1000)) & (timestamps <= (end_ts + embargo_seconds * 1000))

    train_indices = [int(i) for i in base_train_indices if not blocked[int(i)]]
    if len(train_indices) == 0:
        effective_holdout_days = holdout_ms / day_ms
        raise ValueError(
            "Purged holdout removed all train samples. "
            f"effective_holdout_days={effective_holdout_days:.4f}, "
            f"effective_purge_events={effective_purge_events}, "
            f"embargo_events={embargo_events}. "
            "Reduce purge/embargo or holdout_days."
        )
    _log_txy_lengths(
        "purge/after",
        timestamps_len=timestamps.size,
        x_len=x_len,
        y_len=y_len,
        extra=f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}",
    )

    print(
        "Purged holdout split: "
        f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}, "
        f"effective_purge_events={effective_purge_events}, embargo_events={embargo_events}"
    )

    train_ds = TrainSubset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)
    artifacts = _build_split_artifacts(
        strategy="purged_holdout",
        timestamps=timestamps,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        effective_purge_events=effective_purge_events,
        embargo_events=embargo_events,
    )
    artifacts["requested_holdout_days"] = requested_holdout_days
    artifacts["effective_holdout_days"] = float(holdout_ms / day_ms)
    artifacts["aligned_span_hours"] = float(holdout_meta["aligned_span_ms"] / 1000.0 / 3600.0)
    artifacts["holdout_auto_adjusted"] = bool(holdout_meta["auto_adjusted"])
    return train_ds, val_ds, test_ds, train_indices, val_indices, test_indices, artifacts


def split_dataset_by_strategy(full_dataset, args, horizons):
    strategy = getattr(args, "split_strategy", "chronological")
    if strategy == "chronological":
        train_ds, val_ds, test_ds, train_indices, val_indices, test_indices = split_dataset_chronologically(full_dataset)
        timestamps = np.asarray(full_dataset.get_timestamps(), dtype=np.int64)
        artifacts = _build_split_artifacts(
            strategy="chronological",
            timestamps=timestamps,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            effective_purge_events=0,
            embargo_events=0,
        )
        return train_ds, val_ds, test_ds, train_indices, val_indices, test_indices, artifacts
    if strategy == "purged_holdout":
        timestamps = _aligned_timestamps_for_dataset(full_dataset)
        _assert_txy_invariant("pre split_dataset_purged_holdout", full_dataset, timestamps)
        result = split_dataset_purged_holdout(full_dataset, args, horizons)
        timestamps = _aligned_timestamps_for_dataset(full_dataset)
        _assert_txy_invariant("post split_dataset_purged_holdout", full_dataset, timestamps)
        return result
    raise ValueError(f"Unsupported split strategy in prepare_training_data: {strategy}")


def _fit_normalizer_on_train(full_dataset, train_ds, normalizer, args, winsor_limits):
    """Обучает нормализатор только на train-части (channel-space)."""
    print("\nFitting normalizer on original training set (channels-based)...")
    train_indices_for_fit = train_ds.indices
    train_channels_df = full_dataset._compute_channels_for_normalization(train_indices_for_fit)
    print(f"Static features dimension check: {train_channels_df.shape[1]} features (8 static channels × 50 levels, dynamic channels fitted separately)")

    # Задача 324.2: Используем полные кэши train-части, а не три суррогатных столбца
    # Берём все значения ofi_cache, delta_imb_cache, delta_spread_cache для train-индексов
    train_idx_arr = np.array(train_indices_for_fit)

    ofi_raw = full_dataset.ofi_cache[train_idx_arr]
    delta_imb_raw = full_dataset.delta_imb_cache[train_idx_arr]
    delta_spread_raw = full_dataset.delta_spread_cache[train_idx_arr]

    # Задача 324.3: Применяем symlog через ту же функцию что и в runtime (_apply_dynamic_transform).
    # Путь ИДЕНТИЧЕН: symlog_transform → transform_dynamic (median/iqr) → clamp[-4,4].
    # symlog_transform импортирован из normalization.py — единый источник.
    ofi_sym = symlog_transform(ofi_raw)
    delta_imb_sym = symlog_transform(delta_imb_raw)
    delta_spread_sym = symlog_transform(delta_spread_raw)

    dynamic_data = {
        "ofi": ofi_sym,
        "delta_imb": delta_imb_sym,
        "delta_spread": delta_spread_sym
    }

    # Задача 324.8: диагностика до fit — видим raw/sym распределение и будущий IQR
    print("\n" + "=" * 70)
    print("PRE-FIT DYNAMIC CHANNEL DIAGNOSTICS")
    print("=" * 70)
    for _name, _raw, _sym in [
        ("ofi", ofi_raw, ofi_sym),
        ("delta_imb", delta_imb_raw, delta_imb_sym),
        ("delta_spread", delta_spread_raw, delta_spread_sym),
    ]:
        _q25 = float(np.quantile(_sym, 0.25))
        _q75 = float(np.quantile(_sym, 0.75))
        _q10 = float(np.quantile(_sym, 0.10))
        _q90 = float(np.quantile(_sym, 0.90))
        _raw_iqr = _q75 - _q25
        _alt_iqr = _q90 - _q10
        _iqr_floor = 1e-3
        _iqr = _raw_iqr if np.isfinite(_raw_iqr) and _raw_iqr >= _iqr_floor else max(_alt_iqr, _iqr_floor)
        print(f"\n  [{_name}]")
        print(f"    raw:  p01={float(np.percentile(_raw, 1)):.6f}, p50={float(np.percentile(_raw, 50)):.6f}, p99={float(np.percentile(_raw, 99)):.6f}")
        print(f"    sym:  p01={float(np.percentile(_sym, 1)):.6f}, p50={float(np.percentile(_sym, 50)):.6f}, p99={float(np.percentile(_sym, 99)):.6f}")
        print(f"    будущий iqr: raw_iqr={_raw_iqr:.6f}, alt_iqr={_alt_iqr:.6f} → iqr={_iqr:.6f}")
    print("=" * 70 + "\n")

    normalizer.fit(train_channels_df, winsor_limits=winsor_limits, dynamic_data=dynamic_data)
    normalizer.save(scaler_type=args.scaler_type, winsor_limits=winsor_limits)
    print(f"✓ Normalizer fitted on {len(train_channels_df)} samples")

    # Задача 324.4: Агрегированная диагностика по всему train split после fit
    # Применяем полный pipeline (symlog → robust → clamp) и считаем статистику
    diag_metrics = _log_dynamic_train_diagnostics(
        {"ofi": ofi_sym, "delta_imb": delta_imb_sym, "delta_spread": delta_spread_sym},
        normalizer,
        clip_limit=4.0
    )

    # Задача 324.5: Hard guard — останавливаем обучение при плохом scale
    allow_bad = getattr(args, 'allow_bad_dynamic_scale', False)
    _check_dynamic_scale_guard(diag_metrics, allow_bad_scale=allow_bad)


def _log_dynamic_train_diagnostics(dynamic_sym: dict, normalizer, clip_limit: float = 4.0):
    """
    Задача 324.4 + 324.9: Агрегированная диагностика dynamic-каналов по всему train split.
    Печатается один раз после fit normalizer, до старта первой эпохи.
    Возвращает dict с метрик для последующего guard-check.

    Критически важно: pipeline должен быть ТОЧНО ТАКИМ ЖЕ, как в runtime (_apply_dynamic_transform):
      1. symlog already applied (passed as sym_arr)
      2. preclip (p0.01/p0.99) — если задано в dynamic_params
      3. robust normalize: (x - median) / (iqr + eps)  БЕЗ scale_multiplier (Задача 324.9)
      4. clamp [-4, 4]
    """
    print("\n" + "=" * 70)
    print("DYNAMIC CHANNEL TRAIN DIAGNOSTICS (после fit normalizer)")
    print("=" * 70)
    metrics = {}
    for name, sym_arr in dynamic_sym.items():
        p = normalizer.dynamic_params.get(name, {})
        median = p.get("median", 0.0)
        iqr = p.get("iqr", 1.0)
        eps = normalizer.eps

        # Задача 324.9: применяем preclip ДО нормализации (как в runtime _apply_dynamic_transform)
        preclip_low = p.get("preclip_low")
        preclip_high = p.get("preclip_high")
        x = sym_arr.copy()
        if preclip_low is not None and preclip_high is not None:
            x = np.clip(x, preclip_low, preclip_high)

        # Задача 324.9: для dynamic-каналов НЕ используем scale_multiplier (только iqr + eps)
        scale = iqr + eps
        normed = (x - median) / scale

        n = len(normed)
        below = np.sum(normed < -clip_limit)
        above = np.sum(normed > clip_limit)
        sat_pct = (below + above) / n * 100 if n > 0 else 0.0
        zero_pct = np.sum(normed == 0.0) / n * 100 if n > 0 else 0.0

        p01 = float(np.percentile(normed, 1))
        p50 = float(np.percentile(normed, 50))
        p99 = float(np.percentile(normed, 99))
        dyn_range = p99 - p01

        print(f"\n  [{name}]")
        print(f"    fit params: median={median:.6f}, iqr={iqr:.6f}")
        print(f"    preclip: low={preclip_low}, high={preclip_high}")
        print(f"    min={normed.min():.4f}, max={normed.max():.4f}, mean={normed.mean():.4f}, std={normed.std():.4f}")
        print(f"    p01={p01:.4f}, p50={p50:.4f}, p99={p99:.4f}, range(p99-p01)={dyn_range:.4f}")
        print(f"    saturation: below={below/n*100:.2f}%, above={above/n*100:.2f}%, total={sat_pct:.2f}%")
        print(f"    zero%={zero_pct:.2f}%")

        metrics[name] = {
            "sat_pct": sat_pct,
            "zero_pct": zero_pct,
            "dyn_range": dyn_range,
        }

    print("=" * 70 + "\n")
    return metrics


def _check_dynamic_scale_guard(metrics: dict, allow_bad_scale: bool = False):
    """
    Задача 324.5: Hard guard — останавливает обучение при заведомо плохом dynamic scale.
    Проверяет каждый канал на три условия:
      1. saturation > 10%
      2. zero% > 95% (канал схлопнулся в ноль)
      3. range(p99-p01) < 0.01 (канал фактически константный)
    """
    SAT_THRESHOLD = 10.0
    ZERO_THRESHOLD = 95.0
    RANGE_THRESHOLD = 0.01

    violations = []
    for name, m in metrics.items():
        if m["sat_pct"] > SAT_THRESHOLD:
            violations.append(
                f"  [{name}] saturation={m['sat_pct']:.2f}% > {SAT_THRESHOLD}% — слишком много значений за пределами clamp"
            )
        if m["zero_pct"] > ZERO_THRESHOLD:
            violations.append(
                f"  [{name}] zero%={m['zero_pct']:.2f}% > {ZERO_THRESHOLD}% — канал схлопнулся в ноль"
            )
        if m["dyn_range"] < RANGE_THRESHOLD:
            violations.append(
                f"  [{name}] range(p99-p01)={m['dyn_range']:.6f} < {RANGE_THRESHOLD} — канал фактически константный"
            )

    if violations:
        msg = (
            "\n" + "!" * 70 + "\n"
            "DYNAMIC SCALE GUARD: Обнаружены проблемы с нормализацией dynamic-каналов!\n"
            "Обучение остановлено. Исправьте pipeline или используйте --allow-bad-dynamic-scale.\n\n"
            "Нарушения:\n" + "\n".join(violations) + "\n"
            "!" * 70
        )
        if allow_bad_scale:
            print(msg)
            print("[WARN] --allow-bad-dynamic-scale активен, продолжаем несмотря на нарушения.\n")
        else:
            raise RuntimeError(msg)


def _run_normalized_nan_checks(train_ds, full_dataset):
    """Проверка данных на NaN после нормализации (sample-based)."""
    print("\nПроверка данных на NaN после нормализации (sampling)...")
    nan_check_samples = min(100, len(train_ds))
    nan_found = False

    for i in range(0, nan_check_samples, 10):
        try:
            ds = train_ds if i < len(train_ds) else full_dataset
            sample = ds[i]
            x, y, vol_target, weight = sample[:4]

            if i == 0:
                print(f"Sample Normalized Tensor (first 5 features of channel 0): {x[0, :5]}")

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


def _compute_class_weights(full_dataset, train_ds, args):
    """Вычисляет веса классов на основе тренировочного набора."""
    print("Calculating class weights from training set...")
    expected_label_cols = _resolve_dataset_label_columns(full_dataset)
    label_contract = _label_contract_from_args(args)
    label_indices = _resolve_train_label_indices(full_dataset, train_ds)

    train_labels = full_dataset.labels[label_indices]
    flattened_labels = train_labels.reshape(-1) if train_labels.ndim > 1 else train_labels
    valid_labels = flattened_labels[(flattened_labels >= 0) & (flattened_labels <= 2)]
    if valid_labels.size == 0:
        raise ValueError("No valid train labels found for class weights calculation.")

    if train_labels.ndim > 1 and train_labels.shape[1] != len(expected_label_cols):
        raise ValueError(
            "Class weights contract mismatch: labels tensor shape does not match dataset label columns."
        )

    classes, counts_list = np.unique(valid_labels, return_counts=True)

    counts = np.zeros(3, dtype=np.int64)
    for cls, count in zip(classes, counts_list):
        if 0 <= cls < 3:
            counts[int(cls)] = count

    total_samples = np.sum(counts)
    smoothing = args.class_weight_smooth
    n_classes = 3

    weights = total_samples / (n_classes * (counts + smoothing))
    weights = weights / np.mean(weights)

    flat_ratio = counts[0] / total_samples if total_samples > 0 else 1.0
    if flat_ratio > 0.85:
        amplification = 5.0
        print(f"[ADJUST] Flat class dominating: {flat_ratio:.1%}. Amplifying Up/Down weights by {amplification}x.")
        weights[1] *= amplification
        weights[2] *= amplification

    print(f"Effective class weights: [Flat: {weights[0]:.2f}, Up: {weights[1]:.2f}, Down: {weights[2]:.2f}]")
    metadata = {
        "label_cols": expected_label_cols,
        "label_mode": label_contract["label_mode"],
        "time_mode": label_contract["time_mode"],
        "dynamic_threshold": label_contract["dynamic_threshold"],
        "train_samples": int(valid_labels.size),
    }
    print(
        "Class weight contract: "
        f"label_cols={metadata['label_cols']}, "
        f"label_mode={metadata['label_mode']}, time_mode={metadata['time_mode']}"
    )
    return weights, metadata


def prepare_training_data(args, paths, winsor_limits, horizons, num_horizons, horizon_weights,
                          feature_df: pl.DataFrame | None = None) -> PreparedTrainingData:
    """
    Полный data pipeline: load -> feature engineer -> labeler -> dataset
    -> chronological split -> normalizer.fit(train only) -> dataloaders -> class weights.

    Порядок нельзя менять.
    """
    base_path = paths.base_path
    data_path = paths.data_path
    norm_params_path = paths.norm_params_path

    # 1. Парсим лаги past returns
    past_returns_lags = _parse_past_returns_lags(args.past_returns_lags)
    n_past_returns = len(past_returns_lags)
    in_channels = 11  # Задача 319: 11 каналов

    print(f'Using past returns lags: {past_returns_lags}')
    print(f'Total input channels: {in_channels} (MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread, DeltaImb, DeltaSpread)')
    print(f'Data loading mode: {args.data_mode}')

    # 2. Загрузка данных
    if feature_df is None:
        print(f"Loading data for {args.symbol} from {data_path}...")
        loader = LOBDataLoader(str(data_path), args.symbol)
        df = loader.load_data(lazy=False)

    # 3. Feature Engineering
        print("Engineering features...")
        fe = FeatureEngineer(n_levels=50)
        df = fe.transform(df)
    else:
        df = feature_df

    # 4. Разметка (Labeler)
    timestamps_len = df.height if "timestamp_ms" in df.columns else None
    label_cols = [c for c in df.columns if c == "label" or c.startswith("label_h")]
    y_len = df.height if label_cols else None
    _log_txy_lengths("labeling/before", timestamps_len=timestamps_len, x_len=df.height, y_len=y_len)
    print("Adding labels...")
    df = build_labeled_frame(
        df,
        horizon=horizons,
        threshold=args.threshold,
        dynamic_threshold=args.dynamic_threshold,
        args=args,
    )
    effective_threshold_summary = {}
    if "effective_threshold" in df.columns and df.height > 0:
        stats_row = df.select(
            [
                pl.col("effective_threshold").mean().alias("mean"),
                pl.col("effective_threshold").median().alias("p50"),
                pl.col("effective_threshold").quantile(0.95).alias("p95"),
                pl.col("effective_threshold").min().alias("min"),
                pl.col("effective_threshold").max().alias("max"),
            ]
        ).row(0)
        effective_threshold_summary = {
            "mean": float(stats_row[0]),
            "p50": float(stats_row[1]),
            "p95": float(stats_row[2]),
            "min": float(stats_row[3]),
            "max": float(stats_row[4]),
        }
        print(
            "[LABEL] effective_threshold summary: "
            f"p50={effective_threshold_summary['p50']:.6f}, "
            f"p95={effective_threshold_summary['p95']:.6f}, "
            f"min={effective_threshold_summary['min']:.6f}, "
            f"max={effective_threshold_summary['max']:.6f}"
        )
    if num_horizons == 1:
        label_h_cols = [c for c in df.columns if c.startswith("label_h")]
        if "label" not in df.columns and label_h_cols:
            if len(label_h_cols) == 1:
                print("[WARN] Single-horizon run produced label_h* column; renaming to label.")
                df = df.with_columns(pl.col(label_h_cols[0]).alias("label")).drop(label_h_cols[0])
            else:
                raise ValueError(
                    "Single-horizon run got multiple label_h* columns. "
                    "Expected exactly one label or label_h* column."
                )
    label_cols = [c for c in df.columns if c == "label" or c.startswith("label_h")]
    y_len = df.height if label_cols else None
    timestamps_len = df.height if "timestamp_ms" in df.columns else None
    _log_txy_lengths("labeling/after", timestamps_len=timestamps_len, x_len=df.height, y_len=y_len)

    # 5. Инициализация Normalizer (fit будет позже на train set)
    print("Initializing normalizer...")
    normalizer = Normalizer(norm_params_path, scale_multiplier=args.scale_multiplier)

    # 5.5. RegimeDetector — временно отключён (Задача 155 приостановлена)
    regime_detector = None
    regime_weights = None
    num_regimes = 0

    # 6. Параметры временного взвешивания
    if args.use_time_weighting:
        # Placeholder: class_weights будут вычислены позже, передаём None пока
        time_weighting_params = {
            'half_life_hours': args.half_life_hours,
            'min_weight': args.min_sample_weight,
            'class_weights': None
        }
        print(f"Time weighting enabled: half_life={args.half_life_hours}h, min_weight={args.min_sample_weight}")
    else:
        time_weighting_params = {
            'half_life_hours': 24.0,
            'min_weight': 1.0,
            'class_weights': None
        }

    # 7. Создание полного датасета
    timestamps_len = df.height if "timestamp_ms" in df.columns else None
    label_cols = [c for c in df.columns if c == "label" or c.startswith("label_h")]
    y_len = df.height if label_cols else None
    _log_txy_lengths("sequence/before", timestamps_len=timestamps_len, x_len=df.height, y_len=y_len)
    print(f"Creating dataset in 'memory' mode (raw features)...")
    full_dataset = build_full_dataset(
        df, args, past_returns_lags, winsor_limits, normalizer,
        regime_detector, time_weighting_params
    )
    aligned_timestamps = _aligned_timestamps_for_dataset(full_dataset)
    x_len, y_len = _resolve_sequence_lengths(full_dataset)
    max_lag = _resolve_max_lag_from_dataset(full_dataset)
    seq_len = int(getattr(full_dataset, "seq_len", 1))
    offset = max(0, seq_len - 1) + max_lag
    _log_txy_lengths(
        "sequence/after",
        timestamps_len=aligned_timestamps.size,
        x_len=x_len,
        y_len=y_len,
        extra=f"seq_len={seq_len}, max_lag={max_lag}, offset={offset}",
    )

    # 7.1. Проверка NaN в сырых данных
    if np.isnan(full_dataset.x_raw).any():
        raise ValueError("КРИТИЧНО: Входящие features содержат NaN строки для запуска обучения!")

    # 8. Разделение датасета согласно стратегии
    aligned_timestamps = _aligned_timestamps_for_dataset(full_dataset)
    x_len, y_len = _resolve_sequence_lengths(full_dataset)
    max_lag = _resolve_max_lag_from_dataset(full_dataset)
    seq_len = int(getattr(full_dataset, "seq_len", 1))
    offset = max(0, seq_len - 1) + max_lag
    _log_txy_lengths(
        "split/before",
        timestamps_len=aligned_timestamps.size,
        x_len=x_len,
        y_len=y_len,
        extra=f"seq_len={seq_len}, max_lag={max_lag}, offset={offset}",
    )
    train_ds, val_ds, test_ds, train_indices, val_indices, test_indices, split_artifacts = \
        split_dataset_by_strategy(full_dataset, args, horizons)
    aligned_timestamps = _aligned_timestamps_for_dataset(full_dataset)
    x_len, y_len = _resolve_sequence_lengths(full_dataset)
    _log_txy_lengths(
        "split/after",
        timestamps_len=aligned_timestamps.size,
        x_len=x_len,
        y_len=y_len,
        extra=f"seq_len={seq_len}, max_lag={max_lag}, offset={offset}",
    )

    total_len = len(full_dataset)
    print(f"\nSplit verification ({split_artifacts['strategy']}):")
    print(f"  Train: indices {train_indices[0]}-{train_indices[-1]} ({len(train_ds)} samples, {len(train_ds)/total_len*100:.1f}%)")
    print(f"  Val:   indices {val_indices[0]}-{val_indices[-1]} ({len(val_ds)} samples, {len(val_ds)/total_len*100:.1f}%)")
    print(f"  Test:  indices {test_indices[0]}-{test_indices[-1]} ({len(test_ds)} samples, {len(test_ds)/total_len*100:.1f}%)")
    print(
        f"  effective_purge_events={split_artifacts['effective_purge_events']}, "
        f"embargo_events={split_artifacts['embargo_events']}"
    )

    # 9. Fit нормализатора только на train-части
    _fit_normalizer_on_train(
        full_dataset, train_ds, normalizer, args, winsor_limits
    )

    # 10. NaN диагностика после нормализации
    _run_normalized_nan_checks(train_ds, full_dataset)

    print(f"Dataset split ({split_artifacts['strategy']}): Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    if args.use_symmetric_flip or args.volume_jitter_range > 0:
        print(f"Augmentation enabled for training: flip={args.use_symmetric_flip}, jitter={args.volume_jitter_range}, prob={args.augment_prob}")

    # 11. DataLoaders
    train_loader = DataLoader(train_ds, **build_dataloader_kwargs(args, shuffle=True))
    val_loader = DataLoader(val_ds, **build_dataloader_kwargs(args, shuffle=False))
    test_loader = DataLoader(test_ds, **build_dataloader_kwargs(args, shuffle=False))

    # 12. Веса классов
    class_weights, class_weight_metadata = _compute_class_weights(full_dataset, train_ds, args)

    label_columns = list(getattr(full_dataset, "label_cols", []))
    if class_weight_metadata.get("label_cols") != label_columns:
        raise ValueError(
            "Class weights were computed for different label columns than dataset uses: "
            f"weights={class_weight_metadata.get('label_cols')} vs dataset={label_columns}"
        )

    return PreparedTrainingData(
        df=df,
        normalizer=normalizer,
        full_dataset=full_dataset,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        past_returns_lags=past_returns_lags,
        in_channels=in_channels,
        num_horizons=num_horizons,
        horizon_weights=horizon_weights,
        regime_detector=regime_detector,
        regime_weights=regime_weights,
        num_regimes=num_regimes,
        n_past_returns=n_past_returns,
        time_weighting_params=time_weighting_params,
        label_columns=label_columns,
        class_weight_metadata=class_weight_metadata,
        split_artifacts=split_artifacts,
        effective_threshold_summary=effective_threshold_summary,
    )
