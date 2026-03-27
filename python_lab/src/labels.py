from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Union

import numpy as np
import polars as pl


LabelMode = Literal["legacy_mid_return", "execution_mid_return"]
TimeMode = Literal["row", "event", "ms"]


@dataclass(frozen=True)
class LabelDiagnostics:
    label_mode: str
    time_mode: str
    effective_threshold_p50: float
    effective_threshold_p95: float
    row_gap_median_seconds: float | None
    event_gap_median_seconds: float | None


class Labeler:
    """
    Формирует ternary labels для single/multi-horizon сценариев.

    Backward compatibility:
    - `legacy_mid_return` + `time_mode="row"` сохраняет прежнюю семантику shift(-h)
    - имена выходных колонок остаются `label` / `label_h*` и `future_return_h*`
    """

    def __init__(
        self,
        horizon: Union[int, List[int]] = 100,
        threshold: float = 0.0005,
        dynamic_threshold: bool = False,
        window: int = 1000,
        K: float = 0.5,
        label_mode: LabelMode = "legacy_mid_return",
        time_mode: TimeMode = "row",
        event_time_column: str = "feat_update_id",
        cost_floor_bps: float = 0.0,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        use_spread_floor: bool = False,
    ):
        if isinstance(horizon, int):
            self.horizons = [horizon]
            self.single_horizon = True
        else:
            self.horizons = sorted(int(h) for h in horizon)
            self.single_horizon = False

        if not self.horizons:
            raise ValueError("Labeler requires at least one horizon")

        allowed_label_modes = {"legacy_mid_return", "execution_mid_return"}
        if label_mode not in allowed_label_modes:
            raise ValueError(f"Unsupported label_mode={label_mode!r}. Expected one of {sorted(allowed_label_modes)}")

        allowed_time_modes = {"row", "event", "ms"}
        if time_mode not in allowed_time_modes:
            raise ValueError(f"Unsupported time_mode={time_mode!r}. Expected one of {sorted(allowed_time_modes)}")

        if dynamic_threshold and label_mode == "execution_mid_return":
            raise ValueError(
                "dynamic_threshold is allowed only in legacy/debug mode; "
                "execution_mid_return must use effective_threshold contract."
            )

        self.threshold = float(threshold)
        self.dynamic_threshold = bool(dynamic_threshold)
        self.window = int(window)
        self.K = float(K)
        self.max_horizon = max(self.horizons)
        self.label_mode = label_mode
        self.time_mode = time_mode
        self.event_time_column = event_time_column
        self.cost_floor_bps = float(cost_floor_bps)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.use_spread_floor = bool(use_spread_floor)

    def add_labels(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
        is_lazy = isinstance(df, pl.LazyFrame)
        frame = df.collect() if is_lazy else df
        if frame.height == 0:
            return frame.lazy() if is_lazy else frame

        self._validate_required_columns(frame)
        working = frame.with_row_index("_row_idx")

        threshold_by_horizon: dict[int, np.ndarray] = {}
        future_index_by_horizon: dict[int, np.ndarray] = {}
        valid_mask_by_horizon: dict[int, np.ndarray] = {}

        for horizon in self.horizons:
            future_idx = self._map_future_indices(working, horizon, self.time_mode)
            valid_mask = future_idx >= 0
            future_index_by_horizon[horizon] = future_idx
            valid_mask_by_horizon[horizon] = valid_mask
            threshold_by_horizon[horizon] = self._build_threshold_array(working)

        keep_mask = valid_mask_by_horizon[self.max_horizon]
        labeled = working.filter(pl.Series("_keep_mask", keep_mask))
        kept_row_idx = labeled["_row_idx"].to_numpy()

        base_mid = working["mid_price"].to_numpy()
        columns_to_add: list[pl.Series] = [
            pl.Series("effective_threshold", threshold_by_horizon[self.max_horizon][kept_row_idx])
        ]
        label_columns: list[str] = []

        for horizon in self.horizons:
            future_idx = future_index_by_horizon[horizon][kept_row_idx]
            valid_mask = future_idx >= 0

            future_mid = np.full(len(kept_row_idx), np.nan, dtype=np.float64)
            future_mid[valid_mask] = base_mid[future_idx[valid_mask]]

            future_return = np.full(len(kept_row_idx), np.nan, dtype=np.float64)
            future_return[valid_mask] = (future_mid[valid_mask] - base_mid[kept_row_idx][valid_mask]) / base_mid[kept_row_idx][valid_mask]

            threshold_arr = threshold_by_horizon[horizon][kept_row_idx]
            labels = np.full(len(kept_row_idx), -100, dtype=np.int8)
            pos_mask = valid_mask & (future_return > threshold_arr)
            neg_mask = valid_mask & (future_return < -threshold_arr)
            flat_mask = valid_mask & ~(pos_mask | neg_mask)
            labels[pos_mask] = 1
            labels[neg_mask] = 2
            labels[flat_mask] = 0

            future_mid_col = f"mid_future_h{horizon}"
            future_return_col = f"future_return_h{horizon}"
            label_col = "label" if self.single_horizon else f"label_h{horizon}"

            columns_to_add.extend(
                [
                    pl.Series(future_mid_col, future_mid),
                    pl.Series(future_return_col, future_return),
                    pl.Series(label_col, labels),
                ]
            )
            if self.single_horizon:
                columns_to_add.extend(
                    [
                        pl.Series("mid_future", future_mid),
                        pl.Series("future_return", future_return),
                    ]
                )
                if self.label_mode == "execution_mid_return":
                    columns_to_add.append(pl.Series("label_exec", labels))
                elif self.time_mode == "event":
                    columns_to_add.append(pl.Series("label_event", labels))
                elif self.time_mode == "row":
                    columns_to_add.append(pl.Series("label_row", labels))
            label_columns.append(label_col)

        diagnostics = self._build_diagnostics(working, threshold_by_horizon[self.max_horizon][kept_row_idx])
        labeled = labeled.with_columns(
            *columns_to_add,
            pl.lit(self.label_mode).alias("label_contract_mode"),
            pl.lit(self.time_mode).alias("label_contract_time_mode"),
            pl.lit(self.event_time_column).alias("label_contract_event_time_column"),
        ).drop("_keep_mask", "_row_idx")

        self._print_diagnostics(labeled, label_columns, diagnostics)
        return labeled.lazy() if is_lazy else labeled

    def _validate_required_columns(self, df: pl.DataFrame) -> None:
        required = {"mid_price"}
        if self.label_mode == "execution_mid_return" and self.use_spread_floor:
            required.update({"feat_ask_p_0", "feat_bid_p_0"})
        if self.time_mode == "event":
            required.add(self.event_time_column)
        if self.time_mode == "ms":
            required.add("timestamp_ms")

        missing = [col for col in sorted(required) if col not in df.columns]
        if missing:
            raise ValueError(f"Labeler missing required columns: {missing}")

    def _map_future_indices(self, df: pl.DataFrame, horizon_value: int, time_mode: TimeMode) -> np.ndarray:
        n_rows = df.height
        if n_rows == 0:
            return np.empty(0, dtype=np.int64)

        if time_mode == "row":
            future_idx = np.arange(n_rows, dtype=np.int64) + int(horizon_value)
            future_idx[future_idx >= n_rows] = -1
            return future_idx

        if time_mode == "event":
            event_values = df[self.event_time_column].to_numpy()
            event_mask = np.ones(n_rows, dtype=bool)
            event_mask[1:] = event_values[1:] != event_values[:-1]
            event_indices = np.flatnonzero(event_mask)

            current_event_rank = np.cumsum(event_mask.astype(np.int64)) - 1
            target_event_rank = current_event_rank + int(horizon_value)
            future_idx = np.full(n_rows, -1, dtype=np.int64)
            valid = target_event_rank < len(event_indices)
            future_idx[valid] = event_indices[target_event_rank[valid]]
            return future_idx

        timestamps = df["timestamp_ms"].to_numpy()
        target_timestamps = timestamps + int(horizon_value)
        future_idx = np.searchsorted(timestamps, target_timestamps, side="left").astype(np.int64)
        future_idx[future_idx >= n_rows] = -1
        return future_idx

    def _build_threshold_array(self, df: pl.DataFrame) -> np.ndarray:
        if self.dynamic_threshold:
            returns_std = (
                df.select(
                    pl.col("mid_price")
                    .pct_change()
                    .rolling_std(window_size=self.window)
                    .fill_null(strategy="backward")
                    .fill_null(0.0)
                    .alias("returns_std")
                )
                .get_column("returns_std")
                .to_numpy()
            )
            threshold_arr = np.clip(returns_std * self.K, 0.0001, None)
            return threshold_arr.astype(np.float64)

        threshold_arr = np.full(df.height, self.threshold, dtype=np.float64)
        if self.label_mode != "execution_mid_return":
            return threshold_arr

        spread_floor = np.zeros(df.height, dtype=np.float64)
        if self.use_spread_floor:
            spread_bps = (
                (df["feat_ask_p_0"].to_numpy() - df["feat_bid_p_0"].to_numpy())
                / np.maximum(df["mid_price"].to_numpy(), 1e-12)
            ) * 10000.0
            spread_floor = spread_bps / 10000.0

        threshold_cost = (self.cost_floor_bps + 2.0 * self.fee_bps + self.slippage_bps) / 10000.0
        threshold_arr = np.maximum(threshold_arr, threshold_cost)
        threshold_arr = np.maximum(threshold_arr, spread_floor)
        return threshold_arr

    def _build_diagnostics(self, df: pl.DataFrame, effective_threshold: np.ndarray) -> LabelDiagnostics:
        row_gap = None
        event_gap = None
        if "timestamp_ms" in df.columns and df.height > 1:
            timestamps = df["timestamp_ms"].to_numpy()
            row_deltas = np.diff(timestamps)
            if len(row_deltas) > 0:
                row_gap = float(np.median(row_deltas) / 1000.0)

            if self.event_time_column in df.columns:
                event_values = df[self.event_time_column].to_numpy()
                event_mask = np.ones(df.height, dtype=bool)
                event_mask[1:] = event_values[1:] != event_values[:-1]
                event_timestamps = timestamps[event_mask]
                if len(event_timestamps) > 1:
                    event_gap = float(np.median(np.diff(event_timestamps)) / 1000.0)

        return LabelDiagnostics(
            label_mode=self.label_mode,
            time_mode=self.time_mode,
            effective_threshold_p50=float(np.nanpercentile(effective_threshold, 50)),
            effective_threshold_p95=float(np.nanpercentile(effective_threshold, 95)),
            row_gap_median_seconds=row_gap,
            event_gap_median_seconds=event_gap,
        )

    def _print_diagnostics(self, df: pl.DataFrame, label_columns: list[str], diagnostics: LabelDiagnostics) -> None:
        print(f"\n[{self.__class__.__name__}] Labels distribution:")
        if self.dynamic_threshold:
            print(f"  dynamic_threshold: rolling_std * {self.K} (window={self.window})")
        else:
            print(f"  threshold_static: {self.threshold:.6f} ({self.threshold * 100:.2f}%)")
        print(f"  label_mode: {diagnostics.label_mode}")
        print(f"  time_mode: {diagnostics.time_mode}")
        print(f"  effective_threshold_p50: {diagnostics.effective_threshold_p50:.6f}")
        print(f"  effective_threshold_p95: {diagnostics.effective_threshold_p95:.6f}")
        print(f"  row_gap_median_seconds: {diagnostics.row_gap_median_seconds}")
        print(f"  event_gap_median_seconds: {diagnostics.event_gap_median_seconds}")
        print(f"  horizons: {self.horizons}")

        for label_col in label_columns:
            counts = df.filter(pl.col(label_col) != -100)[label_col].value_counts().sort(label_col)
            masked_count = df.filter(pl.col(label_col) == -100).height
            horizon_str = label_col.replace("label_h", "h") if label_col.startswith("label_h") else "single"
            print(f"  {horizon_str}: {counts.to_dict(as_series=False)} | Masked: {masked_count}")


if __name__ == "__main__":
    data = {
        "timestamp_ms": [0, 10, 20, 30, 45, 60, 80, 100, 130, 160, 190],
        "feat_update_id": [1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7],
        "mid_price": [100.0, 100.05, 100.1, 100.0, 99.9, 99.8, 100.0, 100.2, 100.3, 100.4, 100.5],
        "feat_bid_p_0": [99.99, 100.04, 100.09, 99.99, 99.89, 99.79, 99.99, 100.19, 100.29, 100.39, 100.49],
        "feat_ask_p_0": [100.01, 100.06, 100.11, 100.01, 99.91, 99.81, 100.01, 100.21, 100.31, 100.41, 100.51],
    }
    test_df = pl.DataFrame(data)

    print("=== Test 1: Legacy row mode ===")
    labeler_single = Labeler(horizon=2, threshold=0.0005)
    res_single = labeler_single.add_labels(test_df)
    print(res_single.select(["mid_price", "label", "future_return_h2"]))

    print("\n=== Test 2: Execution event mode ===")
    labeler_exec = Labeler(
        horizon=2,
        threshold=0.0005,
        label_mode="execution_mid_return",
        time_mode="event",
        cost_floor_bps=1.0,
        fee_bps=0.5,
        slippage_bps=0.5,
        use_spread_floor=True,
    )
    res_exec = labeler_exec.add_labels(test_df)
    print(res_exec.select(["mid_price", "label", "future_return_h2"]))
