import argparse
from pathlib import Path

import polars as pl

from python_lab.src.features import FeatureEngineer
from python_lab.src.labels import Labeler
from python_lab.src.dataset import LOBDataLoader


def _counts(df: pl.DataFrame, label_col: str = "label") -> dict[int, int]:
    valid = df.filter(pl.col(label_col) != -100)
    counts = valid[label_col].value_counts().sort(label_col)
    values = counts.to_dict(as_series=False)
    return {int(k): int(v) for k, v in zip(values[label_col], values["count"])}


def _num_different(left: pl.DataFrame, right: pl.DataFrame, label_col: str = "label") -> int:
    common = min(left.height, right.height)
    if common == 0:
        return 0
    left_values = left[label_col].head(common).to_numpy()
    right_values = right[label_col].head(common).to_numpy()
    return int((left_values != right_values).sum())


def _num_different_three(row_df: pl.DataFrame, event_df: pl.DataFrame, exec_df: pl.DataFrame) -> int:
    common = min(row_df.height, event_df.height, exec_df.height)
    if common == 0:
        return 0
    row_values = row_df["label"].head(common).to_numpy()
    event_values = event_df["label"].head(common).to_numpy()
    exec_values = exec_df["label_exec"].head(common).to_numpy()
    return int(((row_values != event_values) | (row_values != exec_values) | (event_values != exec_values)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare row/event/execution label contracts")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--data_path", type=str, default="bots")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.0005)
    parser.add_argument("--event_time_column", type=str, default="feat_update_id")
    parser.add_argument("--cost_floor_bps", type=float, default=0.0)
    parser.add_argument("--fee_bps", type=float, default=0.0)
    parser.add_argument("--slippage_bps", type=float, default=0.0)
    parser.add_argument("--use_spread_floor", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    data_root = Path(args.data_path) / args.symbol / "data" / "raw"
    loader = LOBDataLoader(str(data_root), args.symbol)
    raw_df = loader.load_data(lazy=False)
    feature_df = FeatureEngineer(n_levels=50).transform(raw_df)

    row_df = Labeler(
        horizon=args.horizon,
        threshold=args.threshold,
        label_mode="legacy_mid_return",
        time_mode="row",
        event_time_column=args.event_time_column,
    ).add_labels(feature_df)
    event_df = Labeler(
        horizon=args.horizon,
        threshold=args.threshold,
        label_mode="legacy_mid_return",
        time_mode="event",
        event_time_column=args.event_time_column,
    ).add_labels(feature_df)
    exec_df = Labeler(
        horizon=args.horizon,
        threshold=args.threshold,
        label_mode="execution_mid_return",
        time_mode="event",
        event_time_column=args.event_time_column,
        cost_floor_bps=args.cost_floor_bps,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        use_spread_floor=args.use_spread_floor,
    ).add_labels(feature_df)

    print(f"counts(row): {_counts(row_df)}")
    print(f"counts(event): {_counts(event_df)}")
    print(f"counts(exec): {_counts(exec_df, label_col='label_exec')}")
    print(f"num_different: {_num_different_three(row_df, event_df, exec_df)}")


if __name__ == "__main__":
    main()
