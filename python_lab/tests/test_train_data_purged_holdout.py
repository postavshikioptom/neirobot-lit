from types import SimpleNamespace

import numpy as np

from python_lab.src.train_data import split_dataset_purged_holdout


class DummyDataset:
    def __init__(self, timestamps, *, seq_len=100, past_returns_lags=None):
        self._timestamps = np.asarray(timestamps, dtype=np.int64)
        self.seq_len = seq_len
        self.past_returns_lags = list(past_returns_lags or [10, 50, 100])
        self.is_train = False

    def __len__(self):
        return int(self._timestamps.size)

    def __getitem__(self, idx):
        return int(idx)

    def get_timestamps(self):
        return self._timestamps


def _make_args(**overrides):
    base = {
        "holdout_days": 1.0,
        "seq_len": 100,
        "purge_buffer_events": 100,
        "embargo_seconds": 0,
        "embargo_buffer_events": 50,
        "past_returns_lags": "10,50,100",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_purged_holdout_auto_adjusts_on_short_intraday_span():
    timestamps = np.arange(0, 1_000 * 100, 100, dtype=np.int64)
    dataset = DummyDataset(timestamps)

    train_ds, val_ds, test_ds, train_indices, val_indices, test_indices, artifacts = split_dataset_purged_holdout(
        dataset,
        _make_args(),
        horizons=[100],
    )

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    assert len(train_indices) > 0
    assert len(val_indices) > 0
    assert len(test_indices) > 0
    assert artifacts["holdout_auto_adjusted"] is True
    assert artifacts["effective_holdout_days"] < 1.0


def test_purged_holdout_keeps_requested_window_on_long_span():
    timestamps = np.arange(0, 100 * 60 * 60 * 1000, 60 * 60 * 1000, dtype=np.int64)
    dataset = DummyDataset(timestamps, seq_len=1, past_returns_lags=[1])

    _, _, _, _, _, _, artifacts = split_dataset_purged_holdout(
        dataset,
        _make_args(seq_len=1, purge_buffer_events=1, embargo_buffer_events=0, past_returns_lags="1", holdout_days=1.0),
        horizons=[1],
    )

    assert artifacts["holdout_auto_adjusted"] is False
    assert artifacts["effective_holdout_days"] == 1.0
