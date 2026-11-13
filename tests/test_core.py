import numpy as np
import pandas as pd
import pytest

from cvreg.core import (
    assert_time_ordered,
    build_panel,
    clean_transactions,
    future_spend,
    history_features,
    regression_metrics,
    segment_mean_baseline,
)


def _tx(rows):
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return clean_transactions(df.assign(Description="x", Country="x"))


def test_target_window_excludes_snapshot_day_and_includes_horizon_end():
    tx = _tx(
        [
            ["1", "A", 1, "2010-06-01", 10.0, 1.0],  # on snapshot: excluded from target
            ["2", "A", 1, "2010-06-02", 5.0, 1.0],  # inside window
            ["3", "A", 1, "2010-12-28", 7.0, 1.0],  # exactly snapshot + 210 days: outside
            ["4", "A", 1, "2010-11-28", 3.0, 1.0],  # exactly snapshot + 180 days: inside
        ]
    )
    spend = future_spend(tx, pd.Timestamp("2010-06-01"), horizon=180)
    assert spend.loc[1] == pytest.approx(8.0)


def test_history_features_use_only_past_transactions():
    tx = _tx(
        [
            ["1", "A", 1, "2010-05-01", 10.0, 1.0],
            ["2", "A", 1, "2010-06-05", 99.0, 1.0],  # after snapshot
        ]
    )
    feats = history_features(tx, pd.Timestamp("2010-06-01"))
    assert feats.loc[1, "monetary"] == 10.0
    assert feats.loc[1, "frequency"] == 1


def test_panel_stacks_snapshots_with_targets():
    tx = _tx(
        [
            ["1", "A", 1, "2010-01-15", 10.0, 1.0],
            ["2", "A", 1, "2010-02-15", 20.0, 1.0],
        ]
    )
    panel = build_panel(tx, [pd.Timestamp("2010-02-01")], horizon=60)
    assert len(panel) == 1
    assert panel["target"].iloc[0] == 20.0
    assert panel["snapshot"].iloc[0] == pd.Timestamp("2010-02-01")


def test_time_ordering_guard_rejects_leakage():
    with pytest.raises(ValueError):
        assert_time_ordered([pd.Timestamp("2010-06-01")], pd.Timestamp("2010-09-01"), horizon=180)
    assert_time_ordered([pd.Timestamp("2010-01-01")], pd.Timestamp("2010-09-01"), horizon=180)


def test_segment_mean_baseline_predicts_bin_means():
    train = pd.DataFrame({"frequency": [1, 1, 5, 5], "target": [0.0, 2.0, 10.0, 20.0]})
    test = pd.DataFrame({"frequency": [1, 5]})
    pred = segment_mean_baseline(train, test)
    assert pred[0] == pytest.approx(1.0)
    assert pred[1] == pytest.approx(15.0)


def test_regression_metrics_known_values():
    m = regression_metrics(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
    assert m["mae"] == pytest.approx(1.5)
    assert m["rmse"] == pytest.approx(np.sqrt(2.5))
