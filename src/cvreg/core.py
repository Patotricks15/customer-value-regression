"""Future-spend regression on Online Retail II with strictly time-ordered snapshots."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

RANDOM_STATE = 42
HORIZON_DAYS = 180
TRAIN_SNAPSHOTS = [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-02-01"), pd.Timestamp("2010-03-01")]
TEST_SNAPSHOT = pd.Timestamp("2010-09-01")
DEFAULT_DATA_PATH = Path.home() / ".local/share/clv-data/online-retail-ii/online_retail_II.csv"


def data_path() -> Path:
    return Path(os.environ.get("RETAIL_DATA_PATH", DEFAULT_DATA_PATH))


def load_retail(path: Path | None = None) -> pd.DataFrame:
    raw = pd.read_csv(
        path or data_path(),
        encoding="latin1",
        parse_dates=["InvoiceDate"],
        dtype={"Invoice": str, "StockCode": str},
    )
    return clean_transactions(raw)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"Customer ID": "customer_id"})
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype("int64")
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    df["revenue"] = df["Quantity"] * df["Price"]
    return df[["Invoice", "customer_id", "InvoiceDate", "revenue"]]


def history_features(tx: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Features for customers with at least one purchase before the snapshot."""
    past = tx[tx["InvoiceDate"] < snapshot]
    grouped = past.groupby("customer_id")
    feats = pd.DataFrame(
        {
            "recency_days": (snapshot - grouped["InvoiceDate"].max()).dt.days,
            "frequency": grouped["Invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
            "tenure_days": (snapshot - grouped["InvoiceDate"].min()).dt.days,
        }
    )
    recent = past[past["InvoiceDate"] >= snapshot - pd.Timedelta(days=90)]
    feats["monetary_90d"] = recent.groupby("customer_id")["revenue"].sum().reindex(feats.index).fillna(0.0)
    feats["avg_order_value"] = feats["monetary"] / feats["frequency"]
    return feats


def future_spend(tx: pd.DataFrame, snapshot: pd.Timestamp, horizon: int = HORIZON_DAYS) -> pd.Series:
    """Spend in the window (snapshot, snapshot + horizon]. Purchases on the snapshot itself are excluded."""
    end = snapshot + pd.Timedelta(days=horizon)
    window = tx[(tx["InvoiceDate"] > snapshot) & (tx["InvoiceDate"] <= end)]
    return window.groupby("customer_id")["revenue"].sum()


def build_panel(tx: pd.DataFrame, snapshots: list[pd.Timestamp], horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """Stack one row per (snapshot, customer) with features known at the snapshot and the future target."""
    frames = []
    for snap in snapshots:
        feats = history_features(tx, snap)
        feats["target"] = future_spend(tx, snap, horizon).reindex(feats.index).fillna(0.0)
        feats["snapshot"] = snap
        frames.append(feats.reset_index())
    return pd.concat(frames, ignore_index=True)


def assert_time_ordered(train_snapshots: list[pd.Timestamp], test_snapshot: pd.Timestamp, horizon: int = HORIZON_DAYS) -> None:
    """Training targets must end before the test snapshot, so no test period leaks into training."""
    last_train_end = max(train_snapshots) + pd.Timedelta(days=horizon)
    if last_train_end > test_snapshot:
        raise ValueError(f"training target window ends {last_train_end.date()}, after test snapshot {test_snapshot.date()}")


FEATURE_COLS = ["recency_days", "frequency", "monetary", "tenure_days", "monetary_90d", "avg_order_value"]


def model_matrix(panel: pd.DataFrame) -> np.ndarray:
    return np.log1p(panel[FEATURE_COLS].clip(lower=0).to_numpy(dtype=float))


def fit_xgb(X: np.ndarray, y: np.ndarray) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def frequency_bins(train_freq: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Quintile edges learned on training data only."""
    return np.unique(np.quantile(train_freq, np.linspace(0, 1, n_bins + 1)[1:-1]))


def segment_mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict the mean training target of each frequency quintile."""
    edges = frequency_bins(train["frequency"].to_numpy())
    train_bin = np.digitize(train["frequency"], edges)
    test_bin = np.digitize(test["frequency"], edges)
    means = pd.Series(train["target"].to_numpy()).groupby(train_bin).mean()
    return pd.Series(test_bin).map(means).fillna(train["target"].mean()).to_numpy()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }
