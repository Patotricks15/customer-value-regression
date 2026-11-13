# %% [markdown]
# # CS02: 6-month future spend regression (Online Retail II)
#
# Target: revenue from each customer in (snapshot, snapshot + 180 days].
# Features use only transactions before the snapshot.
# Training snapshots end their target windows before the test snapshot starts.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvreg.core import (
    FEATURE_COLS,
    HORIZON_DAYS,
    RANDOM_STATE,
    TEST_SNAPSHOT,
    TRAIN_SNAPSHOTS,
    assert_time_ordered,
    build_panel,
    data_path,
    fit_xgb,
    load_retail,
    model_matrix,
    regression_metrics,
    segment_mean_baseline,
)

FIG = Path.cwd().parent / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
assert_time_ordered(TRAIN_SNAPSHOTS, TEST_SNAPSHOT, HORIZON_DAYS)

# %%
tx = load_retail()
print("data:", data_path())
print("train snapshots:", [d.date().isoformat() for d in TRAIN_SNAPSHOTS], "| test snapshot:", TEST_SNAPSHOT.date())

train = build_panel(tx, TRAIN_SNAPSHOTS)
test = build_panel(tx, [TEST_SNAPSHOT])
print("train rows:", len(train), "| test rows:", len(test))
print("share with zero future spend (train / test):",
      round((train.target == 0).mean(), 3), round((test.target == 0).mean(), 3))
test[["target"]].describe().round(2)

# %%
baseline_pred = segment_mean_baseline(train, test)
base = regression_metrics(test["target"].to_numpy(), baseline_pred)
print("segment-mean baseline (frequency quintile):", {k: round(v, 2) for k, v in base.items()})

# %%
model = fit_xgb(model_matrix(train), train["target"].to_numpy())
model_pred = np.clip(model.predict(model_matrix(test)), 0, None)
mdl = regression_metrics(test["target"].to_numpy(), model_pred)
print("XGBoost:", {k: round(v, 2) for k, v in mdl.items()})

results = pd.DataFrame({"baseline": base, "xgboost": mdl}).T.round(2)
print("improvement in MAE:", round(1 - mdl["mae"] / base["mae"], 4))
results

# %%
fig, ax = plt.subplots(figsize=(6, 6))
lim = float(np.percentile(test["target"], 99))
ax.scatter(test["target"], model_pred, s=6, alpha=0.4)
ax.plot([0, lim], [0, lim], color="red", linestyle="--")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("actual 6-month spend")
ax.set_ylabel("predicted")
ax.set_title(f"Test snapshot {TEST_SNAPSHOT.date()}: predicted vs actual (99th pct axis)")
fig.tight_layout()
fig.savefig(FIG / "cs02_pred_vs_actual.png", dpi=120)
plt.show()

# %%
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(6, 4))
importance.plot.barh(ax=ax, color="steelblue")
ax.set_title("XGBoost feature importance (gain share)")
fig.tight_layout()
fig.savefig(FIG / "cs02_feature_importance.png", dpi=120)
plt.show()
importance.sort_values(ascending=False).round(3)
