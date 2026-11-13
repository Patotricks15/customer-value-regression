# Customer Value Regression (CS02)

Predicts how much a customer will spend in the next 6 months, using gradient-boosted regression on real UK retailer transactions.

**Core question:** how much will a customer spend in the next 6 months?

## Setup

- Target: revenue in (snapshot, snapshot + 180 days]. Purchases on the snapshot day are excluded.
- Features (all from transactions before the snapshot): recency, frequency, monetary, tenure, 90-day monetary and average order value, with `log1p`.
- Time-ordered split, checked by code (`assert_time_ordered`):
  - Train snapshots: 2010-01-01, 2010-02-01, 2010-03-01. Their target windows end by 2010-08-28.
  - Test snapshot: 2010-09-01 (targets through 2011-02-28).
- Baseline: mean training target per frequency quintile (segment mean).
- Model: `XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)`, predictions clipped at 0.

## Results (test snapshot 2010-09-01)

| Model | MAE | RMSE |
|---|---|---|
| Segment-mean baseline | 2,099.2 | 5,076.0 |
| XGBoost | **1,479.3** | **4,389.9** |

XGBoost improves MAE by **29.5%** over the baseline. Customers with zero future spend make up 36% of the test set.

Feature importance (gain share): monetary 0.34, average order value 0.20, 90-day monetary 0.19, frequency 0.13, tenure 0.08, recency 0.06.

Figures: [predicted vs actual](reports/figures/cs02_pred_vs_actual.png), [feature importance](reports/figures/cs02_feature_importance.png).

## Limitations

- The test set has 3,321 customers from one snapshot. A few very large spenders dominate RMSE (the maximum actual value is about 150,000), so treat RMSE as noisy.
- One retailer and one period (2009 to 2011). The results do not transfer to other markets without validation.

## Reproduce

```bash
pip install -e ".[dev]"
RETAIL_DATA_PATH=/path/to/online_retail_II.csv pytest
cd notebooks && jupyter nbconvert --to notebook --execute --inplace customer_value_regression.ipynb
```

Code is in [src/cvreg/core.py](src/cvreg/core.py); tests are in [tests/test_core.py](tests/test_core.py). Data notes are in [data/README.md](data/README.md).
