"""Train baseline models for Level-A task.

Usage:
  python src/train_baseline.py \
    --features data/features_1m.parquet \
    --labels data/labels_1m.parquet \
    --target y_1m
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline model")
    p.add_argument("--features", required=True, help="Features parquet/csv")
    p.add_argument("--labels", required=True, help="Labels parquet/csv")
    p.add_argument("--target", default="y_1m", help="Target label name")
    return p.parse_args()


def _read(path: Path):
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()

    import numpy as np

    X = _read(Path(args.features))
    y = _read(Path(args.labels))

    if args.target not in y.columns:
        raise ValueError(f"target not found: {args.target}")

    data = X.join(y[[args.target]], how="inner")
    data = data.dropna(subset=[args.target])
    # keep more rows in early prototype: median-impute features
    feature_cols = [c for c in data.columns if c != args.target]
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
    if len(data) < 4:
        raise ValueError("not enough rows after target filtering")

    # simple chronological split (proxy for walk-forward v1)
    split = max(1, int(len(data) * 0.8))
    if split >= len(data):
        split = len(data) - 1
    train = data.iloc[:split]
    test = data.iloc[split:]

    X_train = train.drop(columns=[args.target])
    y_train = train[args.target]
    X_test = test.drop(columns=[args.target])
    y_test = test[args.target]

    def mae(a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    def rmse(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(np.sqrt(np.mean(d * d)))

    # Baseline-0: naive y_hat=0
    yhat_naive = np.zeros(len(y_test))
    mae_naive = mae(y_test, yhat_naive)
    rmse_naive = rmse(y_test, yhat_naive)

    # Baseline-1: linear regression via pseudo-inverse (robust fallback)
    Xtr = np.nan_to_num(X_train.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ytr = np.nan_to_num(y_train.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(X_test.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    # add intercept
    Xtr_aug = np.c_[np.ones(len(Xtr)), Xtr]
    Xte_aug = np.c_[np.ones(len(Xte)), Xte]
    beta = np.linalg.pinv(Xtr_aug) @ ytr
    yhat_lin = Xte_aug @ beta

    mae_ridge = mae(y_test, yhat_lin)
    rmse_ridge = rmse(y_test, yhat_lin)

    print("[BASELINE RESULT]")
    print(f"target={args.target} rows_train={len(train)} rows_test={len(test)}")
    print(f"naive_zero  MAE={mae_naive:.6f} RMSE={rmse_naive:.6f}")
    print(f"linear_lstsq MAE={mae_ridge:.6f} RMSE={rmse_ridge:.6f}")


if __name__ == "__main__":
    main()
