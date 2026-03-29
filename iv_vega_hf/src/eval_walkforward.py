"""Walk-forward style prototype evaluation (expanding window).

Usage:
  python src/eval_walkforward.py \
    --features data/features_1m.parquet \
    --labels data/labels_1m.parquet \
    --target y_1m
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward prototype evaluation")
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--target", default="y_1m")
    p.add_argument("--min-train", type=int, default=5)
    p.add_argument("--clip-z", type=float, default=8.0, help="feature z-score clipping")
    p.add_argument("--ridge-alpha", type=float, default=1.0, help="ridge regularization alpha")
    return p.parse_args()


def _read(path: Path):
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    import numpy as np

    args = parse_args()
    X = _read(Path(args.features))
    y = _read(Path(args.labels))

    data = X.join(y[[args.target]], how="inner").dropna(subset=[args.target])
    feat_cols = [c for c in data.columns if c != args.target]
    data[feat_cols] = data[feat_cols].fillna(data[feat_cols].median())

    if len(data) <= args.min_train:
        raise ValueError("not enough rows for walk-forward")

    preds_naive, preds_ridge, trues = [], [], []

    for t in range(args.min_train, len(data)):
        tr = data.iloc[:t]
        te = data.iloc[t:t+1]

        Xtr = np.nan_to_num(tr[feat_cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        ytr = np.nan_to_num(tr[args.target].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(te[feat_cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        # per-fold standardization for numerical stability
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr_n = np.clip((Xtr - mu) / sd, -args.clip_z, args.clip_z)
        Xte_n = np.clip((Xte - mu) / sd, -args.clip_z, args.clip_z)

        Xtr_aug = np.c_[np.ones(len(Xtr_n)), Xtr_n]
        Xte_aug = np.c_[np.ones(len(Xte_n)), Xte_n]

        # ridge closed-form: beta = (X'X + alpha*I)^-1 X'y (no regularization on intercept)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            xtx = Xtr_aug.T @ Xtr_aug
            xty = Xtr_aug.T @ ytr
        xtx = np.nan_to_num(xtx, nan=0.0, posinf=0.0, neginf=0.0)
        xty = np.nan_to_num(xty, nan=0.0, posinf=0.0, neginf=0.0)

        reg = np.eye(xtx.shape[0]) * args.ridge_alpha
        reg[0, 0] = 0.0
        beta = np.linalg.solve(xtx + reg, xty)

        yhat_ridge = float((Xte_aug @ beta)[0])
        yhat_naive = 0.0
        ytrue = float(te[args.target].iloc[0])

        preds_ridge.append(yhat_ridge)
        preds_naive.append(yhat_naive)
        trues.append(ytrue)

    trues = np.array(trues)
    preds_ridge = np.array(preds_ridge)
    preds_naive = np.array(preds_naive)

    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

    print("[WALKFORWARD RESULT]")
    print(f"target={args.target} folds={len(trues)}")
    print(f"naive_zero  MAE={mae(trues, preds_naive):.6f} RMSE={rmse(trues, preds_naive):.6f}")
    print(f"ridge_cf    MAE={mae(trues, preds_ridge):.6f} RMSE={rmse(trues, preds_ridge):.6f}")


if __name__ == "__main__":
    main()
