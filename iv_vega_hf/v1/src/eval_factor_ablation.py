"""Factor-group ablation on mainpool real dataset."""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/mainpool_1m.parquet")
    p.add_argument("--horizon", type=int, default=5)
    return p.parse_args()


def wf(df, ycol, feats, min_train=240):
    d = df.dropna(subset=[ycol]).copy()
    X = d[feats].fillna(d[feats].median())
    y = d[ycol].to_numpy(float)
    if len(d) <= min_train + 20:
        return None
    pr, tr = [], []
    for t in range(min_train, len(d)):
        xtr = X.iloc[:t].to_numpy(float)
        ytr = y[:t]
        xte = X.iloc[t:t+1].to_numpy(float)
        mu = np.nanmean(xtr, axis=0); sd = np.nanstd(xtr, axis=0); sd[sd < 1e-12] = 1
        xtr = np.clip((np.nan_to_num(xtr)-mu)/sd, -8, 8)
        xte = np.clip((np.nan_to_num(xte)-mu)/sd, -8, 8)
        xa = np.c_[np.ones(len(xtr)), xtr]
        xb = np.c_[np.ones(len(xte)), xte]
        xtx = np.nan_to_num(xa.T @ xa); xty = np.nan_to_num(xa.T @ ytr)
        reg = np.eye(xtx.shape[0]) * 5.0; reg[0,0]=0
        beta = np.linalg.solve(xtx + reg, xty)
        pr.append(float((xb @ beta)[0])); tr.append(float(y[t]))
    pr = np.array(pr); tr = np.array(tr)
    mae = float(np.mean(np.abs(tr-pr))); rmse = float(np.sqrt(np.mean((tr-pr)**2))); hit=float(np.mean(np.sign(tr)==np.sign(pr)))
    return dict(mae=mae, rmse=rmse, hit=hit, folds=len(tr))


def main():
    args = parse_args()
    df = pd.read_parquet(args.input).sort_values("dt_exch")
    ycol = f"y_{args.horizon}m"
    df[ycol] = df["iv_pool"].shift(-args.horizon) - df["iv_pool"]

    groups = {
        "all": ["vega_signed_1m","vega_abs_1m","spread_pool_1m","f_ret_1m"],
        "drop_vega": ["spread_pool_1m","f_ret_1m"],
        "drop_micro": ["vega_signed_1m","vega_abs_1m","f_ret_1m"],
        "drop_fut": ["vega_signed_1m","vega_abs_1m","spread_pool_1m"],
        "vega_only": ["vega_signed_1m","vega_abs_1m"],
    }
    for name, feats in groups.items():
        r = wf(df, ycol, feats)
        print(name, r)


if __name__ == "__main__":
    main()
