"""Evaluate real-data IV forecasting on mainpool/top3 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mainpool", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/mainpool_1m.parquet")
    p.add_argument("--top3", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/top3_contract_1m.parquet")
    p.add_argument("--horizons", default="1,5,15")
    p.add_argument("--iv-rsi-window", type=int, default=14)
    return p.parse_args()


def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    up_ema = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    dn_ema = dn.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = up_ema / dn_ema.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def walkforward_eval(df: pd.DataFrame, ycol: str, feature_cols: list[str], min_train: int = 120):
    df = df.dropna(subset=[ycol]).copy()
    if len(df) <= min_train + 10:
        return None
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df[ycol].values

    preds_n, preds_r, trues = [], [], []
    for t in range(min_train, len(df)):
        xtr = X.iloc[:t].to_numpy(float)
        ytr = y[:t]
        xte = X.iloc[t:t+1].to_numpy(float)

        mu = np.nanmean(xtr, axis=0)
        sd = np.nanstd(xtr, axis=0)
        sd[sd < 1e-12] = 1.0
        xtr = np.clip((np.nan_to_num(xtr) - mu) / sd, -8, 8)
        xte = np.clip((np.nan_to_num(xte) - mu) / sd, -8, 8)

        xtr_aug = np.c_[np.ones(len(xtr)), xtr]
        xte_aug = np.c_[np.ones(len(xte)), xte]

        with np.errstate(all="ignore"):
            xtx = np.nan_to_num(xtr_aug.T @ xtr_aug)
            xty = np.nan_to_num(xtr_aug.T @ ytr)
        reg = np.eye(xtx.shape[0]) * 5.0
        reg[0, 0] = 0.0
        beta = np.linalg.solve(xtx + reg, xty)
        yh = float((xte_aug @ beta)[0])

        preds_n.append(0.0)
        preds_r.append(yh)
        trues.append(float(y[t]))

    trues = np.array(trues)
    pn = np.array(preds_n)
    pr = np.array(preds_r)
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    hit = lambda a, b: float(np.mean(np.sign(a) == np.sign(b)))
    return {
        "folds": len(trues),
        "naive_mae": mae(trues, pn), "naive_rmse": rmse(trues, pn), "naive_hit": hit(trues, pn),
        "ridge_mae": mae(trues, pr), "ridge_rmse": rmse(trues, pr), "ridge_hit": hit(trues, pr),
    }


def main():
    args = parse_args()
    hs = [int(x) for x in args.horizons.split(",")]

    m = pd.read_parquet(args.mainpool).sort_values("dt_exch")
    m["iv_rsi"] = calc_rsi(m["iv_pool"], window=args.iv_rsi_window)
    m["iv_rsi_centered"] = (m["iv_rsi"] - 50.0) / 50.0
    for h in hs:
        m[f"y_{h}m"] = m["iv_pool"].shift(-h) - m["iv_pool"]
    m_feat = ["vega_signed_1m", "vega_abs_1m", "spread_pool_1m", "f_ret_1m", "iv_rsi_centered"]

    print("[MAINPOOL]")
    for h in hs:
        r = walkforward_eval(m, f"y_{h}m", m_feat, min_train=240)
        print(h, r)

    t = pd.read_parquet(args.top3).sort_values(["symbol", "dt_exch"])
    t["iv_rsi"] = t.groupby("symbol")["iv"].transform(lambda s: calc_rsi(s, window=args.iv_rsi_window))
    t["iv_rsi_centered"] = (t["iv_rsi"] - 50.0) / 50.0
    for h in hs:
        t[f"y_{h}m"] = t.groupby("symbol")["iv"].shift(-h) - t["iv"]
    t_feat = ["vega_signed_1m", "vega_abs_1m", "spread_1m", "d_volume_1m", "f_ret_1m", "iv_rsi_centered"]

    print("[TOP3]")
    for h in hs:
        vals = []
        for sym, g in t.groupby("symbol"):
            r = walkforward_eval(g, f"y_{h}m", t_feat, min_train=120)
            if r:
                vals.append(r)
        if not vals:
            print(h, None)
            continue
        agg = {k: float(np.mean([v[k] for v in vals])) for k in vals[0].keys()}
        print(h, agg)


if __name__ == "__main__":
    main()
