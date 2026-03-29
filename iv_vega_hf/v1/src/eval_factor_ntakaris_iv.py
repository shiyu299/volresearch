"""Ntakaris-style broad technical/quant factor mining, adapted to IV prediction.

Use many hand-crafted indicators (technical + quantitative proxies),
but target is IV change y_h = iv_pool(t+h)-iv_pool(t).
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/mainpool_1m.parquet")
    p.add_argument("--horizons", default="1,5,15")
    p.add_argument("--min-train", type=int, default=240)
    p.add_argument("--ridge-alpha", type=float, default=8.0)
    p.add_argument("--topk", type=int, default=12)
    return p.parse_args()


def rsi(x: pd.Series, n: int) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    a = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    b = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = a / b.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("dt_exch").copy()
    iv = d["iv_pool"]
    fr = d["f_ret_1m"].fillna(0)
    v = d["vega_signed_1m"].fillna(0)
    va = d["vega_abs_1m"].fillna(0)
    sp = d["spread_pool_1m"].fillna(d["spread_pool_1m"].median())

    # MA/EMA family
    for n in [3, 5, 10, 20, 30, 60]:
        d[f"iv_sma_{n}"] = iv.rolling(n, min_periods=max(2, n // 2)).mean()
        d[f"iv_ema_{n}"] = iv.ewm(span=n, adjust=False).mean()
        d[f"iv_dev_sma_{n}"] = (iv - d[f"iv_sma_{n}"]) / (d[f"iv_sma_{n}"].abs() + 1e-9)
        d[f"iv_dev_ema_{n}"] = (iv - d[f"iv_ema_{n}"]) / (d[f"iv_ema_{n}"].abs() + 1e-9)

    # Momentum / ROC
    for n in [1, 3, 5, 10, 20, 30]:
        d[f"iv_mom_{n}"] = iv - iv.shift(n)
        d[f"iv_roc_{n}"] = iv.pct_change(n)

    # RSI / stochastic / BB / WilliamsR
    for n in [6, 14, 21]:
        d[f"iv_rsi_{n}"] = (rsi(iv, n) - 50) / 50

    for n in [10, 20, 30]:
        ll = iv.rolling(n, min_periods=max(3, n // 3)).min()
        hh = iv.rolling(n, min_periods=max(3, n // 3)).max()
        d[f"iv_stoch_k_{n}"] = (iv - ll) / (hh - ll + 1e-9)
        d[f"iv_willr_{n}"] = -100 * (hh - iv) / (hh - ll + 1e-9)
        d[f"iv_bb_z_{n}"] = (iv - iv.rolling(n, min_periods=max(3, n // 3)).mean()) / (
            iv.rolling(n, min_periods=max(3, n // 3)).std().replace(0, np.nan)
        )

    # MACD-like
    ema12 = iv.ewm(span=12, adjust=False).mean()
    ema26 = iv.ewm(span=26, adjust=False).mean()
    d["iv_macd"] = ema12 - ema26
    d["iv_macd_signal"] = d["iv_macd"].ewm(span=9, adjust=False).mean()
    d["iv_macd_hist"] = d["iv_macd"] - d["iv_macd_signal"]

    # Quantitative/proxy family
    d["flow_pressure"] = v / (va + 1e-9)
    d["flow_pressure_ewm15"] = d["flow_pressure"].ewm(span=15, adjust=False).mean()
    d["vega_shock_z30"] = (v - v.rolling(30, min_periods=10).mean()) / v.rolling(30, min_periods=10).std().replace(0, np.nan)
    d["vega_regime_10"] = np.sign(v).rolling(10, min_periods=5).mean()
    d["rv_fut_5"] = fr.rolling(5, min_periods=3).std()
    d["rv_fut_15"] = fr.rolling(15, min_periods=5).std()
    d["rv_fut_30"] = fr.rolling(30, min_periods=10).std()
    d["liq_stress"] = sp * va
    d["spread_ewm_10"] = sp.ewm(span=10, adjust=False).mean()
    d["d_spread_1"] = sp.diff(1)

    # Interaction terms
    d["shock_x_rsi14"] = d["vega_shock_z30"] * d["iv_rsi_14"]
    d["mom5_x_flow"] = d["iv_mom_5"] * d["flow_pressure_ewm15"]
    d["bb20_x_flow"] = d["iv_bb_z_20"] * d["flow_pressure"]

    return d


def wf_ridge(df: pd.DataFrame, ycol: str, feats: list[str], min_train: int, alpha: float):
    z = df.dropna(subset=[ycol]).copy()
    X = z[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).clip(-1e6, 1e6)
    y = z[ycol].replace([np.inf, -np.inf], np.nan)
    keep = y.notna()
    X = X.loc[keep]
    y = y.loc[keep].to_numpy(float)
    if len(y) <= min_train + 20:
        return None
    tr, pr = [], []
    for t in range(min_train, len(y)):
        xtr = X.iloc[:t].to_numpy(float)
        xte = X.iloc[t:t + 1].to_numpy(float)
        ytr = y[:t]
        mu = np.nanmean(xtr, axis=0)
        sd = np.nanstd(xtr, axis=0)
        sd[sd < 1e-12] = 1.0
        xtr = np.clip((np.nan_to_num(xtr) - mu) / sd, -8, 8)
        xte = np.clip((np.nan_to_num(xte) - mu) / sd, -8, 8)
        xa = np.c_[np.ones(len(xtr)), xtr]
        xb = np.c_[np.ones(len(xte)), xte]
        xtx = np.nan_to_num(xa.T @ xa)
        xty = np.nan_to_num(xa.T @ ytr)
        reg = np.eye(xtx.shape[0]) * alpha
        reg[0, 0] = 0
        beta = np.linalg.solve(xtx + reg, xty)
        pr.append(float((xb @ beta)[0]))
        tr.append(float(y[t]))
    tr = np.array(tr)
    pr = np.array(pr)
    return {
        "rmse": float(np.sqrt(np.mean((tr - pr) ** 2))),
        "hit": float(np.mean(np.sign(tr) == np.sign(pr))),
        "folds": len(tr),
    }


def ic(df: pd.DataFrame, f: str, y: str) -> float:
    z = df[[f, y]].dropna()
    if len(z) < 80:
        return np.nan
    return float(z[f].corr(z[y], method="spearman"))


def greedy(df: pd.DataFrame, y: str, base: list[str], cands: list[str], min_train: int, alpha: float, topk: int):
    chosen = list(base)
    rest = [c for c in cands if c not in chosen]
    best = wf_ridge(df, y, chosen, min_train, alpha)
    path = [("BASE", chosen.copy(), best)]
    if best is None:
        return chosen, path
    for _ in range(topk):
        win = None
        win_rmse = best["rmse"]
        for c in rest:
            r = wf_ridge(df, y, chosen + [c], min_train, alpha)
            if r and r["rmse"] < win_rmse:
                win_rmse = r["rmse"]
                win = (c, r)
        if win is None:
            break
        c, r = win
        chosen.append(c)
        rest.remove(c)
        best = r
        path.append(("ADD", chosen.copy(), r))
    return chosen, path


def main():
    args = parse_args()
    hs = [int(x) for x in args.horizons.split(",")]
    d = add_features(pd.read_parquet(args.input))

    base = ["vega_signed_1m", "vega_abs_1m", "spread_pool_1m", "f_ret_1m"]
    reserved = {"dt_exch", "iv_pool", "F_used", *base}
    cands = [c for c in d.columns if c not in reserved and not c.startswith("y_")]

    for h in hs:
        y = f"y_{h}m"
        d[y] = d["iv_pool"].shift(-h) - d["iv_pool"]
        b = wf_ridge(d, y, base, args.min_train, args.ridge_alpha)
        print(f"\n=== H={h}m ===")
        print("base:", b)
        ranks = sorted([(f, ic(d, f, y)) for f in cands], key=lambda x: abs(x[1]) if pd.notna(x[1]) else -1, reverse=True)
        print("top_ic:")
        for f, v in ranks[:12]:
            print(f"  {f:22s} {v:+.5f}")
        final, path = greedy(d, y, base, cands, args.min_train, args.ridge_alpha, args.topk)
        print("path:")
        for step, feats, r in path:
            print(f"  {step:4s} n={len(feats):2d} rmse={r['rmse']:.6f} hit={r['hit']:.4f} last={feats[-1]}")
        print("final:", final)


if __name__ == "__main__":
    main()
