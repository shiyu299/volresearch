"""Broad factor zoo mining for IV forecasting (traditional + technical + non-traditional proxies).

Uses only numpy/pandas (no sklearn dependency).
Outputs IC ranking + greedy walk-forward feature selection per horizon.
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
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def rsi(x: pd.Series, n: int) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    a = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    b = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = a / b.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("dt_exch").copy()
    iv = d["iv_pool"]
    v = d["vega_signed_1m"].fillna(0)
    va = d["vega_abs_1m"].fillna(0)
    sp = d["spread_pool_1m"].fillna(d["spread_pool_1m"].median())
    fr = d["f_ret_1m"].fillna(0)

    # Traditional
    d["vega_ewm5"] = v.ewm(span=5, adjust=False).mean()
    d["vega_ewm15"] = v.ewm(span=15, adjust=False).mean()
    d["vega_abs_ewm5"] = va.ewm(span=5, adjust=False).mean()
    d["vega_abs_ewm30"] = va.ewm(span=30, adjust=False).mean()
    d["iv_mom_5"] = iv - iv.shift(5)
    d["iv_mom_15"] = iv - iv.shift(15)

    # Technical indicators on IV
    d["iv_rsi6"] = (rsi(iv, 6) - 50) / 50
    d["iv_rsi14"] = (rsi(iv, 14) - 50) / 50
    d["iv_rsi21"] = (rsi(iv, 21) - 50) / 50
    ma20 = iv.rolling(20, min_periods=10).mean()
    sd20 = iv.rolling(20, min_periods=10).std().replace(0, np.nan)
    d["iv_bb_z20"] = (iv - ma20) / sd20
    d["iv_roc_10"] = iv.pct_change(10)

    # Non-traditional/proxy features
    d["f_rv_5"] = fr.rolling(5, min_periods=3).std()
    d["f_rv_15"] = fr.rolling(15, min_periods=5).std()
    d["flow_pressure"] = v / (va + 1e-8)
    d["flow_pressure_ewm15"] = d["flow_pressure"].ewm(span=15, adjust=False).mean()
    d["liquidity_stress"] = sp * va
    d["vega_shock_z30"] = (v - v.rolling(30, min_periods=10).mean()) / v.rolling(30, min_periods=10).std().replace(0, np.nan)
    d["flow_regime"] = np.sign(v).rolling(10, min_periods=5).mean()

    # Interactions
    d["shock_x_rsi"] = d["vega_shock_z30"] * d["iv_rsi14"]
    d["spread_x_rv"] = sp * d["f_rv_15"]
    d["mom_x_pressure"] = d["iv_mom_5"] * d["flow_pressure_ewm15"]

    return d


def wf_ridge(df: pd.DataFrame, ycol: str, feats: list[str], min_train: int, alpha: float):
    z = df.dropna(subset=[ycol]).copy()
    X = z[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).clip(-1e6, 1e6)
    y = z[ycol].replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].to_numpy(float)
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
        pred = float((xb @ beta)[0])
        tr.append(float(y[t]))
        pr.append(pred)
    tr = np.array(tr); pr = np.array(pr)
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


def greedy(df: pd.DataFrame, ycol: str, base: list[str], cands: list[str], min_train: int, alpha: float, topk: int):
    chosen = list(base)
    rest = [c for c in cands if c not in chosen]
    best = wf_ridge(df, ycol, chosen, min_train, alpha)
    path = [("BASE", chosen.copy(), best)]
    if best is None:
        return chosen, path

    for _ in range(topk):
        winner = None
        win_rmse = best["rmse"]
        for c in rest:
            r = wf_ridge(df, ycol, chosen + [c], min_train, alpha)
            if r and r["rmse"] < win_rmse:
                win_rmse = r["rmse"]
                winner = (c, r)
        if winner is None:
            break
        c, r = winner
        chosen.append(c)
        rest.remove(c)
        best = r
        path.append(("ADD", chosen.copy(), r))
    return chosen, path


def main():
    args = parse_args()
    hs = [int(x) for x in args.horizons.split(",")]
    d = add_feats(pd.read_parquet(args.input))

    base = ["vega_signed_1m", "vega_abs_1m", "spread_pool_1m", "f_ret_1m"]
    cands = [c for c in d.columns if c not in {"dt_exch", "iv_pool", "F_used"} and c not in base and not c.startswith("y_")]

    for h in hs:
        y = f"y_{h}m"
        d[y] = d["iv_pool"].shift(-h) - d["iv_pool"]
        b = wf_ridge(d, y, base, args.min_train, args.ridge_alpha)
        print(f"\n=== H={h}m ===")
        print("base:", b)
        ranks = sorted([(f, ic(d, f, y)) for f in cands], key=lambda x: abs(x[1]) if pd.notna(x[1]) else -1, reverse=True)
        print("top_ic:")
        for f, v in ranks[:10]:
            print(f"  {f:20s} {v:+.5f}")
        final, path = greedy(d, y, base, cands, args.min_train, args.ridge_alpha, args.topk)
        print("path:")
        for step, feats, r in path:
            print(f"  {step:4s} n={len(feats):2d} rmse={r['rmse']:.6f} hit={r['hit']:.4f} last={feats[-1]}")
        print("final:", final)


if __name__ == "__main__":
    main()
