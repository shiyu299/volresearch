"""Second-round factor expansion + ablation + greedy ML-like factor mining (ridge walk-forward).

No external ML libs required. Uses:
- engineered vega/RSI/interaction features
- walk-forward ridge evaluation
- greedy forward feature selection (proxy for ML factor mining)
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
    p.add_argument("--ridge-alpha", type=float, default=5.0)
    p.add_argument("--topk", type=int, default=6)
    return p.parse_args()


def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    up_ema = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    dn_ema = dn.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = up_ema / dn_ema.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("dt_exch").copy()
    v = d["vega_signed_1m"].fillna(0.0)
    va = d["vega_abs_1m"].fillna(0.0)

    d["vega_ewm_5"] = v.ewm(span=5, adjust=False).mean()
    d["vega_ewm_15"] = v.ewm(span=15, adjust=False).mean()
    d["vega_abs_ewm_5"] = va.ewm(span=5, adjust=False).mean()
    d["vega_abs_ewm_15"] = va.ewm(span=15, adjust=False).mean()

    m30 = v.rolling(30, min_periods=10).mean()
    s30 = v.rolling(30, min_periods=10).std().replace(0, np.nan)
    d["vega_shock_z30"] = (v - m30) / s30

    d["vega_sign"] = np.sign(v)
    d["vega_sign_persist_5"] = d["vega_sign"].rolling(5, min_periods=3).mean()
    d["vega_sign_persist_15"] = d["vega_sign"].rolling(15, min_periods=5).mean()

    d["iv_rsi_6c"] = (calc_rsi(d["iv_pool"], 6) - 50.0) / 50.0
    d["iv_rsi_14c"] = (calc_rsi(d["iv_pool"], 14) - 50.0) / 50.0
    d["iv_rsi_21c"] = (calc_rsi(d["iv_pool"], 21) - 50.0) / 50.0

    d["vega_shock_x_rsi14"] = d["vega_shock_z30"] * d["iv_rsi_14c"]
    d["vega_abs_x_spread"] = d["vega_abs_1m"] * d["spread_pool_1m"].fillna(d["spread_pool_1m"].median())
    d["fret_abs"] = d["f_ret_1m"].abs()

    return d


def wf_ridge(df: pd.DataFrame, ycol: str, feats: list[str], min_train: int, alpha: float):
    d = df.dropna(subset=[ycol]).copy()
    if len(d) <= min_train + 20:
        return None
    X = d[feats].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = d[ycol].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    valid = np.isfinite(y)
    d = d.loc[valid].copy()
    X = X.loc[valid].copy()
    y = y[valid]

    tr, pr = [], []
    for t in range(min_train, len(d)):
        xtr = X.iloc[:t].to_numpy(float)
        ytr = y[:t]
        xte = X.iloc[t:t + 1].to_numpy(float)

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
        reg[0, 0] = 0.0
        beta = np.linalg.solve(xtx + reg, xty)
        pred = float((xb @ beta)[0])
        tr.append(float(y[t]))
        pr.append(pred)

    tr = np.array(tr)
    pr = np.array(pr)
    mae = float(np.mean(np.abs(tr - pr)))
    rmse = float(np.sqrt(np.mean((tr - pr) ** 2)))
    hit = float(np.mean(np.sign(tr) == np.sign(pr)))
    return {"folds": len(tr), "mae": mae, "rmse": rmse, "hit": hit}


def naive_zero(df: pd.DataFrame, ycol: str, min_train: int):
    d = df.dropna(subset=[ycol]).copy()
    if len(d) <= min_train + 20:
        return None
    y = d[ycol].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    y = y[min_train:]
    p = np.zeros_like(y)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    hit = float(np.mean(np.sign(y) == np.sign(p)))
    return {"folds": len(y), "mae": mae, "rmse": rmse, "hit": hit}


def spearman_ic(df: pd.DataFrame, f: str, y: str) -> float:
    z = df[[f, y]].dropna()
    if len(z) < 50:
        return np.nan
    return float(z[f].corr(z[y], method="spearman"))


def greedy_select(df: pd.DataFrame, ycol: str, base: list[str], cands: list[str], min_train: int, alpha: float, topk: int):
    chosen = list(base)
    rest = [c for c in cands if c not in chosen]
    best = wf_ridge(df, ycol, chosen, min_train, alpha)
    if best is None:
        return chosen, []
    history = [("BASE", chosen.copy(), best)]

    for _ in range(topk):
        best_add = None
        best_score = best["rmse"]
        for c in rest:
            r = wf_ridge(df, ycol, chosen + [c], min_train, alpha)
            if r is None:
                continue
            if r["rmse"] < best_score:
                best_score = r["rmse"]
                best_add = (c, r)
        if best_add is None:
            break
        c, r = best_add
        chosen.append(c)
        rest.remove(c)
        best = r
        history.append(("ADD", chosen.copy(), best))
    return chosen, history


def main():
    args = parse_args()
    hs = [int(x) for x in args.horizons.split(",")]

    df = pd.read_parquet(args.input)
    df = feature_engineer(df)

    base_feats = ["vega_signed_1m", "vega_abs_1m", "spread_pool_1m", "f_ret_1m"]
    candidate_feats = [
        "vega_ewm_5", "vega_ewm_15", "vega_abs_ewm_5", "vega_abs_ewm_15",
        "vega_shock_z30", "vega_sign_persist_5", "vega_sign_persist_15",
        "iv_rsi_6c", "iv_rsi_14c", "iv_rsi_21c",
        "vega_shock_x_rsi14", "vega_abs_x_spread", "fret_abs",
    ]

    for h in hs:
        ycol = f"y_{h}m"
        df[ycol] = df["iv_pool"].shift(-h) - df["iv_pool"]

        print(f"\n=== H={h}m ===")
        n = naive_zero(df, ycol, args.min_train)
        b = wf_ridge(df, ycol, base_feats, args.min_train, args.ridge_alpha)
        print("naive:", n)
        print("base:", b)

        print("-- IC rank (candidate only, spearman) --")
        ics = []
        for f in candidate_feats:
            ic = spearman_ic(df, f, ycol)
            ics.append((f, ic))
        ics = sorted(ics, key=lambda x: abs(x[1]) if pd.notna(x[1]) else -1, reverse=True)
        for f, ic in ics[:8]:
            print(f"  {f:18s} ic={ic:+.5f}")

        chosen, hist = greedy_select(df, ycol, base_feats, candidate_feats, args.min_train, args.ridge_alpha, args.topk)
        print("-- Greedy selection path --")
        for step, feats, r in hist:
            print(f"  {step:4s} n={len(feats):2d} rmse={r['rmse']:.6f} hit={r['hit']:.4f} last={feats[-1]}")
        print("final_feats:", chosen)


if __name__ == "__main__":
    main()
