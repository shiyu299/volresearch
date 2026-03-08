# -*- coding: utf-8 -*-
"""
Preprocess IV dataset for factor research.
- Add bid_iv / ask_iv / mid_iv via Black-76 inversion (clipped to [0.05, 1.5])
- Build futures microprice and imbalance from bid/ask + bidvol/askvol
- Keep columns needed by factor pipeline
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


SIGMA_LOWER = 0.05
SIGMA_UPPER = 1.50


def black76_price(F, K, T, r, sigma, cp):
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return np.nan
    vs = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vs
    d2 = d1 - vs
    disc = np.exp(-r * T)
    if cp == "C":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(F, K, T, r, px, cp, sigma0=0.25):
    if not np.isfinite(px) or px <= 0 or T <= 0 or F <= 0 or K <= 0:
        return np.nan
    s = float(np.clip(sigma0, SIGMA_LOWER, SIGMA_UPPER))
    for _ in range(50):
        p = black76_price(F, K, T, r, s, cp)
        if not np.isfinite(p):
            return np.nan
        diff = p - px
        if abs(diff) < 1e-6:
            return float(np.clip(s, SIGMA_LOWER, SIGMA_UPPER))
        # vega
        vs = s * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * s * s * T) / vs
        v = F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
        if v <= 1e-12 or not np.isfinite(v):
            break
        s = s - diff / v
        s = float(np.clip(s, SIGMA_LOWER, SIGMA_UPPER))
    return float(np.clip(s, SIGMA_LOWER, SIGMA_UPPER))


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["dt_exch"] = pd.to_datetime(x["dt_exch"], errors="coerce")
    x = x.dropna(subset=["dt_exch"]).sort_values("dt_exch")

    # Futures microprice / imbalance
    bv = x.get("bidvol1", 0).fillna(0.0)
    av = x.get("askvol1", 0).fillna(0.0)
    bp = x.get("bidprice1", np.nan)
    ap = x.get("askprice1", np.nan)
    denom = bv + av
    x["fut_microprice"] = np.where(
        (denom > 0) & np.isfinite(bp) & np.isfinite(ap),
        (ap * bv + bp * av) / denom,
        x.get("F_used", np.nan),
    )
    x["fut_book_imbalance"] = np.where(denom > 0, (bv - av) / denom, np.nan)

    # Option bid/ask/mid IV (bounded)
    for c in ["bid_iv", "ask_iv", "mid_iv"]:
        x[c] = np.nan

    opt = x.get("is_option", False) == True
    rows = x[opt & x["cp"].notna() & x["K"].notna() & x["T"].notna() & x["F_used"].notna()]

    for i, r in rows.iterrows():
        F = float(r["F_used"])
        K = float(r["K"])
        T = float(r["T"])
        cp = str(r["cp"]).upper()

        bid_px = float(r["bidprice1"]) if pd.notna(r.get("bidprice1")) else np.nan
        ask_px = float(r["askprice1"]) if pd.notna(r.get("askprice1")) else np.nan
        mid_px = float(r["mid"]) if pd.notna(r.get("mid")) else np.nan

        x.at[i, "bid_iv"] = implied_vol(F, K, T, 0.0, bid_px, cp)
        x.at[i, "ask_iv"] = implied_vol(F, K, T, 0.0, ask_px, cp)
        x.at[i, "mid_iv"] = implied_vol(F, K, T, 0.0, mid_px, cp)

    return x


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.input.lower().endswith(".parquet"):
        df0 = pd.read_parquet(args.input)
    else:
        df0 = pd.read_csv(args.input)
    out = preprocess(df0)
    out.to_parquet(args.output, index=False)
    print(f"saved: {args.output}, rows={len(out)}")
