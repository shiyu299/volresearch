from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def zscore_roll(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=max(5, win // 3)).mean()
    sd = s.rolling(win, min_periods=max(5, win // 3)).std()
    return (s - m) / sd.replace(0, np.nan)


def willr_on_iv(iv: pd.Series, win: int = 10) -> pd.Series:
    hh = iv.rolling(win, min_periods=win).max()
    ll = iv.rolling(win, min_periods=win).min()
    den = (hh - ll).replace(0, np.nan)
    return -100.0 * (hh - iv) / den


def build_sc_factors(mainpool: pd.DataFrame) -> pd.DataFrame:
    """Build SC core + 2nd-batch factors from mainpool 1m data.

    Required cols: dt_exch, iv_pool, vega_signed_1m, F_used
    Optional cols: spread_pool_1m
    """
    df = mainpool.sort_values("dt_exch").copy()

    iv = df["iv_pool"].astype(float)
    flow = df["vega_signed_1m"].fillna(0.0).astype(float)

    e5 = ema(iv, 5)
    df["flow"] = flow
    df["iv_dev_ema5_ratio"] = (iv - e5) / e5.abs().replace(0, np.nan)
    df["iv_mom3"] = iv - iv.shift(3)
    df["iv_willr10"] = willr_on_iv(iv, 10)
    df["flow_ema10"] = ema(flow, 10)

    # F-IV block
    dF = np.log(df["F_used"].astype(float) / df["F_used"].astype(float).shift(1))
    shockF = dF + 0.5 * dF.shift(1) + 0.25 * dF.shift(2)
    dIV = iv.diff(1)

    cov = dIV.rolling(60, min_periods=30).cov(shockF)
    var = shockF.rolling(60, min_periods=30).var()
    betaF = cov / var.replace(0, np.nan)
    resid = dIV - betaF * shockF

    df["shockF"] = shockF
    df["resid_z"] = zscore_roll(resid, 60)

    if "spread_pool_1m" in df.columns:
        df["spread_pool_1m"] = df["spread_pool_1m"].astype(float)

    return df


def add_targets(df: pd.DataFrame, horizons=(1, 3, 5)) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"y_{h}m"] = out["iv_pool"].shift(-h) - out["iv_pool"]
    return out
