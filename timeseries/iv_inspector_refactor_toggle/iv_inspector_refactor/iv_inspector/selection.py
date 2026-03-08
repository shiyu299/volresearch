# -*- coding: utf-8 -*-
"""
Selection helpers for IV Inspector.

Key rule changes (per user spec):
- Selection of "ATM±n contracts" is based on strike proximity to F_used, NOT on iv availability.
- ATM strike is defined as the strike K with minimum |K - F_used| (using the latest F_used in the group).
- All rows at the ATM strike are forcibly included (e.g., both Call and Put if present).
- For non-ATM rows, optional OTM filter can be applied (Call: K>=F, Put: K<=F).
- If some selected contracts have iv missing, they are NOT replaced by further OTM strikes. Downstream
  aggregation should simply exclude NaN iv from weights, resulting in n_used < n_target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

def infer_F(g: pd.DataFrame) -> float:
    if "F_used" not in g.columns:
        return np.nan
    s = g["F_used"].dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])

def otm_atm_mask(df: pd.DataFrame, F: float) -> pd.Series:
    """OTM filter for non-ATM rows."""
    if not np.isfinite(F):
        return pd.Series([True] * len(df), index=df.index)

    if "cp" not in df.columns:
        return (df["K"] >= F)

    cp = df["cp"].astype(str).str.upper()
    is_call = cp.str.startswith("C")
    is_put = cp.str.startswith("P")
    other = ~(is_call | is_put)

    mask_call = is_call & (df["K"] >= F)
    mask_put = is_put & (df["K"] <= F)
    return mask_call | mask_put | other

def _infer_atm_strike(df: pd.DataFrame, F: float) -> float:
    """Return strike K that is closest to F."""
    if df.empty or "K" not in df.columns or not np.isfinite(F):
        return np.nan
    k = df["K"].astype(float)
    k = k[np.isfinite(k)]
    if k.empty:
        return np.nan
    # choose K with min abs distance; if ties, take the smaller K (stable)
    dist = (k - F).abs()
    min_dist = dist.min()
    cand = k[dist == min_dist]
    return float(cand.min())

def pick_atm_n_options(
    group: pd.DataFrame,
    n: int,
    only_otm_atm: bool = True,
    otm_atm_only=None,  # backward-compat alias
) -> pd.DataFrame:
    """
    For a given time-bucket group, pick ATM±n rows (contracts).
    IMPORTANT: selection does NOT drop NaN iv; it only requires F_used and K.
    Downstream should decide how to handle NaN iv (e.g., ffill/state_adjust then drop if still NaN).
    """
    if otm_atm_only is not None:
        only_otm_atm = otm_atm_only

    if group is None or group.empty:
        return group.iloc[0:0].copy()

    need_cols = [c for c in ["F_used", "K"] if c in group.columns]
    if len(need_cols) < 2:
        return group.iloc[0:0].copy()

    g = group.dropna(subset=["F_used", "K"]).copy()
    if g.empty:
        return g

    F = infer_F(g)
    if not np.isfinite(F) or F <= 0:
        return g.iloc[0:0].copy()

    atm_k = _infer_atm_strike(g, F)
    if not np.isfinite(atm_k):
        return g.iloc[0:0].copy()

    # 1) force include all rows at ATM strike (call+put if present)
    atm_rows = g[g["K"].astype(float) == float(atm_k)].copy()
    rest = g.drop(index=atm_rows.index)

    # 2) apply OTM filter only to non-ATM rows
    if only_otm_atm and not rest.empty:
        rest = rest.loc[otm_atm_mask(rest, F)].copy()

    # 3) rank by distance and take first (n - len(atm_rows)) from rest
    rest["atm_dist"] = (rest["K"].astype(float) - F).abs()
    take_rest = max(0, int(n) - len(atm_rows))
    rest_pick = rest.sort_values("atm_dist").head(take_rest)

    out = pd.concat([atm_rows, rest_pick], axis=0, ignore_index=False)
    out["F_bucket"] = F
    out["atm_dist"] = (out["K"].astype(float) - F).abs()
    out = out.sort_values("atm_dist").head(max(1, int(n)))  # safety
    return out

def pick_contracts_atm_unique(df_interval: pd.DataFrame, n: int, otm_atm_only: bool) -> List[str]:
    """
    Used by drilldown. Pick up to n unique symbols nearest to F, based on strike distance.
    Note: This function does NOT require iv availability.
    """
    if df_interval is None or df_interval.empty or "symbol" not in df_interval.columns or "K" not in df_interval.columns:
        return []

    g = df_interval.dropna(subset=["symbol", "K"]).copy()
    if g.empty:
        return []

    F = infer_F(g)
    if not np.isfinite(F) or F <= 0:
        return []

    # Apply OTM filter to entire interval if requested (drilldown intention)
    if otm_atm_only:
        g = g.loc[otm_atm_mask(g, F)].copy()
        if g.empty:
            return []

    g["atm_dist"] = (g["K"].astype(float) - F).abs()
    # For each symbol, use its closest strike distance within the interval
    best = g.groupby("symbol", as_index=False)["atm_dist"].min().sort_values("atm_dist")
    return best["symbol"].head(max(1, int(n))).tolist()

def pick_contracts_top_volume(df_interval: pd.DataFrame, m: int) -> List[str]:
    if df_interval is None or df_interval.empty or "symbol" not in df_interval.columns:
        return []

    cand_cols = ["d_volume", "volume", "trade_volume_lots", "d_totalvaluetraded", "totalvaluetraded"]
    vol_col = next((c for c in cand_cols if c in df_interval.columns), None)
    if vol_col is None:
        return []

    g = df_interval.dropna(subset=["symbol"]).copy()
    s = g.groupby("symbol")[vol_col].sum().sort_values(ascending=False)
    return s.head(max(1, int(m))).index.tolist()
