from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def infer_f(g: pd.DataFrame) -> float:
    if "F_used" not in g.columns:
        return np.nan
    s = g["F_used"].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def otm_atm_mask(df: pd.DataFrame, f_value: float) -> pd.Series:
    if not np.isfinite(f_value):
        return pd.Series([True] * len(df), index=df.index)
    if "cp" not in df.columns:
        return df["K"] >= f_value

    cp = df["cp"].astype(str).str.upper()
    is_call = cp.str.startswith("C")
    is_put = cp.str.startswith("P")
    other = ~(is_call | is_put)
    return (is_call & (df["K"] >= f_value)) | (is_put & (df["K"] <= f_value)) | other


def pick_contracts_atm_unique(df_interval: pd.DataFrame, n: int, otm_atm_only: bool) -> List[str]:
    if df_interval is None or df_interval.empty or "symbol" not in df_interval.columns or "K" not in df_interval.columns:
        return []

    g = df_interval.dropna(subset=["symbol", "K"]).copy()
    if g.empty:
        return []

    f_value = infer_f(g)
    if not np.isfinite(f_value) or f_value <= 0:
        return []

    if otm_atm_only:
        g = g.loc[otm_atm_mask(g, f_value)].copy()
        if g.empty:
            return []

    g["atm_dist"] = (g["K"].astype(float) - f_value).abs()
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

