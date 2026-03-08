# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorDef:
    factor_id: str
    label: str
    description: str
    compute: Callable[[pd.DataFrame], pd.Series]
    default_q: float = 0.95
    default_op: str = ">="  # ">=" | "<="


def _rolling_bins_for_seconds(base_rule: str, sec: int) -> int:
    try:
        step = pd.Timedelta(base_rule).total_seconds()
    except Exception:
        step = 1.0
    step = max(step, 1.0)
    return max(1, int(round(sec / step)))


def build_factor_base_frame(df_raw: pd.DataFrame, symbol: str, base_rule: str) -> pd.DataFrame:
    dfo = df_raw.copy()
    dfo["dt_exch"] = pd.to_datetime(dfo["dt_exch"], errors="coerce")
    dfo = dfo.dropna(subset=["dt_exch"]).sort_values("dt_exch")

    # option leg (selected contract)
    opt = dfo[dfo.get("symbol", "").astype(str) == str(symbol)].copy()

    out = pd.DataFrame()
    if not opt.empty:
        idx = pd.date_range(opt["dt_exch"].min().floor(base_rule), opt["dt_exch"].max().ceil(base_rule), freq=base_rule)
        out = pd.DataFrame(index=idx)

        if "iv" in opt.columns:
            out["iv"] = opt.set_index("dt_exch")["iv"].resample(base_rule).last().reindex(idx).ffill()

        if "traded_vega" in opt.columns:
            out["traded_vega"] = opt.set_index("dt_exch")["traded_vega"].resample(base_rule).sum().reindex(idx).fillna(0.0)
        else:
            out["traded_vega"] = 0.0

        if "spread" in opt.columns:
            out["opt_spread"] = opt.set_index("dt_exch")["spread"].resample(base_rule).median().reindex(idx).ffill()
        else:
            out["opt_spread"] = np.nan

    # futures leg (for no_trade_move / div)
    fut = dfo[dfo.get("is_future", False) == True].copy() if "is_future" in dfo.columns else pd.DataFrame()
    if not fut.empty:
        if out.empty:
            idx = pd.date_range(fut["dt_exch"].min().floor(base_rule), fut["dt_exch"].max().ceil(base_rule), freq=base_rule)
            out = pd.DataFrame(index=idx)
        else:
            idx = out.index

        if "F_used" in fut.columns:
            out["fut_price"] = fut.set_index("dt_exch")["F_used"].resample(base_rule).last().reindex(idx).ffill()
        if "d_volume" in fut.columns:
            out["fut_dvol"] = fut.set_index("dt_exch")["d_volume"].resample(base_rule).sum().reindex(idx).fillna(0.0)
        else:
            out["fut_dvol"] = 0.0

    if out.empty:
        return out

    win60 = _rolling_bins_for_seconds(base_rule, 60)

    # derived common fields
    if "fut_price" in out.columns:
        out["fut_ret_60s"] = out["fut_price"].pct_change(win60)
        out["fut_dvol_60s"] = out["fut_dvol"].rolling(win60).sum() if "fut_dvol" in out.columns else np.nan
    else:
        out["fut_ret_60s"] = np.nan
        out["fut_dvol_60s"] = np.nan

    if "iv" in out.columns:
        out["iv_chg_60s"] = out["iv"].diff(win60)
    else:
        out["iv_chg_60s"] = np.nan

    out["abs_traded_vega"] = out.get("traded_vega", 0.0).abs()
    out["opt_spread_60s"] = out.get("opt_spread", np.nan).rolling(win60).median()
    out["no_trade_move_60"] = out["fut_ret_60s"].abs() / (out["fut_dvol_60s"] + 1.0)
    out["iv_f_div_60"] = out["iv_chg_60s"] * np.sign(out["fut_ret_60s"].fillna(0))

    return out


def _factor_abs_traded_vega(df: pd.DataFrame) -> pd.Series:
    return df.get("abs_traded_vega", pd.Series(index=df.index, dtype=float))


def _factor_opt_spread_60s(df: pd.DataFrame) -> pd.Series:
    return df.get("opt_spread_60s", pd.Series(index=df.index, dtype=float))


def _factor_no_trade_move_60(df: pd.DataFrame) -> pd.Series:
    return df.get("no_trade_move_60", pd.Series(index=df.index, dtype=float))


def _factor_iv_f_div_60_abs(df: pd.DataFrame) -> pd.Series:
    s = df.get("iv_f_div_60", pd.Series(index=df.index, dtype=float))
    return s.abs()


FACTOR_REGISTRY: Dict[str, FactorDef] = {
    "abs_traded_vega": FactorDef(
        factor_id="abs_traded_vega",
        label="abs_traded_vega（单合约）",
        description="单合约绝对成交vega",
        compute=_factor_abs_traded_vega,
        default_q=0.95,
        default_op=">=",
    ),
    "opt_spread_60s": FactorDef(
        factor_id="opt_spread_60s",
        label="opt_spread_60s",
        description="60秒期权中位价差",
        compute=_factor_opt_spread_60s,
        default_q=0.80,
        default_op=">=",
    ),
    "no_trade_move_60": FactorDef(
        factor_id="no_trade_move_60",
        label="no_trade_move_60",
        description="60秒无量波动指标",
        compute=_factor_no_trade_move_60,
        default_q=0.80,
        default_op=">=",
    ),
    "iv_f_div_60_abs": FactorDef(
        factor_id="iv_f_div_60_abs",
        label="abs(iv_f_div_60)",
        description="60秒IV-F背离绝对值",
        compute=_factor_iv_f_div_60_abs,
        default_q=0.85,
        default_op=">=",
    ),
}


def list_factors() -> Dict[str, FactorDef]:
    return FACTOR_REGISTRY


def evaluate_factor_trigger(
    base_df: pd.DataFrame,
    factor_id: str,
    *,
    mode: str = "quantile",  # quantile | absolute
    q: float = 0.95,
    op: str = ">=",
    value: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series, float]:
    """Return (signal_series, trigger_bool_series, threshold_used)."""
    if factor_id not in FACTOR_REGISTRY:
        return pd.Series(dtype=float), pd.Series(dtype=bool), np.nan

    f = FACTOR_REGISTRY[factor_id]
    s = f.compute(base_df).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)

    if s.empty or s.notna().sum() == 0:
        return s, pd.Series(False, index=s.index), np.nan

    if mode == "absolute":
        thr = float(value) if value is not None else np.nan
    else:
        qq = float(q)
        qq = min(max(qq, 0.0), 1.0)
        thr = float(s.quantile(qq))

    if not np.isfinite(thr):
        return s, pd.Series(False, index=s.index), np.nan

    if op == "<=":
        trig = s.le(thr)
    else:
        trig = s.ge(thr)

    trig = trig.fillna(False)
    return s, trig, thr
