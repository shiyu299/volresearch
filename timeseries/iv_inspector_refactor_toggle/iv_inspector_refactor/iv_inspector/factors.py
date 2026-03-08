# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

VCR_MIN_R2 = 0.20  # r^2 gate for vcr_divergence; <=0 means no gate

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

    win5 = _rolling_bins_for_seconds(base_rule, 5)
    win10 = _rolling_bins_for_seconds(base_rule, 10)
    win20 = _rolling_bins_for_seconds(base_rule, 20)
    win60 = _rolling_bins_for_seconds(base_rule, 60)
    win30m = _rolling_bins_for_seconds(base_rule, 30 * 60)

    # derived common fields
    if "fut_price" in out.columns:
        out["fut_dF_5s"] = out["fut_price"].diff(win5)
        out["fut_ret_10s"] = out["fut_price"].pct_change(win10)
        out["fut_ret_60s"] = out["fut_price"].pct_change(win60)
        out["fut_dvol_60s"] = out["fut_dvol"].rolling(win60).sum() if "fut_dvol" in out.columns else np.nan
    else:
        out["fut_dF_5s"] = np.nan
        out["fut_ret_10s"] = np.nan
        out["fut_ret_60s"] = np.nan
        out["fut_dvol_60s"] = np.nan

    if "iv" in out.columns:
        out["iv_dIV_5s"] = out["iv"].diff(win5)
        out["iv_chg_10s"] = out["iv"].diff(win10)
        out["iv_chg_20s_abs"] = out["iv"].diff(win20).abs()
        out["iv_chg_60s"] = out["iv"].diff(win60)
    else:
        out["iv_dIV_5s"] = np.nan
        out["iv_chg_10s"] = np.nan
        out["iv_chg_20s_abs"] = np.nan
        out["iv_chg_60s"] = np.nan

    # vcr_divergence: 30分钟滚动回归 dIV = beta * dF + e，然后比较 real_dIV 与 fair_dIV
    # 增加 r^2 过滤：仅当 r^2 >= VCR_MIN_R2 时，beta/fair_dIV/vcr_divergence 才有效
    x = out.get("fut_dF_5s", pd.Series(np.nan, index=out.index)).astype(float)
    y = out.get("iv_dIV_5s", pd.Series(np.nan, index=out.index)).astype(float)
    cov_xy = y.rolling(win30m, min_periods=max(30, win5)).cov(x)
    var_x = x.rolling(win30m, min_periods=max(30, win5)).var()
    var_y = y.rolling(win30m, min_periods=max(30, win5)).var()
    beta_30m = cov_xy / var_x.replace(0.0, np.nan)
    r2_30m = (cov_xy * cov_xy) / (var_x * var_y).replace(0.0, np.nan)
    r2_30m = r2_30m.replace([np.inf, -np.inf], np.nan)
    out["iv_f_r2_30m"] = r2_30m

    if float(VCR_MIN_R2) > 0.0:
        valid_mask = r2_30m >= float(VCR_MIN_R2)
        beta_30m = beta_30m.where(valid_mask)
    out["iv_f_beta_30m"] = beta_30m.replace([np.inf, -np.inf], np.nan)

    out["fair_dIV_5s"] = out["iv_f_beta_30m"] * out["fut_dF_5s"]
    out["vcr_divergence"] = out["iv_dIV_5s"] - out["fair_dIV_5s"]

    div_roll_std = out["vcr_divergence"].rolling(win30m, min_periods=max(30, win5)).std()
    out["vcr_divergence_z"] = out["vcr_divergence"] / div_roll_std.replace(0.0, np.nan)

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


def _factor_vcr_divergence(df: pd.DataFrame) -> pd.Series:
    return df.get("vcr_divergence", pd.Series(index=df.index, dtype=float))


def _factor_iv_f_r2_30m(df: pd.DataFrame) -> pd.Series:
    return df.get("iv_f_r2_30m", pd.Series(index=df.index, dtype=float))


def _factor_iv_chg_20s_abs(df: pd.DataFrame) -> pd.Series:
    return df.get("iv_chg_20s_abs", pd.Series(index=df.index, dtype=float))


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
    "vcr_divergence": FactorDef(
        factor_id="vcr_divergence",
        label="vcr_divergence",
        description=f"30分钟回归得到 fair_dIV 后，real_dIV - fair_dIV（仅 r2>={VCR_MIN_R2:.2f} 生效）",
        compute=_factor_vcr_divergence,
        default_q=0.90,
        default_op=">=",
    ),
    "iv_f_r2_30m": FactorDef(
        factor_id="iv_f_r2_30m",
        label="iv_f_r2_30m",
        description="30分钟滚动回归 r2（dIV~dF）",
        compute=_factor_iv_f_r2_30m,
        default_q=0.80,
        default_op=">=",
    ),
    "iv_chg_20s_abs": FactorDef(
        factor_id="iv_chg_20s_abs",
        label="abs(iv_chg_20s)",
        description="20秒IV变动绝对值",
        compute=_factor_iv_chg_20s_abs,
        default_q=0.90,
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
