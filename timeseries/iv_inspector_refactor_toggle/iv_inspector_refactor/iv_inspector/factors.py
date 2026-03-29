# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from iv_inspector.feature_store import build_factor_material_frame

VCR_MIN_R2 = 0.20  # r^2 gate for vcr_divergence; <=0 means no gate


@dataclass(frozen=True)
class FactorDef:
    factor_id: str
    label: str
    description: str
    compute: Callable[[pd.DataFrame], pd.Series]
    default_q: float = 0.95
    default_op: str = ">="  # ">=" | "<="


def build_factor_base_frame(df_raw: pd.DataFrame, symbol: str, base_rule: str) -> pd.DataFrame:
    out = build_factor_material_frame(df_raw, symbol, base_rule, vcr_min_r2=VCR_MIN_R2)
    if out.empty:
        return out
    return out.set_index("dt_exch", drop=True)


def _factor_no_trade_move_60(df: pd.DataFrame) -> pd.Series:
    return df.get("no_trade_move_60", pd.Series(index=df.index, dtype=float))


def _factor_vcr_divergence(df: pd.DataFrame) -> pd.Series:
    return df.get("vcr_divergence", pd.Series(index=df.index, dtype=float))


def _factor_iv_f_r2_30m(df: pd.DataFrame) -> pd.Series:
    return df.get("iv_f_r2_30m", pd.Series(index=df.index, dtype=float))


def _factor_vcr_divergence_reversal(df: pd.DataFrame) -> pd.Series:
    return df.get("vcr_divergence_reversal", pd.Series(index=df.index, dtype=float))


FACTOR_REGISTRY: Dict[str, FactorDef] = {
    "no_trade_move_60": FactorDef(
        factor_id="no_trade_move_60",
        label="no_trade_move_60",
        description="60秒无量波动指标",
        compute=_factor_no_trade_move_60,
        default_q=0.80,
        default_op=">=",
    ),
    "vcr_divergence": FactorDef(
        factor_id="vcr_divergence",
        label="vcr_divergence",
        description=f"30分钟回归得到 fair_dIV 后，real_dIV - fair_dIV，仅 r2>={VCR_MIN_R2:.2f} 生效",
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
    "vcr_divergence_reversal": FactorDef(
        factor_id="vcr_divergence_reversal",
        label="vcr_divergence_reversal",
        description="vcr_divergence with fut reversal filter: (Ft-Ft-10s)*(Ft-10s-Ft-10min)<0",
        compute=_factor_vcr_divergence_reversal,
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
