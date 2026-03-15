# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
FACTOR_DATA_DIR = REPO_ROOT / "data" / "factors"


def _normalize_base_rule(base_rule: str) -> str:
    return str(base_rule).strip().lower()


def _rolling_bins_for_seconds(base_rule: str, sec: int) -> int:
    try:
        step = pd.Timedelta(_normalize_base_rule(base_rule)).total_seconds()
    except Exception:
        step = 1.0
    step = max(step, 1.0)
    return max(1, int(round(sec / step)))


def _normalize_data_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def factor_store_dir_for_data(data_path: str | Path) -> Path:
    src = _normalize_data_path(data_path)
    return FACTOR_DATA_DIR / src.stem


def factor_store_path(data_path: str | Path, base_rule: str) -> Path:
    return factor_store_dir_for_data(data_path) / f"factor_materials_{_normalize_base_rule(base_rule)}.parquet"


def build_factor_material_frame(df_raw: pd.DataFrame, symbol: str, base_rule: str, *, vcr_min_r2: float = 0.20) -> pd.DataFrame:
    base_rule = _normalize_base_rule(base_rule)
    dfo = df_raw.copy()
    dfo["dt_exch"] = pd.to_datetime(dfo["dt_exch"], errors="coerce")
    dfo = dfo.dropna(subset=["dt_exch"]).sort_values("dt_exch")

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
    win10m = _rolling_bins_for_seconds(base_rule, 10 * 60)
    win20 = _rolling_bins_for_seconds(base_rule, 20)
    win60 = _rolling_bins_for_seconds(base_rule, 60)
    win30m = _rolling_bins_for_seconds(base_rule, 30 * 60)

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

    x = out.get("fut_dF_5s", pd.Series(np.nan, index=out.index)).astype(float)
    y = out.get("iv_dIV_5s", pd.Series(np.nan, index=out.index)).astype(float)
    cov_xy = y.rolling(win30m, min_periods=max(30, win5)).cov(x)
    var_x = x.rolling(win30m, min_periods=max(30, win5)).var()
    var_y = y.rolling(win30m, min_periods=max(30, win5)).var()
    beta_30m = cov_xy / var_x.replace(0.0, np.nan)
    r2_30m = (cov_xy * cov_xy) / (var_x * var_y).replace(0.0, np.nan)
    r2_30m = r2_30m.replace([np.inf, -np.inf], np.nan)
    out["iv_f_r2_30m"] = r2_30m

    if float(vcr_min_r2) > 0.0:
        valid_mask = r2_30m >= float(vcr_min_r2)
        beta_30m = beta_30m.where(valid_mask)
    out["iv_f_beta_30m"] = beta_30m.replace([np.inf, -np.inf], np.nan)

    out["fair_dIV_5s"] = out["iv_f_beta_30m"] * out["fut_dF_5s"]
    out["vcr_divergence"] = out["iv_dIV_5s"] - out["fair_dIV_5s"]
    out["fut_dF_10s"] = out.get("fut_price", pd.Series(np.nan, index=out.index)).diff(win10)
    out["fut_dF_10m_from_10s"] = (
        out.get("fut_price", pd.Series(np.nan, index=out.index)).shift(win10)
        - out.get("fut_price", pd.Series(np.nan, index=out.index)).shift(win10m)
    )
    out["fut_reversal_10s_10m"] = (out["fut_dF_10s"] * out["fut_dF_10m_from_10s"]) < 0
    out["vcr_divergence_reversal"] = out["vcr_divergence"].where(out["fut_reversal_10s_10m"])

    div_roll_std = out["vcr_divergence"].rolling(win30m, min_periods=max(30, win5)).std()
    out["vcr_divergence_z"] = out["vcr_divergence"] / div_roll_std.replace(0.0, np.nan)

    out["abs_traded_vega"] = out.get("traded_vega", 0.0).abs()
    out["opt_spread_60s"] = out.get("opt_spread", np.nan).rolling(win60).median()
    out["no_trade_move_60"] = out["fut_ret_60s"].abs() / (out["fut_dvol_60s"] + 1.0)
    out["iv_f_div_60"] = out["iv_chg_60s"] * np.sign(out["fut_ret_60s"].fillna(0))

    out = out.reset_index().rename(columns={"index": "dt_exch"})
    out.insert(0, "symbol", str(symbol))
    return out


def precompute_factor_materials(df_raw: pd.DataFrame, base_rule: str, *, data_path: str | Path, symbols: Optional[list[str]] = None, vcr_min_r2: float = 0.20) -> Path:
    if df_raw is None or df_raw.empty:
        raise ValueError("df_raw is empty")

    dfo = df_raw.copy()
    if "symbol" not in dfo.columns:
        raise ValueError("df_raw missing symbol column")

    if "is_option" in dfo.columns:
        option_symbols = sorted(dfo.loc[dfo["is_option"] == True, "symbol"].dropna().astype(str).unique().tolist())
    else:
        option_symbols = sorted(dfo["symbol"].dropna().astype(str).unique().tolist())

    if symbols:
        allow = {str(s) for s in symbols}
        option_symbols = [s for s in option_symbols if s in allow]

    mats = []
    for sym in option_symbols:
        subset = dfo[(dfo["symbol"].astype(str) == sym) | (dfo.get("is_future", False) == True)].copy()
        mat = build_factor_material_frame(subset, sym, base_rule, vcr_min_r2=vcr_min_r2)
        if not mat.empty:
            mats.append(mat)

    if not mats:
        raise ValueError("no factor materials generated")

    out = pd.concat(mats, ignore_index=True, sort=False)
    out["dt_exch"] = pd.to_datetime(out["dt_exch"], errors="coerce")
    out = out.sort_values(["symbol", "dt_exch"]).reset_index(drop=True)

    out_path = factor_store_path(data_path, base_rule)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path


def load_factor_materials(data_path: str | Path, base_rule: str) -> pd.DataFrame:
    fp = factor_store_path(data_path, base_rule)
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    if "dt_exch" in df.columns:
        df["dt_exch"] = pd.to_datetime(df["dt_exch"], errors="coerce")
    return df


def get_symbol_factor_material(materials_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if materials_df is None or materials_df.empty or "symbol" not in materials_df.columns:
        return pd.DataFrame()
    out = materials_df[materials_df["symbol"].astype(str) == str(symbol)].copy()
    if "dt_exch" in out.columns:
        out = out.sort_values("dt_exch").set_index("dt_exch", drop=True)
    return out
