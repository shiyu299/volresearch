# -*- coding: utf-8 -*-
"""
Aggregation logic for IV index series (ATM±n) with optional state-adjust filling.

Key rules (per user spec):
- Candidate pool is determined by strikes near F_now; ATM strike must include BOTH call & put if exist.
- Candidate pool refreshes every `pool_refresh_seconds` (default 120s) and is NOT expanded to compensate for missing IV.
- Within the pool, contracts with IV missing at a time point are simply excluded from weights (n_used < n_target).
- If a base time bucket has no usable contracts, the IV index is filled forward; if the series starts with NaN, it is backfilled
  with the first available value (so K-line always drawable once any value appears in-range).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def colors_from_open_close(ohlc_or_open, close: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Backward/forward compatible:
    - colors_from_open_close(ohlc_df)
    - colors_from_open_close(open_array, close_array)
    Returns array of 0/1 where 1 means close>=open.
    """
    if close is None:
        ohlc = ohlc_or_open
        o = np.asarray(ohlc["open"].values)
        c = np.asarray(ohlc["close"].values)
    else:
        o = np.asarray(ohlc_or_open)
        c = np.asarray(close)
    return (c >= o).astype(int)


def make_ohlc(ser: pd.Series, rule: str) -> pd.DataFrame:
    ohlc = ser.resample(rule).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    return ohlc


@dataclass
class _State:
    iv: float
    delta: float
    vega: float
    f_quote: float


def _required_cols_ok(df: pd.DataFrame) -> bool:
    need = {"dt_exch", "symbol", "K", "F_used"}
    return need.issubset(df.columns)


def _pick_pool_symbols(meta: pd.DataFrame, f_now: float, n: int, otm_atm_only: bool) -> List[str]:
    """
    Pick a fixed pool of symbols given current f_now and metadata with columns:
    ['symbol','cp','K'].
    ATM strike is the strike closest to f_now; include all symbols on that strike (call+put).
    Remaining are added by increasing abs(K-f_now). If otm_atm_only, restrict non-ATM to OTM side.
    Pool size target is n, but may be < n if not enough symbols in meta.
    """
    if meta.empty or not np.isfinite(f_now):
        return []
    ks = meta["K"].values.astype(float)
    # ATM strike
    atm_k = float(meta.iloc[np.argmin(np.abs(ks - f_now))]["K"])
    atm_rows = meta[meta["K"].astype(float) == atm_k]
    picked = list(atm_rows["symbol"].unique())

    if len(picked) >= n:
        return picked[:n]

    rest = meta[~meta["symbol"].isin(picked)].copy()
    if otm_atm_only and "cp" in rest.columns:
        # only apply OTM filter to non-ATM
        is_call = rest["cp"].astype(str).str.upper().str.startswith("C")
        is_put = rest["cp"].astype(str).str.upper().str.startswith("P")
        k = rest["K"].astype(float)
        rest = rest[((is_call) & (k >= f_now)) | ((is_put) & (k <= f_now))]

    rest["dist"] = (rest["K"].astype(float) - f_now).abs()
    rest = rest.sort_values(["dist", "K"])
    for sym in rest["symbol"].tolist():
        if sym not in picked:
            picked.append(sym)
        if len(picked) >= n:
            break
    return picked


def _state_adjust(last: _State, f_now: float, threshold: float) -> float:
    dF = float(f_now) - float(last.f_quote)
    if abs(dF) <= float(threshold):
        return float(last.iv)
    vega = float(last.vega)
    if not np.isfinite(vega) or vega <= 0:
        return float(last.iv)
    return float(last.iv) - (float(last.delta) / vega) * dF


def _session_code(ts: pd.Timestamp) -> str:
    hm = ts.hour * 60 + ts.minute
    if 9 * 60 <= hm <= 11 * 60 + 30:
        return "DAY_AM"
    if 13 * 60 + 30 <= hm <= 15 * 60:
        return "DAY_PM"
    if 21 * 60 <= hm <= 23 * 60:
        return "NIGHT"
    return "OFF"


def make_iv_and_bar_series(
    df: pd.DataFrame,
    base_rule: str,
    n: int,
    otm_atm_only: bool,
    use_vega_weight: bool,
    use_abs_bar: bool,
    *,
    iv_fill_mode: str = "state_adjust",  # "state_adjust" | "ffill" | "quote_only"
    fut_move_threshold: float = 0.0,
    pool_refresh_seconds: int = 120,
    pool_refresh_fut_move: float = 0.0,
    min_valid_n: int = 4,
    is_ultra: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns (iv_index_series, bar_series, debug_df, details_df_or_None)
    """
    if df is None or df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), None
    if not _required_cols_ok(df):
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), None

    dfx = df.copy()
    dfx["dt_exch"] = pd.to_datetime(dfx["dt_exch"])
    dfx = dfx.sort_values("dt_exch")

    # metadata
    meta_cols = ["symbol", "K"]
    if "cp" in dfx.columns:
        meta_cols.append("cp")
    meta = dfx[meta_cols].drop_duplicates(subset=["symbol"]).copy()

    # base times
    t0 = dfx["dt_exch"].min()
    t1 = dfx["dt_exch"].max()
    base_times = pd.date_range(t0.floor(base_rule), t1.ceil(base_rule), freq=base_rule)
    if len(base_times) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), None

    # F_now per base bucket (use last observed F_used in bucket, then ffill)
    f_series = (
        dfx[["dt_exch", "F_used"]]
        .dropna()
        .set_index("dt_exch")["F_used"]
        .sort_index()
        .resample(base_rule)
        .last()
        .reindex(base_times)
        .ffill()
    )

    # bar series (default traded_vega)
    bar_col = "traded_vega" if "traded_vega" in dfx.columns else None
    if bar_col is None:
        bar_ser = pd.Series(0.0, index=base_times)
    else:
        b = (
            dfx[["dt_exch", bar_col]]
            .set_index("dt_exch")[bar_col]
            .resample(base_rule)
            .sum()
            .reindex(base_times)
            .fillna(0.0)
        )
        if use_abs_bar:
            b = b.abs()
        bar_ser = b

    # Pre-group quotes by base bucket: take last row per symbol within bucket
    dfx["_bucket"] = dfx["dt_exch"].dt.floor(base_rule)
    # Ensure IV exists column
    if "iv" not in dfx.columns:
        return pd.Series(dtype=float), bar_ser, pd.DataFrame(), None

    # We'll keep last quote per (bucket, symbol)
    last_quote = (
        dfx.sort_values("dt_exch")
        .groupby(["_bucket", "symbol"], as_index=False)
        .tail(1)
        .set_index(["_bucket", "symbol"])
    )

    state: Dict[str, _State] = {}
    iv_index_raw = []
    dbg_rows = []
    details_rows = [] if not is_ultra else None  # avoid memory in ultra mode

    pool_refresh_seconds = int(pool_refresh_seconds) if pool_refresh_seconds is not None else 0
    pool_refresh_td = pd.Timedelta(seconds=max(pool_refresh_seconds, 0))
    next_refresh_time = None
    last_pool_f = np.nan
    pool_syms: List[str] = []
    prev_bt = None
    prev_sess = None

    for bt in base_times:
        # 跨会话或超长间隔时，不继承上一段状态（避免 23:00->09:00 异常续推）
        cur_sess = _session_code(pd.Timestamp(bt))
        if (prev_bt is not None):
            gap_sec = (pd.Timestamp(bt) - pd.Timestamp(prev_bt)).total_seconds()
            if (cur_sess != prev_sess) or (gap_sec > max(120.0, pool_refresh_seconds + 5.0)):
                state.clear()
                pool_syms = []
                next_refresh_time = None
        prev_bt = bt
        prev_sess = cur_sess

        f_now = f_series.loc[bt]
        if not np.isfinite(f_now):
            iv_index_raw.append(np.nan)
            dbg_rows.append(
                dict(dt=bt, F_now=np.nan, iv_atm_n_raw=np.nan, iv_atm_n=np.nan,
                     n_target=n, n_used=0, sum_vega_used=0.0, min_iv_used=np.nan, max_iv_used=np.nan,
                     bar=float(bar_ser.loc[bt]), iv_fill_mode=iv_fill_mode, fut_move_threshold=float(fut_move_threshold))
            )
            continue

        refresh_by_time = (next_refresh_time is None) or (pool_refresh_seconds == 0) or (bt >= next_refresh_time)
        refresh_by_f_move = (
            np.isfinite(float(pool_refresh_fut_move))
            and float(pool_refresh_fut_move) > 0.0
            and np.isfinite(float(last_pool_f))
            and abs(float(f_now) - float(last_pool_f)) >= float(pool_refresh_fut_move)
        )
        if refresh_by_time or refresh_by_f_move:
            pool_syms = _pick_pool_symbols(meta, float(f_now), int(n), bool(otm_atm_only))
            next_refresh_time = bt + pool_refresh_td if pool_refresh_seconds > 0 else pd.Timestamp.max
            last_pool_f = float(f_now)

        used = []
        used_vega = []
        used_iv = []
        used_source = []
        used_df = []
        for sym in pool_syms:
            src = "none"
            iv_eff = np.nan
            # check quote this bucket
            key = (bt, sym)
            if key in last_quote.index:
                row = last_quote.loc[key]
                iv_q = row.get("iv", np.nan)
                # only update state when iv_q finite
                if pd.notna(iv_q) and np.isfinite(iv_q):
                    delta = float(row.get("delta", np.nan))
                    vega = float(row.get("vega", np.nan))
                    if np.isfinite(vega) and vega > 0:
                        state[sym] = _State(iv=float(iv_q), delta=delta, vega=vega, f_quote=float(f_now))
                        iv_eff = float(iv_q)
                        src = "quote"
            if not np.isfinite(iv_eff):
                if sym in state:
                    last = state[sym]
                    if iv_fill_mode == "quote_only":
                        iv_eff = np.nan
                        src = "none"
                    elif iv_fill_mode == "ffill":
                        iv_eff = float(last.iv)
                        src = "ffill"
                    else:  # state_adjust
                        iv_eff = _state_adjust(last, float(f_now), float(fut_move_threshold))
                        src = "state_adjust" if (abs(float(f_now) - float(last.f_quote)) > float(fut_move_threshold)) else "ffill"
                else:
                    iv_eff = np.nan
                    src = "none"

            if np.isfinite(iv_eff) and iv_eff > 0:
                vega = state[sym].vega if sym in state else np.nan
                if np.isfinite(vega) and vega > 0:
                    used.append(sym)
                    used_iv.append(iv_eff)
                    used_vega.append(float(vega))
                    used_source.append(src)

        min_contracts_required = max(1, int(min_valid_n))
        if len(used) < min_contracts_required:
            iv_idx = np.nan
            sumw = float(np.sum(used_vega) if len(used_vega) else 0.0)
            miniv = float(np.min(used_iv)) if len(used_iv) else np.nan
            maxiv = float(np.max(used_iv)) if len(used_iv) else np.nan
        else:
            w = np.array(used_vega, dtype=float) if use_vega_weight else np.ones(len(used), dtype=float)
            sumw = float(np.sum(w))
            if sumw <= 0:
                iv_idx = np.nan
            else:
                iv_idx = float(np.sum(w * np.array(used_iv, dtype=float)) / sumw)
            miniv = float(np.min(used_iv))
            maxiv = float(np.max(used_iv))

        iv_index_raw.append(iv_idx)

        dbg_rows.append(
            dict(
                dt=bt,
                F_now=float(f_now),
                iv_atm_n_raw=iv_idx,
                n_target=int(n),
                n_used=int(len(used)),
                sum_vega_used=float(np.sum(used_vega) if len(used_vega) else 0.0),
                min_iv_used=miniv,
                max_iv_used=maxiv,
                bar=float(bar_ser.loc[bt]),
                iv_fill_mode=iv_fill_mode,
                fut_move_threshold=float(fut_move_threshold),
                pool_refresh_seconds=int(pool_refresh_seconds),
                pool_refresh_fut_move=float(pool_refresh_fut_move),
                min_valid_n=int(min_contracts_required),
            )
        )

        if details_rows is not None:
            for sym, ivv, veg, src in zip(used, used_iv, used_vega, used_source):
                details_rows.append(
                    dict(dt=bt, symbol=sym, iv_eff=float(ivv), vega=float(veg), source=src, F_now=float(f_now))
                )

    ser_raw = pd.Series(iv_index_raw, index=base_times, name="iv_atm_n_raw")
    # Fill rule for plotting K-line IV: only forward-fill inside active timeline.
    # Do NOT backfill at the head, otherwise pre-first-quote bars become artificial.
    ser = ser_raw.copy()
    ser = ser.ffill()
    ser.name = "iv_atm_n"

    debug_df = pd.DataFrame(dbg_rows)
    debug_df["iv_atm_n"] = ser.values
    details_df = pd.DataFrame(details_rows) if details_rows is not None else None

    return ser, bar_ser, debug_df, details_df
