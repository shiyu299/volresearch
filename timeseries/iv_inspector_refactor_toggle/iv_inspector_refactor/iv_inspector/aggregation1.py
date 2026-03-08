# -*- coding: utf-8 -*-
"""IV Inspector - aggregation (Python 3.9 compatible)

Build per-`base_rule` IV-index series (ATM±n) and traded-vega bars.

Return:
    ser        : pd.Series (index=dt) IV-index at base buckets
    bar_ser    : pd.Series (index=dt) bar value at base buckets
    debug_df   : pd.DataFrame (index=dt) per bucket diagnostics
    details_df : pd.DataFrame columns per-picked-contract diagnostics

Modes (`iv_fill_mode`) for contracts w/o a new quote in current base bucket:
- "quote_only": only use contracts whose IV is quoted in this base bucket
- "ffill_iv"  : forward fill last valid IV
- "state_adjust": ffill last IV; if |dF| > threshold then
      dIV ≈ -(Delta/Vega) * dF  (assuming dP = 0)

Project convention:
- parquet has `vega` in per 1.0 vol units (iv is decimal, e.g. 0.23)
- `vega_1pct` if present is vega/100 (per 1% vol)

"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .selection import pick_atm_n_options


def _vega_col(df: pd.DataFrame) -> Optional[str]:
    """Prefer `vega` for this project; fall back to `vega_1pct` if needed."""
    if "vega" in df.columns:
        return "vega"
    if "vega_1pct" in df.columns:
        return "vega_1pct"
    return None


def agg_iv(picked: pd.DataFrame, use_vega_weight: bool, vcol: Optional[str]) -> float:
    """Aggregate IV across picked contracts."""
    d = picked.dropna(subset=["iv"]).copy()
    if d.empty:
        return np.nan

    if use_vega_weight and (vcol is not None) and (vcol in d.columns):
        w = d[vcol].astype(float).clip(lower=0).fillna(0.0).values
        x = d["iv"].astype(float).values
        s = float(np.sum(w))
        if s > 0:
            return float(np.sum(w * x) / s)

    return float(d["iv"].astype(float).mean())


def _state_adjust_iv(iv_last: float, delta: float, vega: float, dF: float, vega_is_1pct: bool) -> float:
    """Adjust IV under dP≈0: dIV ≈ -(Delta/Vega) * dF.

    If vega is per 1% vol (0.01), convert to per 1.0 vol by *100.
    """
    if (not np.isfinite(iv_last)) or (not np.isfinite(delta)) or (not np.isfinite(vega)) or (vega <= 0) or (not np.isfinite(dF)):
        return iv_last

    vega_eff = float(vega) * (100.0 if vega_is_1pct else 1.0)
    if (not np.isfinite(vega_eff)) or (vega_eff <= 0):
        return iv_last

    return float(iv_last - (float(delta) / vega_eff) * float(dF))


def make_iv_and_bar_series(
    dfx: pd.DataFrame,
    base_rule: str,
    n: int,
    otm_atm_only: bool,
    use_vega_weight: bool,
    use_abs_bar: bool,
    iv_fill_mode: str = "state_adjust",
    fut_move_threshold: float = 0.0,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build per-base_rule IV-index series and bar series.

    Parameters
    ----------
    dfx: DataFrame indexed by dt_exch
    base_rule: e.g. '1S'
    n: number of nearest strikes around ATM to use
    otm_atm_only: if True filter ITM
    use_vega_weight: if True vega-weighted avg, else simple mean
    use_abs_bar: if True use traded_vega else traded_vega_signed
    iv_fill_mode: 'state_adjust' | 'ffill_iv' | 'quote_only'
    fut_move_threshold: in state_adjust mode, if |dF| <= threshold then do not adjust
    """

    if dfx is None or dfx.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    required = {"F_used", "K", "iv", "symbol"}
    missing = [c for c in required if c not in dfx.columns]
    if missing:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    vcol = _vega_col(dfx)
    vega_is_1pct = (vcol == "vega_1pct")
    has_delta = ("delta" in dfx.columns)

    # sort by time
    dfx = dfx.sort_index()

    # per symbol state
    # NOTE: we only update iv/delta/vega/F_quote when current row has valid iv
    state = {}

    buckets_out = []
    debug_out = []
    details_out = []

    for t, bucket in dfx.groupby(pd.Grouper(freq=base_rule)):
        if bucket.empty:
            continue

        # current future price: use last non-null in bucket
        F_now_s = bucket["F_used"].dropna()
        if F_now_s.empty:
            continue
        F_now = float(F_now_s.iloc[-1])
        if (not np.isfinite(F_now)) or (F_now <= 0):
            continue

        # update state with last row per symbol in bucket
        last_rows = bucket.dropna(subset=["symbol"]).groupby("symbol", as_index=False).tail(1)
        for _, r in last_rows.iterrows():
            sym = str(r["symbol"])
            st = state.get(sym, {})
            st["symbol"] = sym

            # strike/cp can update even if iv missing
            if np.isfinite(r.get("K", np.nan)):
                st["K"] = float(r["K"])
            if "cp" in bucket.columns and pd.notna(r.get("cp", np.nan)):
                st["cp"] = r.get("cp")

            # only update iv-related fields if iv is valid
            iv_now = r.get("iv", np.nan)
            if np.isfinite(iv_now):
                st["iv_last"] = float(iv_now)
                st["iv_time"] = t

                if has_delta and np.isfinite(r.get("delta", np.nan)):
                    st["delta_last"] = float(r["delta"])
                if (vcol is not None) and np.isfinite(r.get(vcol, np.nan)):
                    st["vega_last"] = float(r[vcol])
                if np.isfinite(r.get("F_used", np.nan)):
                    st["F_quote"] = float(r["F_used"])

            state[sym] = st

        if not state:
            continue

        # snapshot from state
        rows = []
        for sym, st in state.items():
            K = st.get("K", np.nan)
            iv_last = st.get("iv_last", np.nan)
            if not (np.isfinite(K) and np.isfinite(iv_last)):
                # if never had valid iv, skip
                continue

            iv_eff = float(iv_last)
            source = "ffill"
            dF = np.nan

            if iv_fill_mode == "quote_only":
                # only keep if quoted in this base bucket
                if st.get("iv_time", None) != t:
                    iv_eff = np.nan
                else:
                    source = "quote"

            elif iv_fill_mode == "ffill_iv":
                # keep last iv as-is
                source = "ffill"

            else:  # state_adjust (default)
                F_quote = st.get("F_quote", np.nan)
                dF = (F_now - float(F_quote)) if np.isfinite(F_quote) else np.nan

                if np.isfinite(dF) and (abs(dF) > float(fut_move_threshold)):
                    # adjust
                    delta = st.get("delta_last", np.nan) if has_delta else np.nan
                    vega = st.get("vega_last", np.nan) if (vcol is not None) else np.nan
                    iv_eff = _state_adjust_iv(iv_last, delta, vega, dF, vega_is_1pct=vega_is_1pct)
                    source = "state_adjust"
                else:
                    source = "ffill"

            # guardrails
            if (not np.isfinite(iv_eff)) or (iv_eff <= 0) or (iv_eff > 5):
                iv_eff = np.nan

            row = {
                "symbol": sym,
                "K": float(K),
                "F_used": float(F_now),
                "iv": iv_eff,
                "source": source,
                "dF": dF,
            }
            if "cp" in st:
                row["cp"] = st["cp"]
            if vcol is not None:
                row[vcol] = st.get("vega_last", np.nan)

            rows.append(row)

        snap = pd.DataFrame(rows)
        if snap.empty:
            continue

        # per your requirement: contracts with iv NaN do not participate
        snap_valid = snap.dropna(subset=["iv"]).copy()
        if snap_valid.empty:
            continue

        picked = pick_atm_n_options(snap_valid, n=n, only_otm_atm=otm_atm_only)
        if picked.empty:
            continue

        iv_bucket = agg_iv(picked, use_vega_weight=use_vega_weight, vcol=vcol)

        # weights for details
        if use_vega_weight and (vcol is not None) and (vcol in picked.columns):
            w_raw = picked[vcol].astype(float).clip(lower=0).fillna(0.0).values
            w_sum = float(np.sum(w_raw))
            w = (w_raw / w_sum) if w_sum > 0 else np.zeros_like(w_raw)
        else:
            used_mask = picked["iv"].notna().values
            cnt = int(np.sum(used_mask))
            w = np.zeros(len(picked), dtype=float)
            if cnt > 0:
                w[used_mask] = 1.0 / cnt

        picked_reset = picked.reset_index(drop=True)
        for i, r in picked_reset.iterrows():
            details_out.append({
                "dt": t,
                "symbol": r.get("symbol", None),
                "K": r.get("K", np.nan),
                "iv_eff": r.get("iv", np.nan),
                "vega": r.get(vcol, np.nan) if (vcol is not None and vcol in picked_reset.columns) else np.nan,
                "weight": float(w[i]) if i < len(w) else np.nan,
                "source": r.get("source", None),
                "dF": r.get("dF", np.nan),
            })

        # debug summary
        n_used = int(picked_reset["iv"].notna().sum())
        min_iv_used = float(np.nanmin(picked_reset["iv"].values)) if n_used > 0 else np.nan
        max_iv_used = float(np.nanmax(picked_reset["iv"].values)) if n_used > 0 else np.nan
        sum_vega_used = np.nan
        if vcol is not None and (vcol in picked_reset.columns):
            vv = picked_reset[vcol].astype(float).clip(lower=0).fillna(0.0).values
            sum_vega_used = float(np.sum(vv))

        debug_out.append({
            "dt": t,
            "F_now": F_now,
            "iv_atm_n": iv_bucket,
            "n_target": int(n),
            "n_used": n_used,
            "sum_vega_used": sum_vega_used,
            "min_iv_used": min_iv_used,
            "max_iv_used": max_iv_used,
            "iv_fill_mode": iv_fill_mode,
            "fut_move_threshold": float(fut_move_threshold),
        })

        # bar
        bar = np.nan
        if use_abs_bar and ("traded_vega" in bucket.columns):
            bar = float(np.nansum(bucket["traded_vega"].values))
        elif (not use_abs_bar) and ("traded_vega_signed" in bucket.columns):
            bar = float(np.nansum(bucket["traded_vega_signed"].values))

        buckets_out.append((t, iv_bucket, bar))

    if not buckets_out:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    idx = pd.DatetimeIndex([x[0] for x in buckets_out])
    ser = pd.Series([x[1] for x in buckets_out], index=idx, name="iv_atm_n").sort_index()
    bar_ser = pd.Series([x[2] for x in buckets_out], index=idx, name="bar").sort_index()

    dbg = pd.DataFrame(debug_out)
    if not dbg.empty:
        dbg["dt"] = pd.to_datetime(dbg["dt"])
        dbg = dbg.set_index("dt").sort_index()

    details_df = pd.DataFrame(details_out)
    if not details_df.empty:
        details_df["dt"] = pd.to_datetime(details_df["dt"])
        details_df = details_df.sort_values(["dt", "weight"], ascending=[True, False])

    return ser, bar_ser, dbg, details_df


def make_ohlc(ser: pd.Series, kline_rule: str) -> pd.DataFrame:
    ohlc = ser.resample(kline_rule).agg(open="first", high="max", low="min", close="last")
    return ohlc.dropna()


def colors_from_open_close(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    return np.where(close > open_, "red", np.where(close < open_, "green", "gray"))
