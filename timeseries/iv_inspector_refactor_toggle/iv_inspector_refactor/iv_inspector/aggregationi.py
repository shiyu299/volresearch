# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .selection import pick_atm_n_options


def _vega_col(df: pd.DataFrame) -> Optional[str]:
    """Prefer `vega` (per 1.0 vol). Fallback to `vega_1pct` (per 1% vol)."""
    if "vega" in df.columns:
        return "vega"
    if "vega_1pct" in df.columns:
        return "vega_1pct"
    return None


def agg_iv(picked: pd.DataFrame, use_vega_weight: bool) -> float:
    d = picked.dropna(subset=["iv"]).copy()
    if d.empty:
        return np.nan

    vcol = _vega_col(d)
    if use_vega_weight and (vcol is not None):
        w = d[vcol].clip(lower=0).fillna(0.0).values.astype(float)
        x = d["iv"].values.astype(float)
        s = float(np.sum(w))
        if s > 0:
            return float(np.sum(w * x) / s)
    return float(d["iv"].mean())


def _state_adjust_iv(iv_last: float, delta: float, vega: float, dF: float, vega_is_1pct: bool) -> float:
    """dP ≈ Vega*dIV + Delta*dF, assume no quote update => dP≈0 => dIV ≈ -(Delta/Vega)*dF."""
    if (not np.isfinite(iv_last)) or (not np.isfinite(delta)) or (not np.isfinite(vega)) or (vega <= 0) or (not np.isfinite(dF)):
        return np.nan

    vega_eff = float(vega) * (100.0 if vega_is_1pct else 1.0)
    if (not np.isfinite(vega_eff)) or (vega_eff <= 0):
        return np.nan

    iv_new = float(iv_last - (delta / vega_eff) * dF)
    if (not np.isfinite(iv_new)) or (iv_new <= 0) or (iv_new > 5.0):
        return np.nan
    return iv_new


def make_iv_and_bar_series(
    dfx: pd.DataFrame,
    base_rule: str,
    n: int,
    otm_atm_only: bool,
    use_vega_weight: bool,
    use_abs_bar: bool,
    iv_fill_mode: str = "state_adjust",  # "state_adjust" | "ffill" | "quote_only"
    fut_move_threshold: float = 0.0,
    warmup_seconds: int = 60,
    output_start: Optional[pd.Timestamp] = None,
    output_end: Optional[pd.Timestamp] = None,
    fill_gaps_with_prev: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build base frequency IV index series + bar series.

    Key behaviors (aligned to your requirements):
    - Candidate set selection uses strike distance ONLY (do NOT expand outward due to iv NaN).
    - Weighting excludes iv NaN (do NOT count into numerator/denominator). So n_used may be < n.
    - Warmup: extend calc window backwards to seed last_iv, but only return [output_start, output_end).
    - Gap fill: if a base bucket produces all-NaN (e.g., lunch break), optionally forward-fill iv_index.
    """

    if dfx.empty:
        return (pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame())

    required = {"dt_exch", "symbol", "K", "F_used"}
    missing = [c for c in required if c not in dfx.columns]
    if missing:
        return (pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame())

    df = dfx.copy()
    df["dt_exch"] = pd.to_datetime(df["dt_exch"])

    # Output window
    if output_start is None:
        output_start = df["dt_exch"].min()
    else:
        output_start = pd.to_datetime(output_start)

    if output_end is None:
        output_end = df["dt_exch"].max() + pd.Timedelta(seconds=1)
    else:
        output_end = pd.to_datetime(output_end)

    # Warmup window for state seeding
    warmup = pd.Timedelta(seconds=int(max(0, warmup_seconds)))
    calc_start = output_start - warmup
    df = df[(df["dt_exch"] >= calc_start) & (df["dt_exch"] < output_end)].copy()
    if df.empty:
        return (pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame())

    vcol = _vega_col(df)
    vega_is_1pct = (vcol == "vega_1pct")
    has_delta = "delta" in df.columns
    has_cp = "cp" in df.columns

    # Sort so "last" rows per symbol are consistent
    df = df.sort_values(["dt_exch", "symbol"])

    # Drive the timeline by base_rule buckets
    t0 = output_start.floor(base_rule)
    t1 = (output_end.ceil(base_rule) if hasattr(output_end, "ceil") else output_end)
    t_index = pd.date_range(t0, t1, freq=base_rule)
    if len(t_index) == 0:
        return (pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame())

    # Pre-group by bucket for efficiency
    df["bucket"] = df["dt_exch"].dt.floor(base_rule)
    buckets = dict(tuple(df.groupby("bucket", sort=True)))

    # Per-symbol state
    state = {}
    debug_out = []
    details_out = []
    iv_out = []
    bar_out = []

    last_iv_index = np.nan

    for t in t_index:
        bucket = buckets.get(t, None)

        # F_now: last F_used in bucket; if missing, keep NaN
        F_now = np.nan
        if bucket is not None and (not bucket.empty):
            sF = bucket["F_used"].dropna()
            if not sF.empty:
                F_now = float(sF.iloc[-1])
        if not np.isfinite(F_now):
            iv_out.append((t, np.nan))
            bar_out.append((t, np.nan))
            continue

        # Update state with last row per symbol in this bucket
        if bucket is not None and (not bucket.empty):
            last_rows = bucket.sort_values("dt_exch").groupby("symbol", as_index=False).tail(1)
            for _, r in last_rows.iterrows():
                sym = r.get("symbol")
                if sym is None:
                    continue
                st = state.get(sym, {})
                # Always keep K / cp
                if np.isfinite(r.get("K", np.nan)):
                    st["K"] = float(r.get("K"))
                if has_cp:
                    st["cp"] = r.get("cp", st.get("cp", None))

                iv_now = r.get("iv", np.nan)
                if np.isfinite(iv_now):
                    st["iv_last"] = float(iv_now)
                    st["F_quote"] = float(r.get("F_used", F_now)) if np.isfinite(r.get("F_used", np.nan)) else float(F_now)
                    if has_delta and np.isfinite(r.get("delta", np.nan)):
                        st["delta_last"] = float(r.get("delta"))
                    if vcol is not None and np.isfinite(r.get(vcol, np.nan)):
                        st["vega_last"] = float(r.get(vcol))
                    st["iv_time"] = t
                state[sym] = st

        # Build snapshot from state
        rows = []
        for sym, st in state.items():
            K = st.get("K", np.nan)
            if not np.isfinite(K):
                continue

            # quoted in this bucket?
            quoted = False
            if bucket is not None and (not bucket.empty):
                tmp = bucket[bucket["symbol"] == sym]
                if not tmp.empty:
                    rr = tmp.sort_values("dt_exch").iloc[-1]
                    if np.isfinite(rr.get("iv", np.nan)):
                        quoted = True

            iv_eff = np.nan
            source = "none"
            dF = np.nan

            iv_last = st.get("iv_last", np.nan)
            if iv_fill_mode == "quote_only":
                if quoted and np.isfinite(iv_last):
                    iv_eff = float(iv_last)
                    source = "quote"
            elif iv_fill_mode == "ffill":
                if np.isfinite(iv_last):
                    iv_eff = float(iv_last)
                    source = "quote" if quoted else "ffill"
            else:  # state_adjust
                if quoted and np.isfinite(iv_last):
                    iv_eff = float(iv_last)
                    source = "quote"
                elif np.isfinite(iv_last):
                    F_quote = st.get("F_quote", np.nan)
                    dF = float(F_now - F_quote) if np.isfinite(F_quote) else np.nan
                    if np.isfinite(dF) and (abs(dF) > float(fut_move_threshold)):
                        delta = st.get("delta_last", np.nan) if has_delta else np.nan
                        vega = st.get("vega_last", np.nan) if vcol is not None else np.nan
                        iv_adj = _state_adjust_iv(float(iv_last), float(delta), float(vega), float(dF), vega_is_1pct)
                        if np.isfinite(iv_adj):
                            iv_eff = iv_adj
                            source = "state_adjust"
                        else:
                            iv_eff = np.nan
                            source = "none"
                    else:
                        iv_eff = float(iv_last)
                        source = "ffill"

            # guardrails
            if (not np.isfinite(iv_eff)) or (iv_eff <= 0) or (iv_eff > 5.0):
                iv_eff = np.nan

            row = {
                "symbol": sym,
                "K": float(K),
                "F_used": float(F_now),
                "iv": iv_eff,
                "source": source,
                "dF": dF,
            }
            if has_cp:
                row["cp"] = st.get("cp", None)
            if vcol is not None:
                row[vcol] = st.get("vega_last", np.nan)
            rows.append(row)

        snap = pd.DataFrame(rows)
        if snap.empty:
            iv_out.append((t, np.nan))
            bar_out.append((t, np.nan))
            continue

        # Selection uses strike distance only; do NOT filter by iv availability.
        picked = pick_atm_n_options(snap, n=int(n), only_otm_atm=otm_atm_only)
        picked_used = picked.dropna(subset=["iv"]).copy()
        iv_bucket_raw = agg_iv(picked_used, use_vega_weight=use_vega_weight) if (not picked_used.empty) else np.nan

        # Gap fill (lunch break etc.): if nothing usable, carry forward previous iv_index
        iv_bucket = iv_bucket_raw
        filled_from_prev = False
        if fill_gaps_with_prev and (not np.isfinite(iv_bucket)) and np.isfinite(last_iv_index):
            iv_bucket = float(last_iv_index)
            filled_from_prev = True

        if np.isfinite(iv_bucket):
            last_iv_index = float(iv_bucket)

        # bar
        bar = np.nan
        if bucket is not None and (not bucket.empty):
            if use_abs_bar and "traded_vega" in bucket.columns:
                bar = float(np.nansum(bucket["traded_vega"].values))
            elif (not use_abs_bar) and "traded_vega_signed" in bucket.columns:
                bar = float(np.nansum(bucket["traded_vega_signed"].values))

        # Debug metrics (based on used only)
        n_used = int(len(picked_used))
        sum_vega_used = np.nan
        min_iv_used = float(np.nanmin(picked_used["iv"].values)) if n_used > 0 else np.nan
        max_iv_used = float(np.nanmax(picked_used["iv"].values)) if n_used > 0 else np.nan
        if vcol is not None and (not picked_used.empty):
            vv = picked_used[vcol].clip(lower=0).fillna(0.0).astype(float).values
            sum_vega_used = float(np.sum(vv))

        debug_out.append(
            {
                "dt": t,
                "F_now": float(F_now),
                "iv_atm_n": iv_bucket,
                "iv_atm_n_raw": iv_bucket_raw,
                "filled_from_prev": filled_from_prev,
                "n_target": int(n),
                "n_used": n_used,
                "sum_vega_used": sum_vega_used,
                "min_iv_used": min_iv_used,
                "max_iv_used": max_iv_used,
                "bar": bar,
                "iv_fill_mode": iv_fill_mode,
                "fut_move_threshold": float(fut_move_threshold),
                "warmup_seconds": int(warmup_seconds),
            }
        )

        iv_out.append((t, iv_bucket))
        bar_out.append((t, bar))

        # Details: include picked set; weights only for used
        if not picked.empty:
            dd = picked.copy()
            dd["dt"] = t
            dd["picked"] = True
            dd["used"] = dd["iv"].notna()
            dd["weight"] = np.nan
            if use_vega_weight and (vcol is not None) and (not picked_used.empty):
                w = picked_used[vcol].clip(lower=0).fillna(0.0).astype(float).values
                s = float(np.sum(w))
                if s > 0:
                    weights = w / s
                    dd.loc[dd["used"], "weight"] = weights
            details_out.append(dd)

    ser = pd.Series({t: v for t, v in iv_out}).sort_index()
    bar_ser = pd.Series({t: v for t, v in bar_out}).sort_index()
    debug_df = pd.DataFrame(debug_out)
    details_df = pd.concat(details_out, ignore_index=True) if details_out else pd.DataFrame()

    # Trim to output window
    ser = ser[(ser.index >= output_start) & (ser.index < output_end)]
    bar_ser = bar_ser[(bar_ser.index >= output_start) & (bar_ser.index < output_end)]

    # Final continuity for plotting:
    # - ffill handles mid-session gaps (no quotes)
    # - bfill fills the beginning of the window if the first few points are NaN but later points exist
    if fill_gaps_with_prev and (len(ser) > 0):
        ser = ser.ffill()
        ser = ser.bfill()

    if not debug_df.empty:
        debug_df["dt"] = pd.to_datetime(debug_df["dt"])
        debug_df = debug_df[(debug_df["dt"] >= output_start) & (debug_df["dt"] < output_end)].reset_index(drop=True)

    if not details_df.empty:
        details_df["dt"] = pd.to_datetime(details_df["dt"])
        details_df = details_df[(details_df["dt"] >= output_start) & (details_df["dt"] < output_end)].reset_index(drop=True)

    # If there are stretches where all picked contracts have iv NaN (e.g. trading break / no updates),
    # user wants the iv-index to carry forward the last value.
    if ser is not None and (not ser.empty):
        ser = ser.ffill()

    # Keep debug raw vs filled for inspection
    if not debug_df.empty and ("iv_atm_n" in debug_df.columns):
        debug_df["iv_atm_n_raw"] = debug_df["iv_atm_n"]
        debug_df["iv_atm_n"] = pd.Series(debug_df["iv_atm_n"].values).ffill().values
        debug_df["filled_from_prev"] = debug_df["iv_atm_n_raw"].isna() & debug_df["iv_atm_n"].notna()

    return ser, bar_ser, debug_df, details_df


def make_ohlc(ser: pd.Series, rule: str) -> pd.DataFrame:
    if ser is None or ser.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    o = ser.resample(rule).first()
    h = ser.resample(rule).max()
    l = ser.resample(rule).min()
    c = ser.resample(rule).last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna(how="all")
    return out


def colors_from_open_close(df_ohlc: pd.DataFrame) -> list:
    cols = []
    for _, r in df_ohlc.iterrows():
        o = r.get("open", np.nan)
        c = r.get("close", np.nan)
        if np.isfinite(o) and np.isfinite(c) and c >= o:
            cols.append("green")
        else:
            cols.append("red")
    return cols