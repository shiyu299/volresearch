from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


SIMPLE_ROOT = Path(__file__).resolve().parent
SIMPLE_DATA_DIR = SIMPLE_ROOT / "data"

from simple_inspector.aggregation import make_iv_and_bar_series, make_ohlc
from simple_inspector.data import list_data_files, load_data
from simple_inspector.drilldown import render_drilldown_tabs, tables_to_excel_bytes
from simple_inspector.viz import build_fut_iv_vega_stack_figure


st.set_page_config(layout="wide")

APP_CACHE_DIR = SIMPLE_ROOT / "data" / "cache" / "iv_app_simple"


def _estimate_option_strike_step(df: pd.DataFrame) -> float:
    if df is None or df.empty or "K" not in df.columns:
        return np.nan
    dfo = df[df["is_option"] == True].copy() if "is_option" in df.columns else df.copy()
    if dfo.empty:
        return np.nan
    k = pd.to_numeric(dfo["K"], errors="coerce").dropna().unique()
    if len(k) < 2:
        return np.nan
    diffs = np.diff(np.sort(k.astype(float)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-12)]
    if len(diffs) == 0:
        return np.nan
    vc = pd.Series(np.round(diffs, 8)).value_counts()
    return float(vc.index[0]) if not vc.empty else np.nan


def apply_no_blank_time_axis(fig):
    fig.update_xaxes(type="category")
    return fig


def build_active_mask(raw_dt: pd.Series, target_index: pd.DatetimeIndex, max_stale_seconds: int = 120) -> pd.Series:
    evt = pd.to_datetime(raw_dt, errors="coerce").dropna().drop_duplicates().sort_values()
    if evt.empty or len(target_index) == 0:
        return pd.Series(False, index=target_index)
    left = pd.DataFrame({"t": pd.to_datetime(pd.Series(pd.DatetimeIndex(target_index).values.astype("datetime64[ns]")))})
    right = pd.DataFrame({"evt": pd.to_datetime(pd.Series(pd.DatetimeIndex(evt).values.astype("datetime64[ns]")))})
    m = pd.merge_asof(left.sort_values("t"), right.sort_values("evt"), left_on="t", right_on="evt", direction="backward")
    active = (m["t"] - m["evt"]).dt.total_seconds().le(max_stale_seconds).fillna(False)
    return pd.Series(active.values, index=target_index)


def is_in_trading_session(idx: pd.DatetimeIndex) -> pd.Series:
    t = pd.DatetimeIndex(idx)
    m = t.hour * 60 + t.minute
    return pd.Series(
        ((m >= 9 * 60) & (m <= 11 * 60 + 30))
        | ((m >= 13 * 60 + 30) & (m <= 15 * 60))
        | ((m >= 21 * 60) & (m <= 23 * 60)),
        index=t,
    )


def break_long_flat_segments(ser: pd.Series, max_stale_seconds: int = 120, eps: float = 1e-12) -> pd.Series:
    if ser is None or ser.empty:
        return ser
    s = ser.copy()
    idx = pd.DatetimeIndex(s.index)
    if len(idx) < 3:
        return s
    d = idx.to_series().diff().dt.total_seconds().dropna()
    d = d[d > 0]
    step = float(d.median()) if not d.empty else 1.0
    max_flat_bins = max(1, int(max_stale_seconds / max(step, 1.0)))
    flat = s.diff().abs().le(eps) | s.diff().isna()
    run_id = (flat != flat.shift(1)).cumsum()
    run_len = flat.groupby(run_id).cumcount() + 1
    s[flat & (run_len > max_flat_bins)] = pd.NA
    return s


def sidebar_params():
    with st.sidebar:
        st.markdown("### Parameters")
        files = list_data_files(str(SIMPLE_DATA_DIR / "derived"))
        fp = st.selectbox("Data file", files) if files else None

        strike_gap_suggest = np.nan
        if fp:
            fp_hint = fp if Path(fp).is_absolute() else str((SIMPLE_ROOT / fp).resolve())
            try:
                strike_gap_suggest = _estimate_option_strike_step(load_data(fp_hint))
            except Exception:
                strike_gap_suggest = np.nan

        base_rule = st.selectbox("Base Rule", ["250ms", "500ms", "1S", "2S", "5S"], index=2)
        kline_rule = st.selectbox("Kline Rule", ["2S", "5S", "10S", "30S", "1min", "5min"], index=4)
        n = int(st.slider("ATM nearby contracts", 2, 60, 6, 1))
        otm_atm_only = st.checkbox("OTM + ATM only", value=True)
        use_vega_weight = st.checkbox("Vega-weighted IV", value=True)

        iv_mode = st.radio("IV fill mode", ["state_adjust", "ffill", "quote_only"], index=0)
        fut_move_threshold = float(st.number_input("Future move threshold", value=0.5, step=0.1))
        pool_refresh_seconds = int(st.number_input("Pool refresh seconds", value=120, step=30, min_value=0))

        if "simple_pool_refresh_fut_move" not in st.session_state:
            st.session_state["simple_pool_refresh_fut_move"] = float(strike_gap_suggest) if np.isfinite(strike_gap_suggest) else 0.0
        pool_refresh_fut_move = float(
            st.number_input("Pool refresh future move", key="simple_pool_refresh_fut_move", step=0.1, min_value=0.0)
        )
        min_valid_n = int(st.number_input("Min valid contracts", value=6, step=1, min_value=1))
        use_abs_bar = st.checkbox("Absolute traded vega bars", value=False)
        submitted = st.button("Update")

    return {
        "fp": fp,
        "base_rule": base_rule,
        "kline_rule": kline_rule,
        "n": n,
        "otm_atm_only": otm_atm_only,
        "use_vega_weight": use_vega_weight,
        "iv_fill_mode": iv_mode,
        "fut_move_threshold": fut_move_threshold,
        "pool_refresh_seconds": pool_refresh_seconds,
        "pool_refresh_fut_move": pool_refresh_fut_move,
        "min_valid_n": min_valid_n,
        "use_abs_bar": use_abs_bar,
    }, submitted


def signature(params: dict):
    return (
        params["fp"],
        params["base_rule"],
        params["kline_rule"],
        params["n"],
        params["otm_atm_only"],
        params["use_vega_weight"],
        params["iv_fill_mode"],
        float(params["fut_move_threshold"]),
        int(params["pool_refresh_seconds"]),
        float(params["pool_refresh_fut_move"]),
        int(params["min_valid_n"]),
        params["use_abs_bar"],
    )


def _aggregation_cache_path(params: dict) -> Path:
    fp = params.get("fp") or "unknown"
    src = Path(fp)
    stem = src.stem or "unknown"
    src_abs = (SIMPLE_ROOT / src).resolve() if not src.is_absolute() else src.resolve()
    stat = src_abs.stat() if src_abs.exists() else None
    payload = {"sig": list(signature(params)), "src": str(src_abs), "mtime_ns": getattr(stat, "st_mtime_ns", None), "size": getattr(stat, "st_size", None)}
    key = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    out_dir = APP_CACHE_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"agg_{key}.pkl"


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        obj = pd.read_pickle(cache_path)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, cache_path)


def compute_if_needed(df_raw: pd.DataFrame, params: dict, submitted: bool, cache: dict):
    sig = signature(params)
    cache_path = _aggregation_cache_path(params)
    cache_has = ("sig" in cache) and ("ser" in cache) and ("bar_ser" in cache)
    need = (not cache_has) or submitted or (cache.get("sig") != sig and submitted)
    if not need:
        return

    disk_payload = _load_cache(cache_path)
    if disk_payload and disk_payload.get("sig") == sig:
        cache.update(disk_payload)
        return

    ser, bar_ser, debug_df, details_df = make_iv_and_bar_series(
        df_raw,
        params["base_rule"],
        params["n"],
        params["otm_atm_only"],
        params["use_vega_weight"],
        params["use_abs_bar"],
        iv_fill_mode=params["iv_fill_mode"],
        fut_move_threshold=float(params["fut_move_threshold"]),
        pool_refresh_seconds=int(params["pool_refresh_seconds"]),
        pool_refresh_fut_move=float(params["pool_refresh_fut_move"]),
        min_valid_n=int(params["min_valid_n"]),
        is_ultra=False,
    )
    if ser is None or ser.empty:
        cache.clear()
        return

    if "traded_vega_signed" in df_raw.columns:
        bar_signed = (
            df_raw[["dt_exch", "traded_vega_signed"]]
            .set_index("dt_exch")["traded_vega_signed"]
            .resample(params["base_rule"])
            .sum()
            .reindex(ser.index)
            .fillna(0.0)
        )
    else:
        bar_signed = pd.Series(0.0, index=ser.index)

    active_mask = build_active_mask(df_raw["dt_exch"], ser.index, max_stale_seconds=120)
    ser = ser.where(active_mask)
    bar_ser = bar_ser.where(active_mask, 0.0)
    sess_mask = is_in_trading_session(ser.index)
    ser = ser.where(sess_mask)
    bar_ser = bar_ser.where(sess_mask, 0.0)
    ser = break_long_flat_segments(ser, max_stale_seconds=120)

    ohlc = make_ohlc(ser, params["kline_rule"]).dropna(how="any")
    fut_ser = (
        df_raw[["dt_exch", "F_used"]]
        .dropna()
        .set_index("dt_exch")["F_used"]
        .sort_index()
        .resample(params["kline_rule"])
        .last()
        .reindex(ohlc.index)
    )
    fut_ser = break_long_flat_segments(fut_ser.where(ohlc["close"].notna()), max_stale_seconds=120)
    bar_agg = bar_ser.resample(params["kline_rule"]).sum().reindex(ohlc.index).fillna(0.0)
    bar_signed_agg = bar_signed.resample(params["kline_rule"]).sum().reindex(ohlc.index).fillna(0.0)

    payload = {
        "sig": sig,
        "ser": ser,
        "bar_ser": bar_ser,
        "bar_signed": bar_signed,
        "ohlc": ohlc,
        "fut_ser": fut_ser,
        "bar_agg": bar_agg,
        "bar_signed_agg": bar_signed_agg,
        "debug_df": debug_df,
        "details_df": details_df,
    }
    cache.update(payload)
    _save_cache(cache_path, payload)


def render_main_chart(cache: dict):
    fig = build_fut_iv_vega_stack_figure(cache["fut_ser"], cache["ohlc"], cache["bar_agg"], cache["bar_signed_agg"])
    fig = apply_no_blank_time_axis(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})


def render_single_contract_chart(df_raw: pd.DataFrame, params: dict, main_time_index: Optional[pd.DatetimeIndex] = None):
    st.markdown("### Single Contract IV / Vega / Volume")
    dfo = df_raw[df_raw["is_option"] == True].copy() if "is_option" in df_raw.columns else df_raw.copy()
    if dfo.empty or "symbol" not in dfo.columns or "iv" not in dfo.columns:
        return
    syms = sorted(dfo["symbol"].dropna().astype(str).unique().tolist())
    if not syms:
        return
    sym = st.selectbox("Single contract", syms, key="simple_single_contract")
    ds = dfo[dfo["symbol"].astype(str) == str(sym)].copy().sort_values("dt_exch")
    ds = ds[is_in_trading_session(pd.DatetimeIndex(ds["dt_exch"])).values]
    if ds.empty:
        return

    iv_ser = ds.set_index("dt_exch")["iv"].resample(params["base_rule"]).last().sort_index().ffill().bfill()
    iv_ser = break_long_flat_segments(iv_ser[iv_ser > 0], max_stale_seconds=120)
    ohlc = make_ohlc(iv_ser, params["kline_rule"]).dropna(how="any")
    if ohlc.empty:
        return

    if "traded_vega_signed" in ds.columns:
        vega_ser = ds.set_index("dt_exch")["traded_vega_signed"].resample(params["kline_rule"]).sum()
    elif "traded_vega" in ds.columns:
        vega_ser = ds.set_index("dt_exch")["traded_vega"].resample(params["kline_rule"]).sum()
    else:
        vega_ser = pd.Series(0.0, index=ohlc.index)

    if "trade_volume_lots" in ds.columns:
        vol_ser = ds.set_index("dt_exch")["trade_volume_lots"].resample(params["kline_rule"]).sum()
    elif "d_volume" in ds.columns:
        vol_ser = ds.set_index("dt_exch")["d_volume"].resample(params["kline_rule"]).sum()
    elif "volume" in ds.columns:
        vol_ser = ds.set_index("dt_exch")["volume"].resample(params["kline_rule"]).last().diff().fillna(0.0).clip(lower=0.0)
    else:
        vol_ser = pd.Series(0.0, index=ohlc.index)

    idx = pd.DatetimeIndex(main_time_index) if main_time_index is not None and len(main_time_index) > 0 else ohlc.index
    ohlc = ohlc.reindex(idx)
    vega_ser = vega_ser.reindex(idx).fillna(0.0)
    vol_ser = vol_ser.reindex(idx).fillna(0.0)

    valid_k = ohlc[["open", "close"]].notna().all(axis=1).values
    up = valid_k & (ohlc["close"].values >= ohlc["open"].values)
    vol_colors = ["red" if u else ("gray" if not vk else "green") for u, vk in zip(up, valid_k)]
    vega_colors = ["red" if v > 0 else "green" if v < 0 else "gray" for v in vega_ser.values]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.62, 0.19, 0.19])
    fig.add_trace(go.Candlestick(x=idx, open=ohlc["open"], high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], increasing_line_color="red", decreasing_line_color="green", name="IV"), row=1, col=1)
    fig.add_trace(go.Bar(x=idx, y=np.abs(vega_ser.values), marker_color=vega_colors, opacity=0.75, name="traded vega(abs)"), row=2, col=1)
    fig.add_trace(go.Bar(x=idx, y=vol_ser.values, marker_color=vol_colors, opacity=0.75, name="volume"), row=3, col=1)
    fig.update_layout(height=860, margin=dict(l=40, r=30, t=40, b=30), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig = apply_no_blank_time_axis(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})


def render_debug_table(cache: dict):
    debug_df = cache.get("debug_df")
    if debug_df is not None and not getattr(debug_df, "empty", True):
        st.markdown("### Debug")
        st.dataframe(debug_df, use_container_width=True, height=320)


def render_drilldown(df_raw: pd.DataFrame, params: dict):
    st.markdown("### Drilldown")
    if "dt_exch" not in df_raw.columns or df_raw.empty:
        return
    tmin = df_raw["dt_exch"].min()
    tmax = df_raw["dt_exch"].max()
    default_end = min(tmax, tmin + pd.Timedelta(minutes=2))
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        start = pd.Timestamp(st.text_input("Start", value=tmin.strftime("%Y-%m-%d %H:%M:%S")))
    with c2:
        end = pd.Timestamp(st.text_input("End", value=default_end.strftime("%Y-%m-%d %H:%M:%S")))
    with c3:
        quick_minutes = int(st.number_input("Quick minutes", min_value=0, value=0, step=1))
    if quick_minutes > 0:
        end = start + pd.Timedelta(minutes=quick_minutes)
    start = max(start, tmin)
    end = min(end, tmax)
    if end <= start:
        end = min(tmax, start + pd.Timedelta(minutes=1))
    df_interval = df_raw[(df_raw["dt_exch"] >= start) & (df_raw["dt_exch"] <= end)].copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        n_main = int(st.number_input("ATM n", min_value=1, value=int(params["n"]), step=1))
    with c2:
        n_alt = int(st.number_input("ATM alt", min_value=1, value=max(1, int(params["n"] // 2)), step=1))
    with c3:
        top_m = int(st.number_input("TopVol m", min_value=1, value=20, step=1))
    tables = render_drilldown_tabs(df_interval=df_interval, n_main=n_main, n_alt=n_alt, top_m=top_m, otm_atm_only=params["otm_atm_only"])
    if tables:
        st.download_button("Download drilldown Excel", data=tables_to_excel_bytes(tables), file_name=f"drilldown_{start:%Y%m%d_%H%M%S}_{end:%H%M%S}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def main():
    st.title("IV Inspector Simple")
    cache = st.session_state.setdefault("simple_cache", {})
    params, submitted = sidebar_params()
    if not params["fp"]:
        st.info("No data file found under data/derived")
        return
    fp = params["fp"]
    if fp and not Path(fp).is_absolute():
        fp = str((SIMPLE_ROOT / fp).resolve())
    df_raw = load_data(fp)
    if df_raw is None or df_raw.empty:
        st.warning("Empty data")
        return
    compute_if_needed(df_raw, params, submitted, cache)
    if ("ser" not in cache) or cache.get("ser") is None or cache.get("ser").empty:
        st.warning("No aggregated series generated")
        return
    render_main_chart(cache)
    render_single_contract_chart(df_raw, params, main_time_index=cache.get("ohlc", pd.DataFrame()).index)
    render_debug_table(cache)
    render_drilldown(df_raw, params)


if __name__ == "__main__":
    main()
