# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Optional
import hashlib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from iv_inspector.data import list_data_files, load_data
from iv_inspector.feature_store import factor_store_path, get_symbol_factor_material, load_factor_materials
from iv_inspector.aggregation import make_iv_and_bar_series, make_ohlc
from iv_inspector.viz import build_fut_iv_vega_stack_figure
from iv_inspector.drilldown import render_drilldown_tabs, tables_to_excel_bytes
from iv_inspector.factors import list_factors, build_factor_base_frame, evaluate_factor_trigger

st.set_page_config(layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_CACHE_DIR = REPO_ROOT / "data" / "cache" / "iv_app"


def _estimate_option_strike_step(df: pd.DataFrame) -> float:
    """
    Estimate consecutive strike gap from option rows.
    Return NaN if unavailable.
    """
    if df is None or df.empty or "K" not in df.columns:
        return np.nan

    dfo = df.copy()
    if "is_option" in dfo.columns:
        dfo = dfo[dfo["is_option"] == True]
    if dfo.empty:
        return np.nan

    k = pd.to_numeric(dfo["K"], errors="coerce").dropna().unique()
    if len(k) < 2:
        return np.nan
    k = np.sort(k.astype(float))
    diffs = np.diff(k)
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-12)]
    if diffs.size == 0:
        return np.nan

    # Use the most common positive strike gap as default.
    gap_counts = pd.Series(np.round(diffs, 8)).value_counts()
    if gap_counts.empty:
        return np.nan
    return float(gap_counts.index[0])


def _series_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q90": np.nan,
        }
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "q90": float(s.quantile(0.90)),
    }


def _get_precomputed_factor_materials(params: dict, cache: dict) -> pd.DataFrame:
    data_fp = params.get("fp")
    base_rule = params.get("base_rule")
    sig = (str(data_fp), str(base_rule))
    if cache.get("_factor_materials_sig") == sig:
        return cache.get("_factor_materials_df", pd.DataFrame())

    mats = load_factor_materials(data_fp, base_rule) if data_fp and base_rule else pd.DataFrame()
    cache["_factor_materials_sig"] = sig
    cache["_factor_materials_df"] = mats
    return mats


def _compute_ic_ir_from_series(factor_ser: pd.Series, iv_close_ser: pd.Series) -> dict:
    """
    IC: Spearman corr(factor_t, iv_ret_fwd1_t)
    IR: mean/std of 30-min IC series (same definition, computed per 30min window).
    """
    if factor_ser is None or iv_close_ser is None or len(factor_ser) == 0 or len(iv_close_ser) == 0:
        return {"ic": np.nan, "ir": np.nan, "n_obs": 0, "n_days": 0}

    x = pd.to_numeric(factor_ser, errors="coerce")
    y = pd.to_numeric(iv_close_ser, errors="coerce")
    idx = x.index.intersection(y.index)
    if len(idx) == 0:
        return {"ic": np.nan, "ir": np.nan, "n_obs": 0, "n_days": 0}

    x = x.reindex(idx)
    y = y.reindex(idx)
    y_fwd = y.pct_change().shift(-1)

    df = pd.DataFrame({"x": x, "y_fwd": y_fwd}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return {"ic": np.nan, "ir": np.nan, "n_obs": 0, "n_bins": 0}

    ic = float(df["x"].corr(df["y_fwd"], method="spearman"))

    dics = []
    for _, g in df.groupby(df.index.floor("30min")):
        if len(g) < 5:
            continue
        v = g["x"].corr(g["y_fwd"], method="spearman")
        if pd.notna(v):
            dics.append(float(v))
    if len(dics) >= 2:
        s = pd.Series(dics, dtype=float)
        sd = float(s.std(ddof=1))
        ir = float(s.mean() / sd) if sd > 0 else np.nan
    else:
        ir = np.nan

    return {"ic": ic, "ir": ir, "n_obs": int(len(df)), "n_bins": int(len(dics))}


def _safe_resample_signal(sig_raw: pd.Series, rule: str, target_idx: pd.DatetimeIndex) -> pd.Series:
    s = pd.to_numeric(sig_raw, errors="coerce")
    if not isinstance(getattr(s, "index", None), pd.DatetimeIndex):
        return pd.Series(np.nan, index=target_idx, dtype=float)
    if len(s) == 0:
        return pd.Series(np.nan, index=target_idx, dtype=float)
    s = s.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    if s.empty:
        return pd.Series(np.nan, index=target_idx, dtype=float)
    return s.resample(rule).last().reindex(target_idx)


def apply_no_blank_time_axis(fig):
    """Compress long no-trade/no-update gaps by rendering x-axis as category."""
    fig.update_xaxes(type="category")
    return fig


def build_active_mask(raw_dt: pd.Series, target_index: pd.DatetimeIndex, max_stale_seconds: int = 120) -> pd.Series:
    """Mark points as active only if latest market update is within max_stale_seconds."""
    evt = pd.to_datetime(raw_dt, errors="coerce").dropna().drop_duplicates().sort_values()
    if evt.empty or len(target_index) == 0:
        return pd.Series(False, index=target_index)

    # merge_asof 瑕佹眰宸﹀彸閿?dtype 瀹屽叏涓€鑷达紱缁熶竴鍒?datetime64[ns]
    t_idx = pd.to_datetime(pd.Series(pd.DatetimeIndex(target_index).values.astype("datetime64[ns]")))
    e_idx = pd.to_datetime(pd.Series(pd.DatetimeIndex(evt).values.astype("datetime64[ns]")))

    left = pd.DataFrame({"t": t_idx}).sort_values("t")
    right = pd.DataFrame({"evt": e_idx}).sort_values("evt")
    m = pd.merge_asof(left, right, left_on="t", right_on="evt", direction="backward")
    delta_sec = (m["t"] - m["evt"]).dt.total_seconds()
    active = delta_sec.le(max_stale_seconds).fillna(False)
    return pd.Series(active.values, index=target_index)


def is_in_trading_session(idx: pd.DatetimeIndex) -> pd.Series:
    """SHFE-like sessions: 09:00-11:30, 13:30-15:00, 21:00-23:00."""
    t = pd.DatetimeIndex(idx)
    m = t.hour * 60 + t.minute
    day_am = (m >= 9 * 60) & (m <= 11 * 60 + 30)
    day_pm = (m >= 13 * 60 + 30) & (m <= 15 * 60)
    night = (m >= 21 * 60) & (m <= 23 * 60)
    return pd.Series(day_am | day_pm | night, index=t)


def break_long_flat_segments(ser: pd.Series, max_stale_seconds: int = 120, eps: float = 1e-12) -> pd.Series:
    """Break long flat (unchanged) runs so chart doesn't draw long horizontal stale lines."""
    if ser is None or ser.empty:
        return ser

    s = ser.copy()
    idx = pd.DatetimeIndex(s.index)
    if len(idx) < 3:
        return s

    # infer step seconds from median positive diff
    d = idx.to_series().diff().dt.total_seconds().dropna()
    d = d[d > 0]
    step = float(d.median()) if not d.empty else 1.0
    if step <= 0:
        step = 1.0
    max_flat_bins = max(1, int(max_stale_seconds / step))

    # flat when value change is tiny
    delta = s.diff().abs()
    flat = delta.le(eps) | delta.isna()

    run_id = (flat != flat.shift(1)).cumsum()
    run_len = flat.groupby(run_id).cumcount() + 1

    # for flat runs, keep first max_flat_bins points then break (set NaN)
    to_break = flat & (run_len > max_flat_bins)
    s[to_break] = pd.NA
    return s


def sidebar_params():
    """Sidebar UI -> params dict, and a submit flag."""
    with st.sidebar:
        st.markdown("### 参数（点击“更新图表”后才会重算主图）")

        files = list_data_files()
        fp = st.selectbox("选择数据文件（data/derived）", files) if files else None

        strike_gap_suggest = np.nan
        if fp:
            fp_for_hint = fp
            if not Path(fp_for_hint).is_absolute():
                fp_for_hint = str((REPO_ROOT / fp_for_hint).resolve())
            try:
                df_hint = load_data(fp_for_hint)
                strike_gap_suggest = _estimate_option_strike_step(df_hint)
            except Exception:
                strike_gap_suggest = np.nan

        base_rule = st.selectbox("Base Rule", ["250ms", "500ms", "1S", "2S", "5S"], index=2)
        kline_rule = st.selectbox("Kline Rule", ["2S", "5S", "10S", "30S", "1min", "5min"], index=4)

        n = int(st.slider("ATM nearby contracts", 2, 60, 6, 1))
        otm_atm_only = st.checkbox("OTM + ATM only", value=True)
        use_vega_weight = st.checkbox("Vega-weighted IV", value=True)

        st.markdown("#### IV fill when no fresh quote")
        iv_mode_label = st.radio(
            "Mode",
            options=[
                "state_adjust (-Delta/Vega * dF)",
                "ffill only",
                "quote_only",
            ],
            index=0,
        )
        if iv_mode_label.startswith("state_adjust"):
            iv_fill_mode = "state_adjust"
        elif iv_mode_label.startswith("ffill"):
            iv_fill_mode = "ffill"
        else:
            iv_fill_mode = "quote_only"

        fut_move_threshold = float(
            st.number_input(
                "Future move threshold",
                value=0.5,
                step=0.1,
            )
        )
        pool_refresh_seconds = int(
            st.number_input(
                "Pool refresh seconds",
                value=120,
                step=30,
                min_value=0,
            )
        )

        # Initialize / refresh default when selected file changes.
        fp_state_key = "_pool_refresh_fut_move_fp"
        val_state_key = "pool_refresh_fut_move_input"
        if st.session_state.get(fp_state_key) != fp:
            if np.isfinite(strike_gap_suggest) and strike_gap_suggest > 0:
                st.session_state[val_state_key] = float(strike_gap_suggest)
            elif val_state_key not in st.session_state:
                st.session_state[val_state_key] = 0.0
            st.session_state[fp_state_key] = fp
        elif val_state_key not in st.session_state:
            st.session_state[val_state_key] = float(strike_gap_suggest) if (np.isfinite(strike_gap_suggest) and strike_gap_suggest > 0) else 0.0

        pool_refresh_fut_move = float(
            st.number_input(
                "Pool refresh future move threshold (|ΔF|)",
                key=val_state_key,
                step=0.1,
                min_value=0.0,
            )
        )
        if np.isfinite(strike_gap_suggest) and strike_gap_suggest > 0:
            st.caption(f"Suggested default strike gap: {strike_gap_suggest:g}")

        min_valid_n = int(
            st.number_input(
                "Min valid contracts",
                value=6,
                step=1,
                min_value=1,
            )
        )

        use_abs_bar = st.checkbox("Absolute traded vega bars", value=False)

        submitted = st.button("Update charts")

    params = dict(
        fp=fp,
        base_rule=base_rule,
        kline_rule=kline_rule,
        n=n,
        otm_atm_only=otm_atm_only,
        use_vega_weight=use_vega_weight,
        iv_fill_mode=iv_fill_mode,
        fut_move_threshold=fut_move_threshold,
        pool_refresh_seconds=pool_refresh_seconds,
        pool_refresh_fut_move=pool_refresh_fut_move,
        min_valid_n=min_valid_n,
        use_abs_bar=use_abs_bar,
    )
    return params, submitted


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
    src_abs = (REPO_ROOT / src).resolve() if not src.is_absolute() else src.resolve()
    stat = src_abs.stat() if src_abs.exists() else None
    payload = {
        "sig": list(signature(params)),
        "src": str(src_abs),
        "mtime_ns": getattr(stat, "st_mtime_ns", None),
        "size": getattr(stat, "st_size", None),
    }
    key = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    out_dir = APP_CACHE_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"agg_{key}.pkl"


def _load_aggregation_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        obj = pd.read_pickle(cache_path)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _save_aggregation_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, cache_path)


def compute_if_needed(df_raw: pd.DataFrame, params: dict, submitted: bool, cache: dict):
    """Compute series + aggregates only when needed (button or params changed)."""
    sig = signature(params)
    cache_path = _aggregation_cache_path(params)

    cache_has = ("sig" in cache) and ("ser" in cache) and ("bar_ser" in cache)
    if cache_has and (cache.get("sig") != sig) and (not submitted):
        st.info("Parameters changed but charts are not recomputed yet. Click Update Charts to refresh.")

    need = (not cache_has) or submitted or (cache.get("sig") != sig and submitted)
    if not need:
        cache["_agg_cache_path"] = str(cache_path)
        cache["_agg_cache_source"] = "session"
        return

    disk_payload = _load_aggregation_cache(cache_path)
    if disk_payload and disk_payload.get("sig") == sig:
        cache.update(disk_payload)
        cache["_agg_cache_path"] = str(cache_path)
        cache["_agg_cache_source"] = "disk"
        return

    call_kwargs = dict(
        iv_fill_mode=params["iv_fill_mode"],
        fut_move_threshold=float(params["fut_move_threshold"]),
        pool_refresh_seconds=int(params["pool_refresh_seconds"]),
        pool_refresh_fut_move=float(params["pool_refresh_fut_move"]),
        min_valid_n=int(params["min_valid_n"]),
        is_ultra=False,
    )

    try:
        ser, bar_ser, debug_df, details_df = make_iv_and_bar_series(
            df_raw,
            params["base_rule"],
            params["n"],
            params["otm_atm_only"],
            params["use_vega_weight"],
            params["use_abs_bar"],
            **call_kwargs,
        )
    except TypeError as e:
        # 鍏煎鏃х増 aggregation.py锛堜笉鏀寔 pool_refresh_fut_move 鍙傛暟锛?
        if "pool_refresh_fut_move" in str(e):
            call_kwargs.pop("pool_refresh_fut_move", None)
            ser, bar_ser, debug_df, details_df = make_iv_and_bar_series(
                df_raw,
                params["base_rule"],
                params["n"],
                params["otm_atm_only"],
                params["use_vega_weight"],
                params["use_abs_bar"],
                **call_kwargs,
            )
        else:
            raise

    if ser is None or ser.empty:
        st.warning("Failed to build base series. Filtering may be too strict or required fields are missing.")
        cache.clear()
        return

    # signed vega锛堣嫢鏁版嵁甯︿簡 traded_vega_signed 灏辩敤锛涘惁鍒欑疆 0锛?
    if "traded_vega_signed" in df_raw.columns:
        bar_signed = (
            df_raw[["dt_exch", "traded_vega_signed"]]
            .set_index("dt_exch")["traded_vega_signed"]
            .resample(params["base_rule"]).sum()
            .reindex(ser.index)
            .fillna(0.0)
        )
    else:
        bar_signed = pd.Series(0.0, index=ser.index)

    # 鍘绘帀闀挎椂闂存棤鏂拌鎯呯殑鍋滄粸鍖洪棿锛堥粯璁?>120s锛?
    active_mask = build_active_mask(df_raw["dt_exch"], ser.index, max_stale_seconds=120)
    ser = ser.where(active_mask)
    bar_ser = bar_ser.where(active_mask, 0.0)

    # 浠呬繚鐣欎氦鏄撴椂娈碉紙鍘绘帀 23:00~娆℃棩09:00 绛夊瀮鍦炬椂闂达級
    sess_mask = is_in_trading_session(ser.index)
    ser = ser.where(sess_mask)
    bar_ser = bar_ser.where(sess_mask, 0.0)

    # 棰濆鏂紑闀挎椂闂粹€滄暟鍊间笉鍙樷€濈殑妯嚎娈碉紙榛樿>120绉掞級
    ser = break_long_flat_segments(ser, max_stale_seconds=120)

    ohlc = make_ohlc(ser, params["kline_rule"])
    # 涓㈠純绌烘《锛屼繚璇佹í杞村彧淇濈暀鈥滄湁琛屾儏鈥濈殑K绾跨偣浣?
    ohlc = ohlc.dropna(how="any")

    fut_ser = (
        df_raw[["dt_exch", "F_used"]]
        .dropna()
        .set_index("dt_exch")["F_used"]
        .sort_index()
        .resample(params["kline_rule"])
        .last()
        .reindex(ohlc.index)
    )
    # 鏈熻揣绾夸篃鎸夋椿璺冩帺鐮佸拰鍋滄粸鏂嚎澶勭悊锛岄伩鍏嶈法绌烘。姘村钩绾?
    fut_ser = fut_ser.where(ohlc["close"].notna())
    fut_ser = break_long_flat_segments(fut_ser, max_stale_seconds=120)

    bar_agg = bar_ser.resample(params["kline_rule"]).sum().reindex(ohlc.index).fillna(0.0)
    bar_signed_agg = bar_signed.resample(params["kline_rule"]).sum().reindex(ohlc.index).fillna(0.0)

    cache.update(
        fut_ser=fut_ser,
        sig=sig,
        ser=ser,
        bar_ser=bar_ser,
        bar_signed=bar_signed,
        ohlc=ohlc,
        bar_agg=bar_agg,
        bar_signed_agg=bar_signed_agg,
        debug_df=debug_df,
        details_df=details_df,
    )
    cache["_agg_cache_path"] = str(cache_path)
    cache["_agg_cache_source"] = "fresh"
    _save_aggregation_cache(
        cache_path,
        {
            "fut_ser": fut_ser,
            "sig": sig,
            "ser": ser,
            "bar_ser": bar_ser,
            "bar_signed": bar_signed,
            "ohlc": ohlc,
            "bar_agg": bar_agg,
            "bar_signed_agg": bar_signed_agg,
            "debug_df": debug_df,
            "details_df": details_df,
        },
    )


def _build_selected_agg_factor_input(df_raw: pd.DataFrame, cache: dict, params: dict) -> pd.DataFrame:
    """
    Build a synthetic frame for factor evaluation on "selected contracts aggregate".
    Keep factors.py unchanged by adapting input at app layer.
    """
    details_df = cache.get("details_df")
    base_iv_ser = cache.get("ser")
    if details_df is None or getattr(details_df, "empty", True) or base_iv_ser is None or base_iv_ser.empty:
        return pd.DataFrame()

    selected = details_df[["dt", "symbol"]].dropna().drop_duplicates().copy()
    selected = selected.rename(columns={"dt": "_bucket"})
    selected["_bucket"] = pd.to_datetime(selected["_bucket"], errors="coerce")
    selected = selected.dropna(subset=["_bucket", "symbol"])
    if selected.empty:
        return pd.DataFrame()

    needed_cols = ["dt_exch", "symbol", "is_future", "F_used", "d_volume", "spread", "traded_vega", "traded_vega_signed"]
    use_cols = [c for c in needed_cols if c in df_raw.columns]
    if ("dt_exch" not in use_cols) or ("symbol" not in use_cols):
        return pd.DataFrame()

    dfx = df_raw[use_cols].copy()
    if not pd.api.types.is_datetime64_any_dtype(dfx["dt_exch"]):
        dfx["dt_exch"] = pd.to_datetime(dfx["dt_exch"], errors="coerce")
    dfx = dfx[dfx["dt_exch"].notna()]
    if dfx.empty:
        return pd.DataFrame()

    # shrink rows before merge: keep only selected symbols + selected time range
    sel_syms = selected["symbol"].astype(str).unique().tolist()
    if sel_syms:
        dfx = dfx[dfx["symbol"].astype(str).isin(sel_syms) | (dfx.get("is_future", False) == True)]
    tmin = selected["_bucket"].min()
    tmax = selected["_bucket"].max() + pd.Timedelta(params["base_rule"])
    dfx = dfx[(dfx["dt_exch"] >= tmin) & (dfx["dt_exch"] <= tmax)]
    if dfx.empty:
        return pd.DataFrame()

    dfx["_bucket"] = dfx["dt_exch"].dt.floor(params["base_rule"])

    # Per (bucket, symbol) keep latest quote, then aggregate across selected symbols in bucket.
    opt_rows = dfx.merge(selected, on=["_bucket", "symbol"], how="inner")
    if opt_rows.empty:
        return pd.DataFrame()
    opt_last = (
        opt_rows.sort_values("dt_exch")
        .groupby(["_bucket", "symbol"], as_index=False)
        .tail(1)
    )

    if "traded_vega" in opt_last.columns:
        opt_last["_tv_for_factor"] = pd.to_numeric(opt_last["traded_vega"], errors="coerce").fillna(0.0)
    elif "traded_vega_signed" in opt_last.columns:
        # fallback: aggregate activity by absolute signed vega
        opt_last["_tv_for_factor"] = pd.to_numeric(opt_last["traded_vega_signed"], errors="coerce").abs().fillna(0.0)
    else:
        opt_last["_tv_for_factor"] = 0.0

    opt_last["_spread_for_factor"] = pd.to_numeric(opt_last.get("spread"), errors="coerce")
    opt_agg = opt_last.groupby("_bucket", as_index=False).agg(
        traded_vega=("_tv_for_factor", "sum"),
        spread=("_spread_for_factor", "median"),
    )

    # IV in factor base should match main index series (selected pool weighted index).
    iv_base = base_iv_ser.rename("iv").reset_index()
    iv_base.columns = ["_bucket", "iv"]
    iv_base["_bucket"] = pd.to_datetime(iv_base["_bucket"], errors="coerce")

    opt_base = iv_base.merge(opt_agg, on="_bucket", how="left")
    opt_base["traded_vega"] = opt_base["traded_vega"].fillna(0.0)
    opt_base["symbol"] = "__AGG_SELECTED__"
    opt_base["is_option"] = True
    opt_base["is_future"] = False
    opt_base = opt_base.rename(columns={"_bucket": "dt_exch"})

    fut = dfx[dfx.get("is_future", False) == True].copy() if "is_future" in dfx.columns else pd.DataFrame()
    if fut.empty:
        fut_base = pd.DataFrame(columns=["dt_exch", "symbol", "is_option", "is_future", "F_used", "d_volume"])
    else:
        agg_map = {"F_used": ("F_used", "last")}
        if "d_volume" in fut.columns:
            agg_map["d_volume"] = ("d_volume", "sum")
        fut_base = fut.groupby("_bucket", as_index=False).agg(**agg_map).rename(columns={"_bucket": "dt_exch"})
        if "d_volume" not in fut_base.columns:
            fut_base["d_volume"] = 0.0
        fut_base["symbol"] = "__FUT_BASE__"
        fut_base["is_option"] = False
        fut_base["is_future"] = True

    out = pd.concat(
        [
            opt_base[["dt_exch", "symbol", "is_option", "is_future", "iv", "traded_vega", "spread"]],
            fut_base[["dt_exch", "symbol", "is_option", "is_future", "F_used", "d_volume"]],
        ],
        ignore_index=True,
        sort=False,
    )
    out = out.sort_values("dt_exch").reset_index(drop=True)
    return out


def _compute_main_chart_factor_results(df_raw: pd.DataFrame, cache: dict, params: dict, selected_factor_ids: list[str], factor_settings: dict):
    if cache.get("sig") != signature(params):
        return {}, pd.DataFrame()

    ohlc = cache.get("ohlc", pd.DataFrame())
    idx = ohlc.index if ohlc is not None and not ohlc.empty else pd.DatetimeIndex([])
    if len(idx) == 0 or not selected_factor_ids:
        return {}, pd.DataFrame()

    factor_input_sig = (cache.get("sig"), len(df_raw))
    if cache.get("_main_factor_input_sig") == factor_input_sig:
        factor_input = cache.get("_main_factor_input", pd.DataFrame())
    else:
        factor_input = _build_selected_agg_factor_input(df_raw, cache, params)
        cache["_main_factor_input_sig"] = factor_input_sig
        cache["_main_factor_input"] = factor_input
    if factor_input.empty:
        return {}, pd.DataFrame()

    factor_base = build_factor_base_frame(
        df_raw=factor_input,
        symbol="__AGG_SELECTED__",
        base_rule=params["base_rule"],
    )
    if factor_base is None or factor_base.empty:
        return {}, pd.DataFrame()

    factor_results = {}
    for fid in selected_factor_ids:
        cfg = factor_settings.get(fid, {})
        _signal, _trigger_base, _thr = evaluate_factor_trigger(
            factor_base,
            fid,
            mode=cfg.get("mode", "quantile"),
            q=float(cfg.get("q", 0.95)),
            op=cfg.get("op", ">="),
            value=cfg.get("value"),
        )
        _trigger = _trigger_base.resample(params["kline_rule"]).max() if not _trigger_base.empty else pd.Series(False, index=idx)
        _trigger = _trigger.reindex(idx).fillna(False)
        factor_results[fid] = {"signal": _signal, "trigger": _trigger, "threshold": _thr, "config": cfg}

    return factor_results, factor_base


def render_charts(df_raw: pd.DataFrame, params: dict, cache: dict):
    st.markdown("### 主图")

    # Main-chart factor controls (selected contracts aggregate)
    factors = list_factors()
    factor_keys = list(factors.keys())
    factor_labels = [factors[k].label for k in factor_keys]
    label_to_id = {factors[k].label: k for k in factor_keys}

    selected_labels = st.multiselect(
        "主图触发因子（可多选）",
        options=factor_labels,
        default=factor_labels[:1],
        key="main_factor_multi_labels",
    )
    selected_factor_ids = [label_to_id[x] for x in selected_labels if x in label_to_id]

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        threshold_mode = st.selectbox("Main chart threshold mode", ["quantile", "absolute"], index=0, key="main_factor_threshold_mode")
    with c2:
        if threshold_mode == "quantile":
            q = float(st.slider("Main chart quantile q", 0.50, 0.999, 0.95, 0.005, key="main_factor_q"))
            abs_value = None
        else:
            q = 0.95
            abs_value = float(st.number_input("Main chart absolute threshold", value=0.0, step=0.1, key="main_factor_abs_value"))
    with c3:
        op = st.selectbox("Main chart trigger direction", [">=", "<="], index=0, key="main_factor_op")

    fig = build_fut_iv_vega_stack_figure(
        cache["fut_ser"],
        cache["ohlc"],
        cache["bar_agg"],
        cache["bar_signed_agg"],
    )

    factor_results, _factor_base = _compute_main_chart_factor_results(
        df_raw=df_raw,
        cache=cache,
        params=params,
        selected_factor_ids=selected_factor_ids,
        threshold_mode=threshold_mode,
        q=q,
        op=op,
        abs_value=abs_value,
    )

    if cache.get("sig") != signature(params):
        st.caption("Main-chart factor stats are stale until you click Update Charts.")

    idx = cache["ohlc"].index
    for fid in selected_factor_ids:
        trig_ser = factor_results.get(fid, {}).get("trigger", pd.Series(False, index=idx))
        trig_mask = pd.Series(trig_ser, index=getattr(trig_ser, "index", None)).reindex(idx).fillna(False).astype(bool).to_numpy()
        if len(trig_mask) != len(idx):
            trig_mask = np.zeros(len(idx), dtype=bool)
        trig_idx = idx[trig_mask]
        if len(trig_idx) == 0:
            continue
        y_ref = cache["ohlc"].loc[trig_idx, "high"].astype(float)
        y_ref = y_ref.fillna(cache["ohlc"]["close"]).ffill()
        fig.add_trace(
            go.Scatter(
                x=trig_idx,
                y=y_ref,
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#1f77ff", line=dict(color="#0b3d91", width=1.0)),
                name=f"主图触发:{factors[fid].label}",
                hovertemplate=f"主图因子: {factors[fid].label}<br>时间: %{{x}}<extra></extra>",
            ),
            row=1, col=1, secondary_y=False,
        )

    fig = apply_no_blank_time_axis(fig)

    PLOTLY_CONFIG = dict(
        displaylogo=False,
        scrollZoom=True,
        modeBarButtonsToAdd=["autoScale2d", "resetScale2d"],  # 浣犺鐨勬柟妗圓锛氭墜鍔╕杞存弧灞?
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    if selected_factor_ids:
        st.caption(f"Main chart threshold mode: {threshold_mode}; trigger direction: {op}")
        sum_rows = []
        for fid in selected_factor_ids:
            tser = factor_results.get(fid, {}).get("trigger", pd.Series(False, index=idx))
            tser = pd.Series(tser, index=getattr(tser, "index", None)).reindex(idx).fillna(False).astype(bool)
            thr = factor_results.get(fid, {}).get("threshold", np.nan)
            sig_raw = factor_results.get(fid, {}).get("signal", pd.Series(dtype=float))
            sig_stats = _series_stats(sig_raw)
            sig_kline = _safe_resample_signal(sig_raw, params["kline_rule"], idx)
            icir = _compute_ic_ir_from_series(sig_kline, cache["ohlc"]["close"])
            cnt = int(tser.sum()) if len(tser) else 0
            sum_rows.append(
                {
                    "factor": factors[fid].label,
                    "threshold": thr,
                    "trigger_count": cnt,
                    "ic": icir["ic"],
                    "ir": icir["ir"],
                    "ic_n_obs": icir["n_obs"],
                    "ir_n_bins_30m": icir["n_bins"],
                    "min": sig_stats["min"],
                    "max": sig_stats["max"],
                    "q25": sig_stats["q25"],
                    "q50": sig_stats["q50"],
                    "q75": sig_stats["q75"],
                    "q90": sig_stats["q90"],
                }
            )
        st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, height=160)

def render_charts_v2(df_raw: pd.DataFrame, params: dict, cache: dict):
    st.markdown("### 主图")

    factors = list_factors()
    factor_keys = list(factors.keys())
    factor_labels = [factors[k].label for k in factor_keys]
    label_to_id = {factors[k].label: k for k in factor_keys}

    selected_labels = st.multiselect(
        "主图触发因子（可多选）",
        options=factor_labels,
        default=factor_labels[:1],
        key="main_factor_multi_labels_v2",
    )
    selected_factor_ids = [label_to_id[x] for x in selected_labels if x in label_to_id]

    factor_form_sig = (tuple(selected_factor_ids), cache.get("sig"))
    if st.session_state.get("_main_factor_applied_sig") != factor_form_sig:
        st.session_state["_main_factor_applied_sig"] = factor_form_sig
        st.session_state["_main_factor_applied_settings"] = {}

    applied_settings = st.session_state.get("_main_factor_applied_settings", {})
    pending_settings = {}
    if selected_factor_ids:
        st.caption("Choose factor parameters first, then click Apply Main-Chart Factor Settings.")
    for fid in selected_factor_ids:
        fdef = factors[fid]
        prev_cfg = applied_settings.get(fid, {})
        c1, c2, c3 = st.columns([1, 1, 1])
        prev_mode = prev_cfg.get("mode", "quantile")
        mode = c1.selectbox(
            f"{fdef.label} threshold mode",
            ["quantile", "absolute"],
            index=0 if prev_mode == "quantile" else 1,
            key=f"main_factor_threshold_mode_v2_{fid}",
        )
        if mode == "quantile":
            q = float(
                c2.slider(
                    f"{fdef.label} quantile",
                    0.50,
                    0.999,
                    float(prev_cfg.get("q", fdef.default_q)),
                    0.005,
                    key=f"main_factor_q_v2_{fid}",
                )
            )
            value = None
        else:
            q = float(prev_cfg.get("q", fdef.default_q))
            value = float(
                c2.number_input(
                    f"{fdef.label} absolute threshold",
                    value=float(prev_cfg.get("value", 0.0) or 0.0),
                    step=0.1,
                    key=f"main_factor_abs_value_v2_{fid}",
                )
            )
        prev_op = prev_cfg.get("op", fdef.default_op)
        op = c3.selectbox(
            f"{fdef.label} trigger direction",
            [">=", "<="],
            index=0 if prev_op == ">=" else 1,
            key=f"main_factor_op_v2_{fid}",
        )
        pending_settings[fid] = {"mode": mode, "q": q, "op": op, "value": value}

    factor_settings = applied_settings
    factor_submit = st.button("应用主图因子参数", key="main_factor_apply_v2") if selected_factor_ids else False
    if factor_submit:
        st.session_state["_main_factor_applied_settings"] = pending_settings
        factor_settings = pending_settings

    fig = build_fut_iv_vega_stack_figure(
        cache["fut_ser"],
        cache["ohlc"],
        cache["bar_agg"],
        cache["bar_signed_agg"],
    )

    can_run_main_factors = bool(selected_factor_ids) and all(fid in factor_settings for fid in selected_factor_ids)
    if can_run_main_factors:
        factor_results, _factor_base = _compute_main_chart_factor_results(
            df_raw=df_raw,
            cache=cache,
            params=params,
            selected_factor_ids=selected_factor_ids,
            factor_settings=factor_settings,
        )
    else:
        factor_results, _factor_base = {}, pd.DataFrame()

    if cache.get("sig") != signature(params):
        st.caption("Factor stats are stale until you click Update Charts.")

    idx = cache["ohlc"].index
    for fid in selected_factor_ids:
        trig_ser = factor_results.get(fid, {}).get("trigger", pd.Series(False, index=idx))
        trig_mask = pd.Series(trig_ser, index=getattr(trig_ser, "index", None)).reindex(idx).fillna(False).astype(bool).to_numpy()
        if len(trig_mask) != len(idx):
            trig_mask = np.zeros(len(idx), dtype=bool)
        trig_idx = idx[trig_mask]
        if len(trig_idx) == 0:
            continue
        y_ref = cache["ohlc"].loc[trig_idx, "high"].astype(float)
        y_ref = y_ref.fillna(cache["ohlc"]["close"]).ffill()
        fig.add_trace(
            go.Scatter(
                x=trig_idx,
                y=y_ref,
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#1f77ff", line=dict(color="#0b3d91", width=1.0)),
                name=f"主图触发:{factors[fid].label}",
                hovertemplate=f"主图因子: {factors[fid].label}<br>时间: %{{x}}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    fig = apply_no_blank_time_axis(fig)
    plotly_config = dict(
        displaylogo=False,
        scrollZoom=True,
        modeBarButtonsToAdd=["autoScale2d", "resetScale2d"],
    )
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    if cache.get("_agg_cache_path"):
        st.caption(f"Main chart aggregation cache: {cache.get('_agg_cache_path')} [{cache.get('_agg_cache_source', 'unknown')}]")

    if selected_factor_ids and not can_run_main_factors:
        st.caption("Main-chart factor settings have not been applied yet.")

    if selected_factor_ids and can_run_main_factors:
        sum_rows = []
        for fid in selected_factor_ids:
            tser = factor_results.get(fid, {}).get("trigger", pd.Series(False, index=idx))
            tser = pd.Series(tser, index=getattr(tser, "index", None)).reindex(idx).fillna(False).astype(bool)
            thr = factor_results.get(fid, {}).get("threshold", np.nan)
            cfg = factor_results.get(fid, {}).get("config", {})
            sig_raw = factor_results.get(fid, {}).get("signal", pd.Series(dtype=float))
            sig_stats = _series_stats(sig_raw)
            sig_kline = _safe_resample_signal(sig_raw, params["kline_rule"], idx)
            icir = _compute_ic_ir_from_series(sig_kline, cache["ohlc"]["close"])
            cnt = int(tser.sum()) if len(tser) else 0
            sum_rows.append(
                {
                    "factor": factors[fid].label,
                    "mode": cfg.get("mode"),
                    "op": cfg.get("op"),
                    "threshold": thr,
                    "trigger_count": cnt,
                    "ic": icir["ic"],
                    "ir": icir["ir"],
                    "ic_n_obs": icir["n_obs"],
                    "ir_n_bins_30m": icir["n_bins"],
                    "min": sig_stats["min"],
                    "max": sig_stats["max"],
                    "q25": sig_stats["q25"],
                    "q50": sig_stats["q50"],
                    "q75": sig_stats["q75"],
                    "q90": sig_stats["q90"],
                }
            )
        st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, height=180)


def render_single_contract_iv_chart(df_raw: pd.DataFrame, params: dict, cache: dict, main_time_index: Optional[pd.DatetimeIndex] = None):
    """Render single-contract IV K-line + traded vega + traded volume."""
    st.markdown("### Single Contract IV / Traded Vega / Volume")

    if "symbol" not in df_raw.columns or "iv" not in df_raw.columns:
        st.info("Missing symbol or iv column; cannot draw single-contract chart.")
        return

    dfo = df_raw.copy()
    if "is_option" in dfo.columns:
        dfo = dfo[dfo["is_option"] == True]

    syms = sorted(dfo["symbol"].dropna().astype(str).unique().tolist())
    if not syms:
        st.info("No option contract is available in the current dataset.")
        return

    default_idx = 0
    if "single_contract_symbol" in st.session_state:
        try:
            default_idx = syms.index(st.session_state["single_contract_symbol"])
        except Exception:
            default_idx = 0

    sym = st.selectbox("Select single contract", syms, index=default_idx, key="single_contract_symbol")

    # --- 閫氱敤鍥犲瓙瑙﹀彂鎺ュ彛锛堝彲鎵╁睍锛?---
    factors = list_factors()
    factor_keys = list(factors.keys())
    factor_labels = [factors[k].label for k in factor_keys]
    label_to_id = {factors[k].label: k for k in factor_keys}

    selected_labels = st.multiselect(
        "触发因子（可多选）",
        options=factor_labels,
        default=factor_labels[:1],
        key="factor_multi_labels",
    )
    selected_factor_ids = [label_to_id[x] for x in selected_labels if x in label_to_id]

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        threshold_mode = st.selectbox("Threshold mode", ["quantile", "absolute"], index=0, key="factor_threshold_mode")
    with c2:
        if threshold_mode == "quantile":
            q = float(st.slider("Quantile q", 0.50, 0.999, 0.95, 0.005, key="factor_q"))
            abs_value = None
        else:
            q = 0.95
            abs_value = float(st.number_input("Absolute threshold", value=0.0, step=0.1, key="factor_abs_value"))
    with c3:
        op = st.selectbox("Trigger direction", [">=", "<="], index=0, key="factor_op")

    cols = ["dt_exch", "symbol", "iv"]
    if "traded_vega" in dfo.columns:
        cols.append("traded_vega")
    if "traded_vega_signed" in dfo.columns:
        cols.append("traded_vega_signed")
    if "trade_volume_lots" in dfo.columns:
        cols.append("trade_volume_lots")
    elif "d_volume" in dfo.columns:
        cols.append("d_volume")
    elif "volume" in dfo.columns:
        cols.append("volume")

    ds = dfo[dfo["symbol"].astype(str) == str(sym)][cols].copy()
    ds = ds.dropna(subset=["dt_exch"]).sort_values("dt_exch")
    ds = ds[is_in_trading_session(pd.DatetimeIndex(ds["dt_exch"])) .values]

    if ds.empty:
        st.warning(f"No usable data for contract {sym}.")
        return

    iv_ser = (
        ds.set_index("dt_exch")["iv"]
        .resample(params["base_rule"])
        .last()
        .sort_index()
        .ffill()
        .bfill()
    )
    iv_ser = iv_ser[iv_ser > 0]
    if iv_ser.empty:
        st.warning(f"Contract {sym} has no valid IV (> 0) data.")
        return

    iv_ser = break_long_flat_segments(iv_ser, max_stale_seconds=120)

    ohlc_single = make_ohlc(iv_ser, params["kline_rule"]).dropna(how="any")
    if ohlc_single.empty:
        st.warning(f"Contract {sym} cannot form a K-line under the current display rule.")
        return

    # 鎴愪氦 vega锛氫紭鍏?signed锛堝彲鐪嬫柟鍚戯級锛屽惁鍒欑敤 traded_vega
    if "traded_vega_signed" in ds.columns:
        vega_ser = ds.set_index("dt_exch")["traded_vega_signed"].resample(params["kline_rule"]).sum()
    elif "traded_vega" in ds.columns:
        vega_ser = ds.set_index("dt_exch")["traded_vega"].resample(params["kline_rule"]).sum()
    else:
        vega_ser = pd.Series(0.0, index=ohlc_single.index)

    # 鎴愪氦閲忥紙寮犳暟锛夛細浼樺厛 trade_volume_lots锛屽啀 d_volume锛屽啀 volume 宸垎
    if "trade_volume_lots" in ds.columns:
        vol_ser = ds.set_index("dt_exch")["trade_volume_lots"].resample(params["kline_rule"]).sum()
    elif "d_volume" in ds.columns:
        vol_ser = ds.set_index("dt_exch")["d_volume"].resample(params["kline_rule"]).sum()
    elif "volume" in ds.columns:
        vol_ser = (
            ds.set_index("dt_exch")["volume"]
            .resample(params["kline_rule"]).last()
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
        )
    else:
        vol_ser = pd.Series(0.0, index=ohlc_single.index)

    orig_idx = ohlc_single.index
    idx = pd.DatetimeIndex(main_time_index) if (main_time_index is not None and len(main_time_index) > 0) else orig_idx
    ohlc_single = ohlc_single.reindex(idx)
    vega_ser = vega_ser.reindex(idx).fillna(0.0)
    vol_ser = vol_ser.reindex(idx).fillna(0.0)

    # 鑻ュ榻愪富鏃堕棿杞村悗K绾垮叏绌猴紝鍒欏洖閫€鍒拌鍚堢害鑷韩鏃堕棿杞达紙閬垮厤鈥滃浘鍒蜂笉鍑烘潵鈥濓級
    if ohlc_single[["open", "high", "low", "close"]].notna().sum().sum() == 0:
        idx = orig_idx
        ohlc_single = ohlc_single.reindex(idx)
        vega_ser = vega_ser.reindex(idx).fillna(0.0)
        vol_ser = vol_ser.reindex(idx).fillna(0.0)

    # 鍥犲瓙瑙﹀彂锛堥€氱敤鎺ュ彛锛氭柊澧炲洜瀛愬彧闇€鍦?iv_inspector/factors.py 娉ㄥ唽锛?
    precomputed = _get_precomputed_factor_materials(params, cache)
    factor_base = get_symbol_factor_material(precomputed, str(sym))
    using_precomputed = factor_base is not None and not factor_base.empty
    if not using_precomputed:
        factor_base = build_factor_base_frame(df_raw=df_raw, symbol=str(sym), base_rule=params["base_rule"])
    factor_results = {}
    for fid in selected_factor_ids:
        _signal, _trigger_base, _thr = evaluate_factor_trigger(
            factor_base,
            fid,
            mode=threshold_mode,
            q=q,
            op=op,
            value=abs_value,
        )
        _trigger = _trigger_base.resample(params["kline_rule"]).max() if not _trigger_base.empty else pd.Series(False, index=idx)
        _trigger = _trigger.reindex(idx).fillna(False)
        factor_results[fid] = {"signal": _signal, "trigger": _trigger, "threshold": _thr}

    # Vega 鏌憋細缁熶竴鐢荤粷瀵瑰€硷紱棰滆壊琛ㄧず鏂瑰悜锛堟=绾紝璐?缁匡級
    vega_sign = vega_ser.copy()
    vega_plot = vega_ser.abs()
    bar_colors = ["red" if v > 0 else "green" if v < 0 else "gray" for v in vega_sign.values]

    # 鎴愪氦閲忔煴鎸塊绾挎定璺岄厤鑹诧紱鏃燢绾垮鐏拌壊
    valid_k = ohlc_single[["open", "close"]].notna().all(axis=1).values
    up = valid_k & (ohlc_single["close"].values >= ohlc_single["open"].values)
    vol_colors = ["red" if u else ("gray" if not vk else "green") for u, vk in zip(up, valid_k)]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.19, 0.19],
        subplot_titles=(f"IV K-line: {sym}", "Traded Vega", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=idx,
            open=ohlc_single["open"],
            high=ohlc_single["high"],
            low=ohlc_single["low"],
            close=ohlc_single["close"],
            name="IV",
            increasing_line_color="red",
            decreasing_line_color="green",
        ),
        row=1, col=1,
    )

    # 澶氬洜瀛愯Е鍙戠偣鎵撴爣锛堜笉鍚岄鑹诧級
    factor_colors = ["#8a2be2", "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#17becf"]
    for i, fid in enumerate(selected_factor_ids):
        trig_ser = factor_results.get(fid, {}).get("trigger", pd.Series(False, index=idx))
        trig_mask = pd.Series(trig_ser, index=getattr(trig_ser, "index", None)).reindex(idx).fillna(False).astype(bool).to_numpy()
        if len(trig_mask) != len(idx):
            trig_mask = np.zeros(len(idx), dtype=bool)
        trig_idx = idx[trig_mask]
        if len(trig_idx) == 0:
            continue
        y_ref = ohlc_single.loc[trig_idx, "high"].astype(float)
        y_ref = y_ref.fillna(ohlc_single["close"]).ffill()
        fig.add_trace(
            go.Scatter(
                x=trig_idx,
                y=y_ref,
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#1f77ff", line=dict(color="#0b3d91", width=1.0)),
                name=f"触发:{factors[fid].label}",
                hovertemplate=f"因子: {factors[fid].label}<br>时间: %{{x}}<extra></extra>",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Bar(x=idx, y=vega_plot.values, marker_color=bar_colors, name="traded vega(abs)", opacity=0.75),
        row=2, col=1,
    )

    fig.add_trace(
        go.Bar(x=idx, y=vol_ser.values, marker_color=vol_colors, name="volume_lots", opacity=0.75),
        row=3, col=1,
    )

    fig.update_layout(height=860, margin=dict(l=40, r=30, t=40, b=30), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig = apply_no_blank_time_axis(fig)

    PLOTLY_CONFIG = dict(
        displaylogo=False,
        scrollZoom=True,
        modeBarButtonsToAdd=["autoScale2d", "resetScale2d"],
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    if using_precomputed:
        st.caption(f"Factor materials: {factor_store_path(params['fp'], params['base_rule'])}")
    else:
        st.caption("Factor materials not found for current base_rule; using in-app fallback build.")

    if not selected_factor_ids:
        st.caption("No factor selected")
        return

    st.caption(f"Threshold mode: {threshold_mode}; trigger direction: {op}")

    sum_rows = []
    trig_tables = []
    for fid in selected_factor_ids:
        tser = factor_results.get(fid, {}).get("trigger", pd.Series(False, index=idx))
        tser = pd.Series(tser, index=getattr(tser, "index", None)).reindex(idx).fillna(False).astype(bool)
        thr = factor_results.get(fid, {}).get("threshold", np.nan)
        sig_raw = factor_results.get(fid, {}).get("signal", pd.Series(dtype=float))
        sig_stats = _series_stats(sig_raw)
        sig_kline = _safe_resample_signal(sig_raw, params["kline_rule"], idx)
        icir = _compute_ic_ir_from_series(sig_kline, ohlc_single["close"])
        cnt = int(tser.sum()) if len(tser) else 0
        sum_rows.append({
            "factor": factors[fid].label,
            "threshold": thr,
            "trigger_count": cnt,
            "ic": icir["ic"],
            "ir": icir["ir"],
            "ic_n_obs": icir["n_obs"],
            "ir_n_bins_30m": icir["n_bins"],
            "min": sig_stats["min"],
            "max": sig_stats["max"],
            "q25": sig_stats["q25"],
            "q50": sig_stats["q50"],
            "q75": sig_stats["q75"],
            "q90": sig_stats["q90"],
        })
        if cnt > 0:
            tdf = pd.DataFrame({"factor": factors[fid].label, "trigger_time": idx[tser.values]})
            trig_tables.append(tdf)

    st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, height=180)

    if trig_tables:
        trig_table = pd.concat(trig_tables, ignore_index=True)
        st.dataframe(trig_table.head(500), use_container_width=True, height=260)


def render_debug_table(cache: dict):
    """Debug: show the per-base-bucket aggregation table to validate synthesis."""
    debug_df = cache.get("debug_df")
    if debug_df is None or getattr(debug_df, "empty", True):
        return

    st.markdown("### 调试")
    st.dataframe(debug_df, use_container_width=True, height=320)



def render_drilldown(df_raw: pd.DataFrame, params: dict, cache: dict):
    """Drilldown tables over a user-selected time window."""
    st.markdown("### 钻取（自选时间段）")

    if "dt_exch" not in df_raw.columns:
        st.error("Missing dt_exch column; cannot choose drilldown window.")
        return

    tmin = df_raw["dt_exch"].min()
    tmax = df_raw["dt_exch"].max()

    default_start = tmin
    default_end = min(tmax, tmin + pd.Timedelta(minutes=2))

    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    with col_t1:
        start_str = st.text_input(
            "Start time (YYYY-MM-DD HH:MM:SS)",
            value=default_start.strftime("%Y-%m-%d %H:%M:%S"),
        )
    with col_t2:
        end_str = st.text_input(
            "End time (YYYY-MM-DD HH:MM:SS)",
            value=default_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
    with col_t3:
        quick_minutes = int(st.number_input("Quick window minutes (0 = off)", min_value=0, value=0, step=1))

    def _parse_ts(s: str, fallback: pd.Timestamp) -> pd.Timestamp:
        try:
            return pd.Timestamp(s)
        except Exception:
            return fallback

    start = _parse_ts(start_str, default_start)
    end = _parse_ts(end_str, default_end)

    if quick_minutes and quick_minutes > 0:
        end = start + pd.Timedelta(minutes=quick_minutes)

    # clamp
    if start < tmin:
        start = tmin
    if end > tmax:
        end = tmax
    if end <= start:
        end = min(tmax, start + pd.Timedelta(minutes=1))

    df_interval = df_raw[(df_raw["dt_exch"] >= start) & (df_raw["dt_exch"] <= end)].copy()
    details_df = cache.get("details_df")
    if details_df is not None and not getattr(details_df, "empty", True) and "dt" in details_df.columns:
        details_interval = details_df[(pd.to_datetime(details_df["dt"]) >= start) & (pd.to_datetime(details_df["dt"]) <= end)].copy()
    else:
        details_interval = pd.DataFrame()

    n_main = int(st.number_input("ATM池合约数", min_value=1, value=int(params["n"]), step=1))
    n_alt = n_main + 2
    top_m = 20

    tables = render_drilldown_tabs(
        df_interval=df_interval,
        details_interval=details_interval,
        n_main=n_main,
        n_alt=n_alt,
        top_m=top_m,
        otm_atm_only=params["otm_atm_only"],
    )

    if tables:
        xbytes = tables_to_excel_bytes(tables)
        st.download_button(
            "Download drilldown tables (Excel)",
            data=xbytes,
            file_name=f"drilldown_{start:%Y%m%d_%H%M%S}_{end:%H%M%S}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def main():
    st.title("IV曲线 + 成交Vega")

    cache = st.session_state.setdefault("cache", {})
    params, submitted = sidebar_params()

    if not params["fp"]:
        st.info("No selectable file found under data/derived.")
        return

    # Resolve selected path: support relative display paths (relative to repo root)
    fp = params["fp"]
    if fp and not Path(fp).is_absolute():
        fp = str((REPO_ROOT / fp).resolve())
    df_raw = load_data(fp)
    if df_raw is None or df_raw.empty:
        st.warning("Empty data.")
        return

    compute_if_needed(df_raw, params, submitted, cache)
    if ("ser" not in cache) or cache.get("ser") is None or cache.get("ser").empty:
        return

    render_charts_v2(df_raw, params, cache)
    render_single_contract_iv_chart(df_raw, params, cache, main_time_index=cache.get("ohlc", pd.DataFrame()).index)
    render_debug_table(cache)
    render_drilldown(df_raw, params, cache)


if __name__ == "__main__":
    main()
