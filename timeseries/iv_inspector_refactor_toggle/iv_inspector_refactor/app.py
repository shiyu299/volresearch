# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from iv_inspector.data import list_data_files, load_data
from iv_inspector.aggregation import make_iv_and_bar_series, make_ohlc
from iv_inspector.viz import build_fut_iv_vega_stack_figure
from iv_inspector.drilldown import render_drilldown_tabs, tables_to_excel_bytes
from iv_inspector.factors import list_factors, build_factor_base_frame, evaluate_factor_trigger

st.set_page_config(layout="wide")


def apply_no_blank_time_axis(fig):
    """Compress long no-trade/no-update gaps by rendering x-axis as category."""
    fig.update_xaxes(type="category")
    return fig


def build_active_mask(raw_dt: pd.Series, target_index: pd.DatetimeIndex, max_stale_seconds: int = 120) -> pd.Series:
    """Mark points as active only if latest market update is within max_stale_seconds."""
    evt = pd.to_datetime(raw_dt, errors="coerce").dropna().drop_duplicates().sort_values()
    if evt.empty or len(target_index) == 0:
        return pd.Series(False, index=target_index)

    # merge_asof 要求左右键 dtype 完全一致；统一到 datetime64[ns]
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
        st.markdown("### 参数（只在点“更新图表”后才会重算/重画上方K线与成交量）")

        files = list_data_files()
        fp = st.selectbox("选择数据文件（data/ 下）", files) if files else None

        base_rule = st.selectbox("基础频率（生成iv序列粒度）", ["250ms", "500ms", "1S", "2S", "5S"], index=2)
        kline_rule = st.selectbox("展示周期（细周期折线，粗周期K线）", ["2S", "5S", "10S", "30S", "1min", "5min"], index=4)

        n = int(st.slider("ATM附近 n 个合约（用于画图）", 2, 60, 8, 1))
        otm_atm_only = st.checkbox("只选 OTM + ATM（过滤 ITM）", value=True)
        use_vega_weight = st.checkbox("iv 按 vega 加权", value=True)

        st.markdown("#### 无新行情时的 IV 处理")
        iv_mode_label = st.radio(
            "模式",
            options=[
                "状态推算（按 -Delta/Vega * dF 修正）",
                "纯延续（ffill iv，不做修正）",
                "仅使用当秒报价（quote_only，用于校验）",
            ],
            index=0,
        )
        if iv_mode_label.startswith("状态推算"):
            iv_fill_mode = "state_adjust"
        elif iv_mode_label.startswith("纯延续"):
            iv_fill_mode = "ffill"
        else:
            iv_fill_mode = "quote_only"

        fut_move_threshold = float(
            st.number_input(
                "期货变动阈值（<=阈值不做IV推算；建议=1 tick）",
                value=0.5,
                step=0.1,
            )
        )
        pool_refresh_seconds = int(
            st.number_input(
                "候选池刷新间隔（秒，默认120=2分钟）",
                value=120,
                step=30,
                min_value=0,
            )
        )

        min_valid_n = int(
            st.number_input(
                "最少有效合约数（少于此值则不更新IV，沿用前值）",
                value=4,
                step=1,
                min_value=1,
            )
        )

        use_abs_bar = st.checkbox("柱状图取绝对值", value=False)

        submitted = st.button("更新图表")

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
        int(params["min_valid_n"]),
        params["use_abs_bar"],
    )


def compute_if_needed(df_raw: pd.DataFrame, params: dict, submitted: bool, cache: dict):
    """Compute series + aggregates only when needed (button or params changed)."""
    sig = signature(params)

    cache_has = ("sig" in cache) and ("ser" in cache) and ("bar_ser" in cache)
    if cache_has and (cache.get("sig") != sig) and (not submitted):
        st.info("你已修改参数但未点击“更新图表”。当前上方图表仍显示上一次计算结果。")

    need = (not cache_has) or submitted or (cache.get("sig") != sig and submitted)
    if not need:
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
        min_valid_n=int(params["min_valid_n"]),
        is_ultra=False,
    )

    if ser is None or ser.empty:
        st.warning("无法生成基础序列（过滤太严或字段缺失）。")
        cache.clear()
        return

    # signed vega（若数据带了 traded_vega_signed 就用；否则置 0）
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

    # 去掉长时间无新行情的停滞区间（默认 >120s）
    active_mask = build_active_mask(df_raw["dt_exch"], ser.index, max_stale_seconds=120)
    ser = ser.where(active_mask)
    bar_ser = bar_ser.where(active_mask, 0.0)

    # 仅保留交易时段（去掉 23:00~次日09:00 等垃圾时间）
    sess_mask = is_in_trading_session(ser.index)
    ser = ser.where(sess_mask)
    bar_ser = bar_ser.where(sess_mask, 0.0)

    # 额外断开长时间“数值不变”的横线段（默认>120秒）
    ser = break_long_flat_segments(ser, max_stale_seconds=120)

    ohlc = make_ohlc(ser, params["kline_rule"])
    # 丢弃空桶，保证横轴只保留“有行情”的K线点位
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
    # 期货线也按活跃掩码和停滞断线处理，避免跨空档水平线
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


def render_charts(cache: dict):
    st.markdown("### 图表")

    fig = build_fut_iv_vega_stack_figure(
        cache["fut_ser"],
        cache["ohlc"],
        cache["bar_agg"],
        cache["bar_signed_agg"],
    )
    fig = apply_no_blank_time_axis(fig)

    PLOTLY_CONFIG = dict(
        displaylogo=False,
        scrollZoom=True,
        modeBarButtonsToAdd=["autoScale2d", "resetScale2d"],  # 你要的方案A：手动Y轴满屏
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)



def render_single_contract_iv_chart(df_raw: pd.DataFrame, params: dict, main_time_index: Optional[pd.DatetimeIndex] = None):
    """Render single-contract IV K-line + traded vega + traded volume."""
    st.markdown("### 单合约 IV / 成交Vega / 成交量（张数）")

    if "symbol" not in df_raw.columns or "iv" not in df_raw.columns:
        st.info("数据缺少 symbol 或 iv 字段，无法绘制单合约图。")
        return

    dfo = df_raw.copy()
    if "is_option" in dfo.columns:
        dfo = dfo[dfo["is_option"] == True]

    syms = sorted(dfo["symbol"].dropna().astype(str).unique().tolist())
    if not syms:
        st.info("当前数据没有可选期权合约。")
        return

    default_idx = 0
    if "single_contract_symbol" in st.session_state:
        try:
            default_idx = syms.index(st.session_state["single_contract_symbol"])
        except Exception:
            default_idx = 0

    sym = st.selectbox("选择单合约", syms, index=default_idx, key="single_contract_symbol")

    # --- 通用因子触发接口（可扩展） ---
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
        threshold_mode = st.selectbox("阈值模式", ["quantile", "absolute"], index=0, key="factor_threshold_mode")
    with c2:
        if threshold_mode == "quantile":
            q = float(st.slider("分位数 q（多因子统一）", 0.50, 0.999, 0.95, 0.005, key="factor_q"))
            abs_value = None
        else:
            q = 0.95
            abs_value = float(st.number_input("绝对阈值（多因子统一）", value=0.0, step=0.1, key="factor_abs_value"))
    with c3:
        op = st.selectbox("触发方向", [">=", "<="], index=0, key="factor_op")

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
        st.warning(f"合约 {sym} 无可用数据。")
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
        st.warning(f"合约 {sym} 无有效 IV（>0）数据。")
        return

    iv_ser = break_long_flat_segments(iv_ser, max_stale_seconds=120)

    ohlc_single = make_ohlc(iv_ser, params["kline_rule"]).dropna(how="any")
    if ohlc_single.empty:
        st.warning(f"合约 {sym} 在当前周期下无法生成K线。")
        return

    # 成交 vega：优先 signed（可看方向），否则用 traded_vega
    if "traded_vega_signed" in ds.columns:
        vega_ser = ds.set_index("dt_exch")["traded_vega_signed"].resample(params["kline_rule"]).sum()
    elif "traded_vega" in ds.columns:
        vega_ser = ds.set_index("dt_exch")["traded_vega"].resample(params["kline_rule"]).sum()
    else:
        vega_ser = pd.Series(0.0, index=ohlc_single.index)

    # 成交量（张数）：优先 trade_volume_lots，再 d_volume，再 volume 差分
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

    # 若对齐主时间轴后K线全空，则回退到该合约自身时间轴（避免“图刷不出来”）
    if ohlc_single[["open", "high", "low", "close"]].notna().sum().sum() == 0:
        idx = orig_idx
        ohlc_single = ohlc_single.reindex(idx)
        vega_ser = vega_ser.reindex(idx).fillna(0.0)
        vol_ser = vol_ser.reindex(idx).fillna(0.0)

    # 因子触发（通用接口：新增因子只需在 iv_inspector/factors.py 注册）
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
        factor_results[fid] = {"trigger": _trigger, "threshold": _thr}

    # Vega 柱：统一画绝对值；颜色表示方向（正=红，负=绿）
    vega_sign = vega_ser.copy()
    vega_plot = vega_ser.abs()
    bar_colors = ["red" if v > 0 else "green" if v < 0 else "gray" for v in vega_sign.values]

    # 成交量柱按K线涨跌配色；无K线处灰色
    valid_k = ohlc_single[["open", "close"]].notna().all(axis=1).values
    up = valid_k & (ohlc_single["close"].values >= ohlc_single["open"].values)
    vol_colors = ["red" if u else ("gray" if not vk else "green") for u, vk in zip(up, valid_k)]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.19, 0.19],
        subplot_titles=(f"IV K线：{sym}", "成交 Vega", "成交量（张数）"),
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

    # 多因子触发点打标（不同颜色）
    factor_colors = ["#8a2be2", "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#17becf"]
    for i, fid in enumerate(selected_factor_ids):
        trig_ser = factor_results.get(fid, {}).get("trigger", pd.Series(False, index=idx))
        trig_idx = idx[trig_ser.values]
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

    if not selected_factor_ids:
        st.caption("未选择因子")
        return

    st.caption(f"阈值模式：{threshold_mode}｜触发方向：{op}")

    sum_rows = []
    trig_tables = []
    for fid in selected_factor_ids:
        tser = factor_results.get(fid, {}).get("trigger", pd.Series(False, index=idx))
        thr = factor_results.get(fid, {}).get("threshold", np.nan)
        cnt = int(tser.sum()) if len(tser) else 0
        sum_rows.append({
            "factor": factors[fid].label,
            "threshold": thr,
            "trigger_count": cnt,
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

    st.markdown("### 调试（合成校验）")
    st.dataframe(debug_df, use_container_width=True, height=320)



def render_drilldown(df_raw: pd.DataFrame, params: dict):
    """Drilldown tables over a user-selected time window."""
    st.markdown("### 钻取（自选时间段，只看你关心的一段）")

    if "dt_exch" not in df_raw.columns:
        st.error("数据中缺少 dt_exch 字段，无法选择时间段")
        return

    tmin = df_raw["dt_exch"].min()
    tmax = df_raw["dt_exch"].max()

    default_start = tmin
    default_end = min(tmax, tmin + pd.Timedelta(minutes=2))

    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    with col_t1:
        start_str = st.text_input(
            "开始时间（YYYY-MM-DD HH:MM:SS）",
            value=default_start.strftime("%Y-%m-%d %H:%M:%S"),
        )
    with col_t2:
        end_str = st.text_input(
            "结束时间（YYYY-MM-DD HH:MM:SS）",
            value=default_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
    with col_t3:
        quick_minutes = int(st.number_input("快速窗口（分钟，0=不用）", min_value=0, value=0, step=1))

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

    colA, colB, colC = st.columns(3)
    with colA:
        n_main = int(st.number_input("n（主）", min_value=1, value=int(params["n"]), step=1))
    with colB:
        n_alt = int(st.number_input("k（副）", min_value=1, value=max(1, int(params["n"] // 2)), step=1))
    with colC:
        top_m = int(st.number_input("TopVol m", min_value=1, value=20, step=1))

    tables = render_drilldown_tabs(
        df_interval=df_interval,
        n_main=n_main,
        n_alt=n_alt,
        top_m=top_m,
        otm_atm_only=params["otm_atm_only"],
    )

    if tables:
        xbytes = tables_to_excel_bytes(tables)
        st.download_button(
            "下载三张表（Excel）",
            data=xbytes,
            file_name=f"drilldown_{start:%Y%m%d_%H%M%S}_{end:%H%M%S}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def main():
    st.title("波动率K线 + 成交Vega（Refactored Project v20260303-1238）")

    cache = st.session_state.setdefault("cache", {})
    params, submitted = sidebar_params()

    if not params["fp"]:
        st.info("data/ 下没有可选文件。")
        return

    # Resolve selected path: support relative display paths (relative to repo root)
    fp = params["fp"]
    if fp and not Path(fp).is_absolute():
        fp = str((Path.cwd() / fp).resolve())
    df_raw = load_data(fp)
    if df_raw is None or df_raw.empty:
        st.warning("数据为空。")
        return

    compute_if_needed(df_raw, params, submitted, cache)
    if ("ser" not in cache) or cache.get("ser") is None or cache.get("ser").empty:
        return

    render_charts(cache)
    render_single_contract_iv_chart(df_raw, params, main_time_index=cache.get("ohlc", pd.DataFrame()).index)
    render_debug_table(cache)
    render_drilldown(df_raw, params)


if __name__ == "__main__":
    main()
