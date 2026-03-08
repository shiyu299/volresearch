# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from pathlib import Path

from iv_inspector.data import list_data_files, load_data
from iv_inspector.aggregation import make_iv_and_bar_series, make_ohlc
from iv_inspector.viz import build_fut_iv_vega_stack_figure
from iv_inspector.drilldown import render_drilldown_tabs, tables_to_excel_bytes

st.set_page_config(layout="wide")

PLOTLY_CONFIG = dict(
    displaylogo=False,
    scrollZoom=True,
    modeBarButtonsToAdd=["autoScale2d", "resetScale2d"],
)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def sidebar_params() -> tuple[dict, bool]:
    """Sidebar UI → params dict + submit flag."""
    with st.sidebar:
        st.markdown("### 参数（只在点"更新图表"后才会重算/重画上方K线与成交量）")

        files = list_data_files()
        if not files:
            st.warning("data/ 目录下没有可选文件（支持 parquet / csv / feather）")
            fp = None
        else:
            fp = st.selectbox("选择数据文件（data/ 下）", files)

        base_rule = st.selectbox(
            "基础频率（生成iv序列粒度）",
            ["250ms", "500ms", "1S", "2S", "5S"],
            index=2,
        )
        kline_rule = st.selectbox(
            "展示周期（细周期折线，粗周期K线）",
            ["2S", "5S", "10S", "30S", "1min", "5min"],
            index=4,
        )

        n = int(st.slider("ATM附近 n 个合约（用于画图）", 2, 60, 28, 1))
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
                min_value=0.0,
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
        warmup_seconds = int(
            st.number_input(
                "预热窗口（秒，用于补齐区间开始时的 iv_last）",
                value=60,
                step=10,
                min_value=0,
            )
        )

        use_abs_bar = st.checkbox("柱状图取绝对值", value=False)

        submitted = st.button("更新图表", type="primary")

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
        warmup_seconds=warmup_seconds,
        use_abs_bar=use_abs_bar,
    )
    return params, submitted


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────

def _signature(params: dict) -> tuple:
    """Stable cache key derived from all params that affect computation."""
    return tuple(
        (k, params[k])
        for k in sorted(params)
    )


def _cache_is_valid(cache: dict, sig: tuple) -> bool:
    return (
        "sig" in cache
        and "ser" in cache
        and "bar_ser" in cache
        and cache.get("sig") == sig
    )


# ─────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────

def compute_if_needed(
    df_raw: pd.DataFrame,
    params: dict,
    submitted: bool,
    cache: dict,
) -> None:
    """Trigger heavy computation only when params changed + button clicked,
    or on first load (no cached data yet)."""
    sig = _signature(params)
    valid = _cache_is_valid(cache, sig)

    # Params drifted but user hasn't clicked update → show stale warning
    if cache and not valid and not submitted:
        st.info("你已修改参数但未点击"更新图表"。当前图表仍显示上一次计算结果。")

    # First load: auto-compute so users see something immediately
    first_load = not cache
    if not (first_load or submitted):
        return
    # If cache is valid and user didn't click, nothing to do
    if valid and not submitted:
        return

    warmup = pd.Timedelta(seconds=int(params["warmup_seconds"]))
    t_start = df_raw["dt_exch"].min()

    # Extend df backwards for warmup (used to seed iv_last at interval start)
    df_calc = df_raw  # default: no warmup slice needed (full dataset already loaded)
    output_start = t_start  # keep full range for output

    with st.spinner("计算中…"):
        ser, bar_ser, debug_df, details_df = make_iv_and_bar_series(
            df_calc,
            params["base_rule"],
            params["n"],
            params["otm_atm_only"],
            params["use_vega_weight"],
            params["use_abs_bar"],
            iv_fill_mode=params["iv_fill_mode"],
            fut_move_threshold=float(params["fut_move_threshold"]),
            pool_refresh_seconds=int(params["pool_refresh_seconds"]),
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

    ohlc = make_ohlc(ser, params["kline_rule"])
    if ohlc is None or ohlc.empty:
        st.warning("OHLC 为空：基础序列太稀疏，请放宽参数。")
        cache.clear()
        return

    fut_ser = (
        df_raw[["dt_exch", "F_used"]]
        .dropna(subset=["F_used"])
        .set_index("dt_exch")["F_used"]
        .sort_index()
        .resample(params["kline_rule"])
        .last()
        .reindex(ohlc.index)
        .ffill()
    )
    # Reindex to ohlc.index ensures bar_agg / bar_signed_agg are always aligned with OHLC
    bar_agg = bar_ser.resample(params["kline_rule"]).sum().reindex(ohlc.index).fillna(0.0)
    bar_signed_agg = bar_signed.resample(params["kline_rule"]).sum().reindex(ohlc.index).fillna(0.0)

    cache.update(
        sig=sig,
        ser=ser,
        bar_ser=bar_ser,
        bar_signed=bar_signed,
        ohlc=ohlc,
        fut_ser=fut_ser,
        bar_agg=bar_agg,
        bar_signed_agg=bar_signed_agg,
        debug_df=debug_df,
        details_df=details_df,
    )


# ─────────────────────────────────────────────
# Render: main chart
# ─────────────────────────────────────────────

def render_charts(cache: dict) -> None:
    st.markdown("### 图表")
    fig = build_fut_iv_vega_stack_figure(
        cache["fut_ser"],
        cache["ohlc"],
        cache["bar_agg"],
        cache["bar_signed_agg"],
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ─────────────────────────────────────────────
# Render: debug table (collapsible)
# ─────────────────────────────────────────────

def render_debug_table(cache: dict) -> None:
    debug_df = cache.get("debug_df")
    if debug_df is None or getattr(debug_df, "empty", True):
        return
    with st.expander("调试（合成校验）", expanded=False):
        st.dataframe(debug_df, use_container_width=True, height=320)


# ─────────────────────────────────────────────
# Render: drilldown
# ─────────────────────────────────────────────

def render_drilldown(df_raw: pd.DataFrame, params: dict) -> None:
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
            st.warning(f"时间格式有误：「{s}」，已恢复默认值。请使用 YYYY-MM-DD HH:MM:SS 格式。")
            return fallback

    start = _parse_ts(start_str, default_start)
    end = _parse_ts(end_str, default_end)

    if quick_minutes > 0:
        end = start + pd.Timedelta(minutes=quick_minutes)

    # clamp to data range
    start = max(start, tmin)
    end = min(end, tmax)
    if end <= start:
        end = min(tmax, start + pd.Timedelta(minutes=1))

    df_interval = df_raw[(df_raw["dt_exch"] >= start) & (df_raw["dt_exch"] <= end)].copy()
    st.caption(f"当前区间：{start}  ~  {end}（共 {len(df_interval):,} 行）")

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


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    st.title("波动率K线 + 成交Vega")

    cache = st.session_state.setdefault("cache", {})
    params, submitted = sidebar_params()

    if not params["fp"]:
        st.info("请先将 parquet / csv / feather 文件放入 data/ 目录，然后刷新页面。")
        return

    # Resolve relative path (list_data_files may return relative paths)
    fp = params["fp"]
    if not Path(fp).is_absolute():
        fp = str((Path.cwd() / fp).resolve())

    df_raw = load_data(fp)
    if df_raw is None or df_raw.empty:
        st.warning("数据为空或读取失败，请检查文件格式与 dt_exch 字段。")
        return

    compute_if_needed(df_raw, params, submitted, cache)

    if not cache or cache.get("ser") is None or cache["ser"].empty:
        return

    render_charts(cache)
    render_debug_table(cache)
    render_drilldown(df_raw, params)


if __name__ == "__main__":
    main()
