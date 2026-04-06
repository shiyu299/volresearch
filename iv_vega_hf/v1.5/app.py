from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(layout="wide", page_title="V1.5 Logistic Viewer")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = REPO_ROOT / "data" / "prediction" / "logistic"
DEFAULT_FACTOR_PRIORITY = [
    "flow_5m_sum",
    "iv_dev_ema5m_ratio",
    "iv_mom_5m",
    "iv_willr_5m",
    "resid_z_5mctx",
    "flow_5m_ema",
    "shockF",
    "cum_flow_trade_date",
    "cum_abs_flow_trade_date",
    "cum_flow_session",
    "cum_abs_flow_session",
]
NON_FACTOR_COLS = {
    "dt_exch",
    "trade_date",
    "product",
    "source_file",
    "session_id",
    "session_open",
    "session_close",
    "iv_pool",
    "F_used",
    "p",
    "conf",
    "triggered",
    "pred_sign",
    "true_sign",
    "n_train_seen",
    "train_allowed",
    "predict_allowed",
}


@st.cache_data(show_spinner=False)
def list_products(root: str) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


@st.cache_data(show_spinner=False)
def list_trade_dates(root: str, product: str) -> list[str]:
    base = Path(root) / product
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()], reverse=True)


@st.cache_data(show_spinner=False)
def load_visualization_frame(root: str, product: str, trade_date: str) -> pd.DataFrame:
    day_dir = Path(root) / product / trade_date
    for name in ["visualization.parquet", "timeseries.parquet"]:
        fp = day_dir / name
        if fp.exists():
            try:
                df = pd.read_parquet(fp)
                df["dt_exch"] = pd.to_datetime(df["dt_exch"])
                return df.sort_values("dt_exch").reset_index(drop=True)
            except Exception:
                pass
    for name in ["timeseries.csv"]:
        fp = day_dir / name
        if fp.exists():
            df = pd.read_csv(fp)
            if "dt_exch" in df.columns:
                df["dt_exch"] = pd.to_datetime(df["dt_exch"])
            for col in [
                "trade_date",
                "session_open",
                "session_close",
                "label_ready_ts",
            ]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df.sort_values("dt_exch").reset_index(drop=True)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_evaluation(root: str, product: str, trade_date: str) -> dict:
    fp = Path(root) / product / trade_date / "evaluation.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_detail_frame(root: str, product: str, trade_date: str) -> pd.DataFrame:
    day_dir = Path(root) / product / trade_date
    for name in ["details.parquet"]:
        fp = day_dir / name
        if fp.exists():
            try:
                df = pd.read_parquet(fp)
                if "dt_exch" in df.columns:
                    df["dt_exch"] = pd.to_datetime(df["dt_exch"])
                return df.sort_values(["dt_exch", "symbol"]).reset_index(drop=True)
            except Exception:
                pass
    for name in ["details.csv"]:
        fp = day_dir / name
        if fp.exists():
            df = pd.read_csv(fp)
            if "dt_exch" in df.columns:
                df["dt_exch"] = pd.to_datetime(df["dt_exch"])
            for col in ["trade_date", "session_open", "session_close"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df.sort_values(["dt_exch", "symbol"]).reset_index(drop=True)
    return pd.DataFrame()


def factor_candidates(df: pd.DataFrame) -> list[str]:
    numeric_cols = []
    for col in df.columns:
        if col in NON_FACTOR_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    ordered = [c for c in DEFAULT_FACTOR_PRIORITY if c in numeric_cols]
    remaining = sorted([c for c in numeric_cols if c not in ordered])
    return ordered + remaining


def build_chart(
    df: pd.DataFrame,
    factor_col: str,
    p_upper: float,
    p_lower: float,
    compress_gaps: bool,
) -> go.Figure:
    x = df["dt_exch"].dt.strftime("%m-%d %H:%M:%S") if compress_gaps else df["dt_exch"]
    triggered_up = df["p"].ge(p_upper)
    triggered_down = df["p"].le(p_lower)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=("IV", factor_col, "Logistic p"),
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["iv_pool"],
            mode="lines",
            name="iv_pool",
            line=dict(color="#1f5aa6", width=1.8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x[triggered_up],
            y=df.loc[triggered_up, "iv_pool"],
            mode="markers",
            name=f"p >= {p_upper:.2f}",
            marker=dict(color="#d62728", size=8, symbol="triangle-up"),
            customdata=df.loc[triggered_up, ["p", "conf"]],
            hovertemplate="IV=%{y:.6f}<br>p=%{customdata[0]:.4f}<br>conf=%{customdata[1]:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x[triggered_down],
            y=df.loc[triggered_down, "iv_pool"],
            mode="markers",
            name=f"p <= {p_lower:.2f}",
            marker=dict(color="#2ca02c", size=8, symbol="triangle-down"),
            customdata=df.loc[triggered_down, ["p", "conf"]],
            hovertemplate="IV=%{y:.6f}<br>p=%{customdata[0]:.4f}<br>conf=%{customdata[1]:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[factor_col],
            mode="lines",
            name=factor_col,
            line=dict(color="#ff7f0e", width=1.5),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p"],
            mode="lines",
            name="p",
            line=dict(color="#111111", width=1.6),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=p_upper, line_dash="dot", line_color="#d62728", row=3, col=1)
    fig.add_hline(y=p_lower, line_dash="dot", line_color="#2ca02c", row=3, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#888888", row=3, col=1)
    fig.add_trace(
        go.Scatter(
            x=x[triggered_up],
            y=df.loc[triggered_up, "p"],
            mode="markers",
            name="up trigger",
            marker=dict(color="#d62728", size=7),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x[triggered_down],
            y=df.loc[triggered_down, "p"],
            mode="markers",
            name="down trigger",
            marker=dict(color="#2ca02c", size=7),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=960,
        margin=dict(l=30, r=30, t=45, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    if compress_gaps:
        fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="IV", row=1, col=1)
    fig.update_yaxes(title_text=factor_col, row=2, col=1)
    fig.update_yaxes(title_text="p", row=3, col=1, range=[-0.02, 1.02])
    return fig


def main() -> None:
    st.title("V1.5 Logistic Viewer")
    st.caption(f"Data root: {EXPORT_ROOT}")

    products = list_products(str(EXPORT_ROOT))
    if not products:
        st.warning("还没有找到可视化输出。先跑 v1.5 脚本生成 `data/prediction/logistic/...`。")
        return

    with st.sidebar:
        st.markdown("### Data")
        product = st.selectbox("Product", products)
        trade_dates = list_trade_dates(str(EXPORT_ROOT), product)
        if not trade_dates:
            st.warning("当前品种还没有交易日输出。")
            return
        trade_date = st.selectbox("Trade date", trade_dates)

        df = load_visualization_frame(str(EXPORT_ROOT), product, trade_date)
        if df.empty:
            st.warning("当前交易日没有 visualization/timeseries parquet。")
            return

        factors = factor_candidates(df)
        default_factor = factors[0] if factors else "flow_5m_sum"

        st.markdown("### View")
        factor_col = st.selectbox("Factor", factors, index=factors.index(default_factor) if default_factor in factors else 0)
        p_upper = st.slider("Upper p trigger", min_value=0.50, max_value=0.99, value=0.70, step=0.01)
        p_lower = st.slider("Lower p trigger", min_value=0.01, max_value=0.50, value=0.30, step=0.01)
        compress_gaps = st.checkbox("Compress session gaps", value=True)

    eval_info = load_evaluation(str(EXPORT_ROOT), product, trade_date)
    details_df = load_detail_frame(str(EXPORT_ROOT), product, trade_date)
    metrics = eval_info.get("metrics", {}) if isinstance(eval_info, dict) else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Pred rows", f"{int(df['p'].notna().sum()):,}")
    c3.metric("Triggered", f"{int((df['p'].ge(p_upper) | df['p'].le(p_lower)).sum()):,}")
    c4.metric("IV range", f"{df['iv_pool'].min():.4f} - {df['iv_pool'].max():.4f}")

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("All hit", f"{metrics.get('all_hit', float('nan')):.4f}" if metrics.get("all_hit") is not None else "nan")
        m2.metric("Triggered hit", f"{metrics.get('triggered_hit', float('nan')):.4f}" if metrics.get("triggered_hit") is not None else "nan")
        m3.metric("Coverage", f"{metrics.get('coverage', float('nan')):.4f}" if metrics.get("coverage") is not None else "nan")
        m4.metric("Avg vol pts", f"{metrics.get('avg_vol_points', float('nan')):.4f}" if metrics.get("avg_vol_points") is not None else "nan")

    fig = build_chart(df, factor_col=factor_col, p_upper=p_upper, p_lower=p_lower, compress_gaps=compress_gaps)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview data"):
        tmin = pd.Timestamp(df["dt_exch"].min())
        tmax = pd.Timestamp(df["dt_exch"].max())
        default_end = tmax.to_pydatetime()
        default_start = max(tmin, tmax - pd.Timedelta(minutes=10)).to_pydatetime()
        win_start, win_end = st.slider(
            "Time window",
            min_value=tmin.to_pydatetime(),
            max_value=tmax.to_pydatetime(),
            value=(default_start, default_end),
            format="MM/DD HH:mm:ss",
        )
        mask = df["dt_exch"].between(pd.Timestamp(win_start), pd.Timestamp(win_end))
        df_window = df.loc[mask].copy()
        if df_window.empty:
            st.warning("当前时间窗口没有数据。")
            return
        focus_options = [pd.Timestamp(ts).to_pydatetime() for ts in df_window["dt_exch"].drop_duplicates().tolist()]
        focus_ts = st.select_slider(
            "Focus timestamp",
            options=focus_options,
            value=focus_options[-1],
            format_func=lambda x: pd.Timestamp(x).strftime("%m-%d %H:%M:%S"),
        )
        focus_ts = pd.Timestamp(focus_ts)
        preview_cols = [
            c
            for c in [
                "dt_exch",
                "iv_pool",
                factor_col,
                "flow",
                "cum_flow_trade_date",
                "cum_flow_session",
                "p",
                "conf",
                "pred_sign",
                "true_sign",
                "pool_n",
                "used_n",
                "grace_n",
            ]
            if c in df_window.columns
        ]
        left, right = st.columns([1.2, 1.3])
        with left:
            st.markdown("**Main time series**")
            st.caption(
                f"{len(df_window):,} rows in window, focus at {focus_ts.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            st.dataframe(df_window[preview_cols], use_container_width=True, height=320)
        with right:
            focus_row = df_window[df_window["dt_exch"] == focus_ts].copy()
            if not focus_row.empty:
                st.markdown("**Focused second**")
                st.dataframe(focus_row[preview_cols], use_container_width=True, height=120)
            else:
                st.caption("Focused second is not available in the filtered time series.")

        if not details_df.empty:
            dmask = details_df["dt_exch"].between(pd.Timestamp(win_start), pd.Timestamp(win_end))
            details_window = details_df.loc[dmask].copy()
            snapshot = details_window[details_window["dt_exch"] == focus_ts].copy()
            detail_cols = [
                c
                for c in [
                    "dt_exch",
                    "symbol",
                    "cp",
                    "K",
                    "in_pool",
                    "in_grace",
                    "decay_mult",
                    "iv_contract",
                    "vega_contract",
                    "vega_weight_used",
                    "traded_vega_signed",
                    "spread",
                    "iv_pool",
                    "F_used",
                ]
                if c in details_window.columns
            ]
            snapshot_cols = [
                c
                for c in [
                    "dt_exch",
                    "symbol",
                    "cp",
                    "K",
                    "in_pool",
                    "in_grace",
                    "decay_mult",
                    "iv_contract",
                    "vega_contract",
                    "vega_weight_used",
                    "traded_vega_signed",
                    "spread",
                    "iv_pool",
                    "F_used",
                ]
                if c in snapshot.columns
            ]
            if not snapshot.empty:
                snapshot = snapshot.sort_values(
                    by=["in_pool", "in_grace", "K", "symbol"],
                    ascending=[False, False, True, True],
                )
                st.markdown("**IV composition at focused second**")
                st.caption(
                    f"{len(snapshot):,} contracts contributing at {focus_ts.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                st.dataframe(snapshot[snapshot_cols], use_container_width=True, height=260)
            else:
                st.caption("Focused second has no composition detail rows.")

            st.markdown("**Window composition details**")
            st.caption(
                "Use this table to inspect whether symbols enter or leave the ATM pool around IV spikes."
            )
            st.dataframe(details_window[detail_cols], use_container_width=True, height=360)
        else:
            st.caption("No details table found for this trade date.")

    with st.expander("Evaluation JSON"):
        st.json(eval_info if eval_info else {})


if __name__ == "__main__":
    main()
