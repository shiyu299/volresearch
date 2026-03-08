# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plotly.subplots import make_subplots

def plot_candles_matplotlib(ax, ohlc: pd.DataFrame, colors: np.ndarray):
    x = np.arange(len(ohlc))
    o = ohlc["open"].values
    h = ohlc["high"].values
    l = ohlc["low"].values
    c = ohlc["close"].values

    span = float(np.nanmax(h) - np.nanmin(l)) if len(ohlc) else 0.0
    min_body = span * 0.002 if np.isfinite(span) and span > 0 else 1e-6

    width = 0.6
    for i in range(len(ohlc)):
        ax.vlines(x[i], l[i], h[i], linewidth=1, color=colors[i], alpha=0.9)
        y0 = min(o[i], c[i])
        body_h = abs(c[i] - o[i])
        if body_h < min_body:
            body_h = min_body
            y0 = (o[i] + c[i]) / 2 - body_h / 2
        rect = Rectangle(
            (x[i] - width / 2, y0), width, body_h,
            facecolor=colors[i], edgecolor=colors[i], alpha=0.6
        )
        ax.add_patch(rect)

    ax.set_xlim(-1, len(ohlc))
    step = max(1, len(x) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([t.strftime("%m-%d %H:%M:%S") for t in ohlc.index[::step]],
                       rotation=30, ha="right")

def plot_ultra_plotly(ser: pd.Series, bar: pd.Series, title: str):
    ser_plot = ser.dropna().copy()
    if ser_plot.empty:
        st.warning("基础序列为空（ser 为空）。")
        return

    bar_plot = bar.reindex(ser_plot.index).fillna(0.0)
    d = ser_plot.diff()
    cols = np.where(d > 0, "red", np.where(d < 0, "green", "gray"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ser_plot.index, y=ser_plot.values,
        mode="lines",
        name="IV (ATM±n)",
        line=dict(width=2),
        line_shape="hv"
    ))
    fig.add_trace(go.Bar(
        x=bar_plot.index, y=bar_plot.values,
        marker_color=cols,
        name="traded vega",
        opacity=0.7
    ))
    fig.update_layout(
        title=title,
        height=650,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(title="time", rangeslider=dict(visible=False)),
        yaxis=dict(title="IV"),
        yaxis2=dict(title="traded vega", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h")
    )
    fig.update_traces(selector=dict(type="bar"), yaxis="y2")
    st.plotly_chart(fig, use_container_width=True)

def build_ohlc_figure(
    ohlc: pd.DataFrame,
    bar: pd.Series,
    colors: np.ndarray,
):
    """
    用 plotly 画：
    - 上：IV K线
    - 下：成交 vega（bar）
    """
    fig = go.Figure()

    # --- K线 ---
    fig.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name="IV OHLC",
            increasing_line_color="red",
            decreasing_line_color="green",
        )
    )

    # --- 成交 vega ---
    bar_plot = bar.reindex(ohlc.index).fillna(0.0)
    fig.add_trace(
        go.Bar(
            x=bar_plot.index,
            y=bar_plot.values,
            name="traded vega",
            yaxis="y2",
            opacity=0.6,
        )
    )

    fig.update_layout(
        height=700,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="time", rangeslider=dict(visible=False)),
        yaxis=dict(title="IV"),
        yaxis2=dict(
            title="traded vega",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h"),
    )

    return fig
def build_iv_candles_figure(ohlc: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name="IV OHLC",
            increasing_line_color="red",
            decreasing_line_color="green",
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="time", rangeslider=dict(visible=False)),
        yaxis=dict(title="IV"),
        legend=dict(orientation="h"),
    )
    return fig


def build_vega_bars_figure(bar_traded: pd.Series, bar_signed: pd.Series):
    # 对齐 index
    idx = bar_traded.index.union(bar_signed.index)
    traded = bar_traded.reindex(idx).fillna(0.0)
    signed = bar_signed.reindex(idx).fillna(0.0)

    # --- traded vega 颜色（固定蓝色，可自行改）
    traded_colors = ["steelblue"] * len(traded)

    # --- signed traded vega 颜色规则
    signed_colors = [
        "red" if v > 0 else "green" if v < 0 else "gray"
        for v in signed
    ]

    # 两行子图
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("traded vega", "signed traded vega")
    )

    # --- 上：traded vega
    fig.add_trace(
        go.Bar(
            x=idx,
            y=traded.values,
            marker_color=traded_colors,
            name="traded vega",
            opacity=0.7,
        ),
        row=1, col=1
    )

    # --- 下：signed traded vega
    fig.add_trace(
        go.Bar(
            x=idx,
            y=signed.values,
            marker_color=signed_colors,
            name="signed traded vega",
            opacity=0.7,
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )

    fig.update_yaxes(title_text="vega", row=1, col=1)
    fig.update_yaxes(title_text="signed vega", row=2, col=1)
    fig.update_xaxes(title_text="time", row=2, col=1)

    return fig

def build_iv_vega_stack_figure(ohlc: pd.DataFrame,
                              bar_traded: pd.Series,
                              bar_signed: pd.Series):
    # 对齐时间轴（用ohlc为主更直观）
    idx = ohlc.index
    traded = bar_traded.reindex(idx).fillna(0.0)
    signed = bar_signed.reindex(idx).fillna(0.0)

    # traded vega：固定颜色（你可以换成你想要的）
    traded_color = "steelblue"

    # signed traded vega：<0 绿色，>0 红色
    signed_colors = ["red" if v > 0 else "green" if v < 0 else "gray" for v in signed.values]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.70, 0.15, 0.15],
        subplot_titles=("IV", "traded vega", "signed traded vega"),
    )

    # Row1: IV candles
    fig.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name="IV OHLC",
            increasing_line_color="red",
            decreasing_line_color="green",
        ),
        row=1, col=1
    )

    # Row2: traded vega
    fig.add_trace(
        go.Bar(
            x=idx,
            y=traded.values,
            name="traded vega",
            marker_color=traded_color,
            opacity=0.7,
        ),
        row=2, col=1
    )

    # Row3: signed traded vega
    fig.add_trace(
        go.Bar(
            x=idx,
            y=signed.values,
            name="signed traded vega",
            marker_color=signed_colors,
            opacity=0.7,
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )

    # 关掉rangeslider，避免干扰
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    return fig

def build_fut_iv_vega_stack_figure(
    fut_ser: pd.Series,
    ohlc: pd.DataFrame,
    bar_traded: pd.Series,
    bar_signed: pd.Series,
):
    """
    主图（两层）：
    - Row1: IV K线 + 期货价格虚线叠加（右轴）
    - Row2: traded vega（柱色按 IV 涨跌：涨红跌绿）
    注：bar_signed 参数保留仅为兼容调用，不再绘制。
    """
    idx = ohlc.index
    # 不跨空档前向填充，避免在无行情时画长水平线
    fut = fut_ser.reindex(idx)
    traded = bar_traded.reindex(idx).fillna(0.0)

    up = (ohlc['close'].values >= ohlc['open'].values)
    bar_colors = np.where(up, 'red', 'green')

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.74, 0.26],
        subplot_titles=("IV（叠加期货虚线）[v20260303-1238]", "traded vega"),
        specs=[[{"secondary_y": True}], [{}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=idx,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name='IV',
            increasing_line_color='red',
            decreasing_line_color='green',
        ),
        row=1, col=1, secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=idx,
            y=fut.values,
            mode='lines',
            name='Futures',
            line=dict(color='royalblue', width=1.6, dash='dash'),
            opacity=0.9,
        ),
        row=1, col=1, secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=idx,
            y=traded.values,
            name='traded vega',
            marker_color=bar_colors,
            opacity=0.78,
        ),
        row=2, col=1,
    )

    fig.update_layout(height=860, margin=dict(l=40, r=36, t=40, b=30), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    fig.update_yaxes(title_text='IV', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Futures', row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text='Traded Vega', row=2, col=1)

    return fig
