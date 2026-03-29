import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_fut_iv_vega_stack_figure(
    fut_ser: pd.Series,
    ohlc: pd.DataFrame,
    bar_traded: pd.Series,
    bar_signed: pd.Series,
):
    idx = ohlc.index
    fut = fut_ser.reindex(idx)
    traded = bar_traded.reindex(idx).fillna(0.0)

    up = ohlc["close"].values >= ohlc["open"].values
    bar_colors = np.where(up, "red", "green")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.74, 0.26],
        subplot_titles=("IV + Futures", "Traded Vega"),
        specs=[[{"secondary_y": True}], [{}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=idx,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name="IV",
            increasing_line_color="red",
            decreasing_line_color="green",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=fut.values,
            mode="lines",
            name="Futures",
            line=dict(color="royalblue", width=1.6, dash="dash"),
            opacity=0.9,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=idx,
            y=traded.values,
            name="traded vega",
            marker_color=bar_colors,
            opacity=0.78,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=860, margin=dict(l=40, r=36, t=40, b=30), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text="IV", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Futures", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Traded Vega", row=2, col=1)
    return fig

