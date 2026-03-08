# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    path = (path or "").strip().strip('"').strip("'")
    if not path:
        return pd.DataFrame()

    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        df = pd.read_parquet(path)
    elif lower.endswith(".feather"):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)

    if "dt_exch" not in df.columns:
        raise ValueError("数据中缺少 dt_exch 列")

    df["dt_exch"] = pd.to_datetime(df["dt_exch"], errors="coerce")
    df = df.dropna(subset=["dt_exch"]).copy()

    for c in [
        "F_used", "K", "iv", "vega", "vega_1pct",
        "traded_vega", "traded_vega_signed",
        "bidprice1", "askprice1",
        "trade_price", "d_volume", "volume", "trade_volume_lots",
        "d_totalvaluetraded", "totalvaluetraded",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("dt_exch").reset_index(drop=True)
    return df
