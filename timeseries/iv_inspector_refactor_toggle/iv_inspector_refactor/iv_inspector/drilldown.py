# -*- coding: utf-8 -*-
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from .selection import (
    pick_contracts_atm_plus_one_otm_each_side,
    pick_contracts_atm_unique,
    pick_contracts_top_volume,
)


def make_detail_table(df_interval: pd.DataFrame, symbols: list[str], atm_pool_n: int | None = None) -> pd.DataFrame:
    if df_interval.empty or not symbols:
        return pd.DataFrame()
    g = df_interval[df_interval["symbol"].isin(symbols)].copy()
    if "dt_exch" in g.columns:
        g = g.sort_values(["dt_exch", "symbol"])
    if atm_pool_n is not None:
        g["atm_pool_n"] = int(atm_pool_n)

    cols = []
    for c in [
        "atm_pool_n",
        "dt_exch",
        "symbol",
        "underlying",
        "cp",
        "K",
        "F_used",
        "T",
        "iv",
        "mid",
        "spread",
        "bidprice1",
        "bidvol1",
        "askprice1",
        "askvol1",
        "lastprice",
        "trade_price",
        "has_trade",
        "trade_sign",
        "traded_vega",
        "traded_vega_signed",
        "d_volume",
        "trade_volume_lots",
        "volume",
        "d_totalvaluetraded",
        "totalvaluetraded",
    ]:
        if c in g.columns:
            cols.append(c)
    return g[cols].copy() if cols else g


def tables_to_excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            safe = sheet[:31]
            (df if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=safe, index=False)
    return bio.getvalue()


def pick_contracts_from_details(details_interval: pd.DataFrame) -> list[str]:
    if details_interval is None or details_interval.empty or "symbol" not in details_interval.columns:
        return []
    g = details_interval.copy()
    if "decay_mult" in g.columns:
        g = g[pd.to_numeric(g["decay_mult"], errors="coerce").fillna(0.0) > 0.0]
    if g.empty:
        return []
    return g["symbol"].dropna().astype(str).drop_duplicates().tolist()


def render_drilldown_tabs(
    *,
    df_interval: pd.DataFrame,
    details_interval: pd.DataFrame | None,
    n_main: int,
    n_alt: int,
    top_m: int,
    otm_atm_only: bool,
):
    if df_interval.empty:
        st.warning("该区间内没有原始数据。")
        return {}

    symbols_main = pick_contracts_from_details(details_interval if details_interval is not None else pd.DataFrame())
    if not symbols_main:
        symbols_main = pick_contracts_atm_unique(df_interval, n=n_main, otm_atm_only=otm_atm_only)
    symbols_alt = pick_contracts_atm_plus_one_otm_each_side(df_interval, n=len(symbols_main) if symbols_main else n_main, otm_atm_only=otm_atm_only)
    symbols_top = pick_contracts_top_volume(df_interval, m=top_m)

    tab1, tab2, tab3 = st.tabs(["ATM池", "ATM池上下各加一个OTM", "TopVol20"])

    with tab1:
        t_main = make_detail_table(df_interval, symbols_main, atm_pool_n=len(symbols_main))
        st.dataframe(t_main, use_container_width=True, height=420)
        st.write("选中的合约：", symbols_main)

    with tab2:
        t_alt = make_detail_table(df_interval, symbols_alt, atm_pool_n=len(symbols_main))
        st.dataframe(t_alt, use_container_width=True, height=420)
        st.write("选中的合约：", symbols_alt)

    with tab3:
        t_top = make_detail_table(df_interval, symbols_top)
        st.dataframe(t_top, use_container_width=True, height=420)
        st.write("选中的合约：", symbols_top)

    return {"ATM_pool": t_main, "ATM_pool_plus_otm": t_alt, "TopVol_20": t_top}
