import io

import pandas as pd
import streamlit as st

from .selection import pick_contracts_atm_unique, pick_contracts_top_volume


def make_detail_table(df_interval: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if df_interval.empty or not symbols:
        return pd.DataFrame()
    g = df_interval[df_interval["symbol"].isin(symbols)].copy()
    if "dt_exch" in g.columns:
        g = g.sort_values(["dt_exch", "symbol"])

    cols = []
    for c in [
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
            (df if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=sheet[:31], index=False)
    return bio.getvalue()


def render_drilldown_tabs(*, df_interval: pd.DataFrame, n_main: int, n_alt: int, top_m: int, otm_atm_only: bool):
    if df_interval.empty:
        st.warning("No raw rows found in the selected interval.")
        return {}

    symbols_main = pick_contracts_atm_unique(df_interval, n=n_main, otm_atm_only=otm_atm_only)
    symbols_alt = pick_contracts_atm_unique(df_interval, n=n_alt, otm_atm_only=otm_atm_only)
    symbols_top = pick_contracts_top_volume(df_interval, m=top_m)

    tab1, tab2, tab3 = st.tabs([f"ATM {n_main}", f"ATM {n_alt}", f"TopVol {top_m}"])
    with tab1:
        t_main = make_detail_table(df_interval, symbols_main)
        st.dataframe(t_main, use_container_width=True, height=420)
        st.write("Selected contracts:", symbols_main)
    with tab2:
        t_alt = make_detail_table(df_interval, symbols_alt)
        st.dataframe(t_alt, use_container_width=True, height=420)
        st.write("Selected contracts:", symbols_alt)
    with tab3:
        t_top = make_detail_table(df_interval, symbols_top)
        st.dataframe(t_top, use_container_width=True, height=420)
        st.write("Selected contracts:", symbols_top)

    return {f"ATM_{n_main}": t_main, f"ATM_{n_alt}": t_alt, f"TopVol_{top_m}": t_top}

