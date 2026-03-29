from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


SIMPLE_ROOT = Path(__file__).resolve().parents[1]

CORE_COLS = {
    "timestamp",
    "dt_exch",
    "symbol",
    "underlying",
    "cp",
    "K",
    "is_option",
    "is_future",
    "F_used",
    "iv",
    "vega",
    "vega_1pct",
    "delta",
    "traded_vega",
    "traded_vega_signed",
    "bidprice1",
    "askprice1",
    "bidvol1",
    "askvol1",
    "trade_price",
    "lastprice",
    "d_volume",
    "volume",
    "trade_volume_lots",
    "d_totalvaluetraded",
    "totalvaluetraded",
    "mid",
    "spread",
    "has_trade",
    "trade_sign",
    "T",
}


def _resolve_local_path(path_like: str) -> Path:
    path = Path((path_like or "").strip().strip('"').strip("'")).expanduser()
    return path if path.is_absolute() else (SIMPLE_ROOT / path)


@st.cache_data
def list_data_files(base_dir: str = "data/derived") -> list[str]:
    base = _resolve_local_path(base_dir)
    if not base.exists() or not base.is_dir():
        return []

    files = []
    for pat in ["*.parquet", "*.pq", "*.feather", "*.csv.gz", "*.csv"]:
        for fp in base.rglob(pat):
            if fp.is_file():
                files.append(str(fp.resolve()))
    return list(dict.fromkeys(files))


def _read_csv_fast(path: str) -> pd.DataFrame:
    try:
        head = pd.read_csv(path, nrows=0)
        available = set(head.columns)
        cols = [c for c in CORE_COLS if c in available] or None
        return pd.read_csv(path, engine="pyarrow", usecols=cols)
    except Exception:
        try:
            head = pd.read_csv(path, nrows=0)
            available = set(head.columns)
            cols = [c for c in CORE_COLS if c in available] or None
            return pd.read_csv(path, usecols=cols, low_memory=False)
        except Exception:
            return pd.read_csv(path, low_memory=False)


def load_data(path: str) -> pd.DataFrame:
    resolved = _resolve_local_path(path)
    lower = str(resolved).lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        df = pd.read_parquet(resolved)
    elif lower.endswith(".feather"):
        df = pd.read_feather(resolved)
    else:
        df = _read_csv_fast(str(resolved))

    if "dt_exch" not in df.columns:
        if "timestamp" not in df.columns:
            raise ValueError("Missing dt_exch and timestamp columns.")
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().any():
            abs_med = float(ts.abs().median())
            if abs_med >= 1e17:
                unit = "ns"
            elif abs_med >= 1e14:
                unit = "us"
            elif abs_med >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            dt = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
            df["dt_exch"] = dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        else:
            df["dt_exch"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["dt_exch"] = pd.to_datetime(df["dt_exch"], errors="coerce")
    df = df.dropna(subset=["dt_exch"]).copy()

    for c in ["F_used", "K", "iv", "vega", "vega_1pct", "traded_vega", "traded_vega_signed", "bidprice1", "askprice1", "trade_price", "d_volume", "volume", "trade_volume_lots", "d_totalvaluetraded", "totalvaluetraded", "delta", "T"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "mid" not in df.columns and {"bidprice1", "askprice1"}.issubset(df.columns):
        df["mid"] = (df["bidprice1"] + df["askprice1"]) / 2.0

    if "traded_vega_signed" not in df.columns and "traded_vega" in df.columns and "trade_price" in df.columns and "mid" in df.columns and "symbol" in df.columns:
        df = df.sort_values(["symbol", "dt_exch"]).copy()
        df["mid_prev"] = df.groupby("symbol")["mid"].shift(1)
        diff = df["trade_price"] - df["mid_prev"]
        df["trade_sign"] = np.where(diff > 0, 1, np.where(diff < 0, -1, 0)).astype(float)
        df["traded_vega_signed"] = df["traded_vega"] * df["trade_sign"]
        df = df.drop(columns=["mid_prev", "trade_sign"], errors="ignore")

    for c in ["symbol", "cp", "underlying"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype("category")

    return df.sort_values("dt_exch").reset_index(drop=True)
