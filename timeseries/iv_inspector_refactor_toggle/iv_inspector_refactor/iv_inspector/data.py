# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import pandas as pd
import streamlit as st


# App会用到的核心列（CSV读取时优先只读这些，显著提速/降内存）
CORE_COLS = {
    "timestamp", "dt_exch", "symbol", "underlying", "cp", "K",
    "is_option", "is_future",
    "F_used", "iv", "vega", "vega_1pct", "delta",
    "traded_vega", "traded_vega_signed",
    "bidprice1", "askprice1", "bidvol1", "askvol1",
    "trade_price", "lastprice",
    "d_volume", "volume", "trade_volume_lots",
    "d_totalvaluetraded", "totalvaluetraded",
    "mid", "spread", "has_trade", "trade_sign", "T",
}


@st.cache_data
def list_data_files(base_dir: str = "data") -> list[str]:
    """List candidate data files under base_dir (parquet preferred)."""
    base = Path(base_dir).expanduser()
    if not base.exists() or not base.is_dir():
        return []

    repo_root = Path.cwd().resolve()
    patterns = ["*.parquet", "*.pq", "*.feather", "*.csv.gz", "*.csv"]

    files = []
    for pat in patterns:
        for fp in base.rglob(pat):
            if not fp.is_file():
                continue
            fp_resolved = fp.resolve()
            try:
                show = str(fp_resolved.relative_to(repo_root)) if repo_root in fp_resolved.parents else str(fp_resolved)
            except Exception:
                show = str(fp_resolved)
            files.append(show)

    # 去重并保持“前缀模式优先级”
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _read_csv_fast(path: str) -> pd.DataFrame:
    """Fast-ish CSV loader with graceful fallback."""
    # 先尝试 pyarrow 引擎（通常更快）
    try:
        cols = None
        # 先读表头决定可读列，避免无效usecols报错
        head = pd.read_csv(path, nrows=0)
        available = set(head.columns)
        picked = [c for c in CORE_COLS if c in available]
        if picked:
            cols = picked

        return pd.read_csv(path, engine="pyarrow", usecols=cols)
    except Exception:
        # 回退到默认引擎
        try:
            head = pd.read_csv(path, nrows=0)
            available = set(head.columns)
            picked = [c for c in CORE_COLS if c in available]
            cols = picked if picked else None
            return pd.read_csv(path, usecols=cols, low_memory=False)
        except Exception:
            return pd.read_csv(path, low_memory=False)


def load_data(path: str) -> pd.DataFrame:
    path = (path or "").strip().strip('"').strip("'")
    if not path:
        return pd.DataFrame()

    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        df = pd.read_parquet(path)
    elif lower.endswith(".feather"):
        df = pd.read_feather(path)
    elif lower.endswith(".csv") or lower.endswith(".csv.gz"):
        df = _read_csv_fast(path)
    else:
        df = _read_csv_fast(path)

    # 时间列兼容：timestamp(ns) 或 dt_exch
    if "dt_exch" not in df.columns:
        if "timestamp" in df.columns:
            # 兼容 Linux epoch（s/ms/us/ns），并统一转到上海时间
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
        else:
            raise ValueError("数据中缺少 dt_exch 列（且无 timestamp 可回退）")

    df["dt_exch"] = pd.to_datetime(df["dt_exch"], errors="coerce")
    df = df.dropna(subset=["dt_exch"]).copy()

    for c in [
        "F_used", "K", "iv", "vega", "vega_1pct",
        "traded_vega", "traded_vega_signed",
        "bidprice1", "askprice1",
        "trade_price", "d_volume", "volume", "trade_volume_lots",
        "d_totalvaluetraded", "totalvaluetraded",
        "delta", "T",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 自动补充 mid / traded_vega_signed（如果缺失）----
    if "mid" not in df.columns and ("bidprice1" in df.columns) and ("askprice1" in df.columns):
        df["mid"] = (df["bidprice1"] + df["askprice1"]) / 2.0

    if (
        ("traded_vega_signed" not in df.columns)
        and ("traded_vega" in df.columns)
        and ("trade_price" in df.columns)
        and ("mid" in df.columns)
        and ("symbol" in df.columns)
    ):
        df = df.sort_values(["symbol", "dt_exch"]).copy()
        df["mid_prev"] = df.groupby("symbol")["mid"].shift(1)

        # 规则：trade_price > 上一个mid => +1； < => -1；否则 0
        diff = df["trade_price"] - df["mid_prev"]
        df["trade_sign"] = np.where(diff > 0, 1, np.where(diff < 0, -1, 0)).astype(float)

        df["traded_vega_signed"] = df["traded_vega"] * df["trade_sign"]
        df = df.drop(columns=["mid_prev", "trade_sign"], errors="ignore")

    # 低基数列转category，降内存
    for c in ["symbol", "cp", "underlying"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype("category")

    df = df.sort_values("dt_exch").reset_index(drop=True)
    return df
