# -*- coding: utf-8 -*-
"""
PL603 期权隐波/vega/traded_vega 计算（含 delta；同时保留期货行用于 F-only 更新）

运行：
    python marketvolseries.py

输出：
    - out_path: parquet（全量）
    - out_csv_preview_path: csv（预览，默认前 3000 行）
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm


# ========= Black-76 =========
def black76_price(F, K, T, r, sigma, cp):
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return np.nan
    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    disc = np.exp(-r * T)
    if cp == "C":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif cp == "P":
        return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError("cp must be 'C' or 'P'")


def black76_vega(F, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return 0.0
    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    return F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)


def black76_delta(F, K, T, r, sigma, cp):
    """Black-76 delta w.r.t. futures price F (discounted)."""
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return np.nan
    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    disc = np.exp(-r * T)
    if cp == "C":
        return disc * norm.cdf(d1)
    elif cp == "P":
        return disc * (norm.cdf(d1) - 1.0)
    else:
        return np.nan


def implied_vol_newton(
    F, K, T, r, market_price, cp,
    sigma0=0.2, tol=1e-6, max_iter=50,
    sigma_min=1e-8, sigma_max=5.0
):
    if T <= 0 or F <= 0 or K <= 0 or market_price <= 0:
        return np.nan

    disc = np.exp(-r * T)
    if cp == "C":
        lower = disc * max(F - K, 0.0)
        upper = disc * F
    else:
        lower = disc * max(K - F, 0.0)
        upper = disc * K

    # 市场价不在理论范围：隐波不存在
    if not (lower - 1e-10 <= market_price <= upper + 1e-10):
        return np.nan

    sigma = float(np.clip(sigma0, sigma_min, sigma_max))
    for _ in range(max_iter):
        price = black76_price(F, K, T, r, sigma, cp)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        v = black76_vega(F, K, T, r, sigma)
        if v < 1e-12:
            break
        sigma = float(np.clip(sigma - diff / v, sigma_min, sigma_max))

    return sigma


# ========= symbol parse =========
_opt_re = re.compile(r"^(?P<under>.*?)(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)$")


def parse_option_symbol(sym: str):
    m = _opt_re.match(sym)
    if not m:
        return None
    return m.group("under"), m.group("cp"), float(m.group("strike"))



def estimate_tick_size(fut_df: pd.DataFrame) -> float:
    """
    从期货 bid/ask 盘口估算最小 tick。
    方法：收集 bidprice1/askprice1 的唯一价位，排序后取最小正差。
    若无法估计返回 np.nan。
    """
    if fut_df is None or fut_df.empty:
        return np.nan
    prices = []
    for c in ["bidprice1", "askprice1", "lastprice", "mid"]:
        if c in fut_df.columns:
            v = fut_df[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            if len(v):
                prices.append(v)
    if not prices:
        return np.nan
    x = np.unique(np.concatenate(prices))
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    x.sort()
    diffs = np.diff(x)
    diffs = diffs[diffs > 1e-12]
    if diffs.size == 0:
        return np.nan
    return float(diffs.min())

# ========= main =========
def _session_code(dt: pd.Timestamp) -> str:
    h = dt.hour
    m = dt.minute
    hm = h * 60 + m
    if 9 * 60 <= hm <= 11 * 60 + 30:
        return "DAY_AM"
    if 13 * 60 + 30 <= hm <= 15 * 60:
        return "DAY_PM"
    if 21 * 60 <= hm <= 23 * 60:
        return "NIGHT"
    return "OFF"


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
DERIVED_DATA_DIR = REPO_ROOT / "data" / "derived"


def _resolve_repo_path(path_like) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_pl603_iv_traded_v4(
    csv_path="data/raw/pl.csv",
    underlying="PL603",
    expiry_date="2026-02-11",   # 可填日期或日期时间；仅日期时默认 15:00:00
    spread_limit=15.0,
    r=0.0,
    day_count=365.0,
    tz_exchange="Asia/Shanghai",
    out_path="data/derived/PL603_option_iv_vega_traded_v4.parquet",
    out_csv_preview_path="data/derived/PL603_option_iv_vega_traded_v4_preview3000.csv",
    csv_preview_n=3000,
):
    csv_path = _resolve_repo_path(csv_path)
    out_path = _resolve_repo_path(out_path)
    out_csv_preview_path = _resolve_repo_path(out_csv_preview_path) if out_csv_preview_path else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_csv_preview_path is not None:
        out_csv_preview_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["symbol"] = df["symbol"].astype(str)

    # 1) 只取包含 underlying 的行（期货 + 期权）
    df = df[df["symbol"].str.contains(re.escape(underlying), na=False)].copy()

    # 2) 时间：兼容 Linux 常见 epoch（s/ms/us/ns），统一转上海时间
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    abs_med = float(np.nanmedian(np.abs(ts.values))) if np.isfinite(ts).any() else np.nan
    if np.isfinite(abs_med):
        if abs_med >= 1e17:
            unit = "ns"
        elif abs_med >= 1e14:
            unit = "us"
        elif abs_med >= 1e11:
            unit = "ms"
        else:
            unit = "s"
    else:
        unit = "ns"

    df["dt_utc"] = pd.to_datetime(ts, unit=unit, utc=True)
    df["dt_exch"] = df["dt_utc"].dt.tz_convert(tz_exchange)

    # 为了做累计差分：先按 symbol 内部排序
    df = df.sort_values(["symbol", "dt_utc"]).reset_index(drop=True)

    # 3) 盘口 mid / spread
    df["mid"] = (df["askprice1"] + df["bidprice1"]) / 2.0
    df["spread"] = df["askprice1"] - df["bidprice1"]

    # 4) 识别期权并拆字段
    parsed = df["symbol"].apply(parse_option_symbol)
    df["is_option"] = parsed.notna()
    df.loc[df["is_option"], "underlying"] = parsed[df["is_option"]].apply(lambda x: x[0])
    df.loc[df["is_option"], "cp"] = parsed[df["is_option"]].apply(lambda x: x[1])
    df.loc[df["is_option"], "K"] = parsed[df["is_option"]].apply(lambda x: x[2])

    # 5) 期货行：symbol == underlying
    df["is_future"] = df["symbol"].eq(underlying)

    # 期货 F：盘口有效时优先用“微观价格 microprice（按买一卖一量加权）”；
    # 但如果盘口价差 >= 2 个 tick，则仍用简单中价 mid（避免极端盘口/撮合噪声）
    fut_df = df[df["is_future"]].copy()
    tick = estimate_tick_size(fut_df)
    # microprice: (ask*bidvol + bid*askvol) / (bidvol+askvol)
    denom = (df["bidvol1"].fillna(0.0) + df["askvol1"].fillna(0.0))
    microprice = np.where(
        denom > 0,
        (df["askprice1"] * df["bidvol1"].fillna(0.0) + df["bidprice1"] * df["askvol1"].fillna(0.0)) / denom,
        df["mid"]
    )
    spread = df["askprice1"] - df["bidprice1"]
    use_mid_when_wide = np.isfinite(tick) and (spread >= 2.0 * tick)

    df["fut_price"] = np.where(
        df["is_future"] & (df["askprice1"] > 0) & (df["bidprice1"] > 0),
        np.where(use_mid_when_wide, df["mid"], microprice),
        np.where(df["is_future"], df["lastprice"], np.nan),
    )
    # ========= 成交识别：用累计值增量 =========
    df["d_totalvaluetraded"] = df.groupby("symbol")["totalvaluetraded"].diff()
    df["d_volume"] = df.groupby("symbol")["volume"].diff()

    for col in ["d_totalvaluetraded", "d_volume"]:
        df[col] = df[col].where(df[col].notna(), 0.0)
        df[col] = df[col].where(df[col] > 0, 0.0)

    df["has_trade"] = (df["is_option"]) & (df["d_totalvaluetraded"] > 0) & (df["lastprice"] > 0)
    df["trade_price"] = np.where(df["has_trade"], df["lastprice"], np.nan)

    df["trade_lot_x_mult"] = np.where(df["has_trade"], df["d_totalvaluetraded"] / df["trade_price"], 0.0)

    mask_mult = df["has_trade"] & (df["d_volume"] > 0) & (df["trade_price"] > 0)
    if mask_mult.any():
        df.loc[mask_mult, "mult_est"] = df.loc[mask_mult, "d_totalvaluetraded"] / (
            df.loc[mask_mult, "d_volume"] * df.loc[mask_mult, "trade_price"]
        )
        multiplier = float(round(float(np.nanmedian(df.loc[mask_mult, "mult_est"]))))
    else:
        multiplier = np.nan

    df["trade_volume_lots"] = np.where(
        df["has_trade"] & (df["d_volume"] > 0),
        df["d_volume"],
        np.where(
            df["has_trade"] & np.isfinite(multiplier) & (multiplier > 0),
            df["trade_lot_x_mult"] / multiplier,
            np.nan,
        )
    )

    # ========= 反算 IV / Vega / Delta =========
    df = df.sort_values("dt_utc").reset_index(drop=True)

    # 到期时间：支持传完整时间（如 2026-04-13 15:00:00）；若仅日期默认 15:00:00
    exp_ts = pd.Timestamp(expiry_date)
    if exp_ts.hour == 0 and exp_ts.minute == 0 and exp_ts.second == 0 and exp_ts.nanosecond == 0:
        exp_ts = exp_ts + pd.Timedelta(hours=15)
    exp_dt = exp_ts.tz_localize(tz_exchange)

    latest_F = np.nan
    last_iv = {}
    prev_sess = None

    F_used, T_list, iv_list, vega_list, delta_list = [], [], [], [], []
    input_price, input_type = [], []

    for _, row in df.iterrows():
        if row["is_future"]:
            latest_F = row["fut_price"]
            F_used.append(latest_F); T_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan); delta_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue

        if not row["is_option"]:
            F_used.append(latest_F); T_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan); delta_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue

        # 会话切换（如 15:00->21:00, 23:00->09:00）时，不跨会话继承旧 IV 状态
        cur_sess = _session_code(row["dt_exch"])
        if (prev_sess is not None) and (cur_sess != prev_sess):
            last_iv = {}
        prev_sess = cur_sess

        F = latest_F
        if not np.isfinite(F) or F <= 0:
            F_used.append(F); T_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan); delta_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue

        T = (exp_dt - row["dt_exch"]).total_seconds() / (day_count * 24 * 3600)
        sym = row["symbol"]

        has_trade = bool(row["has_trade"])
        sp = float(row["spread"]) if np.isfinite(row["spread"]) else np.nan
        bid = float(row["bidprice1"]) if np.isfinite(row["bidprice1"]) else np.nan
        ask = float(row["askprice1"]) if np.isfinite(row["askprice1"]) else np.nan

        # 盘口有效性：bid/ask>0 且 spread>=0
        quote_valid = np.isfinite(bid) and np.isfinite(ask) and (bid > 0) and (ask > 0) and np.isfinite(sp) and (sp >= 0)

        # 规则：只有“有成交且 spread>spread_limit”时才用成交价反算；否则用 mid（但 mid 需 quote_valid）
        use_trade_price = has_trade and np.isfinite(sp) and (sp > spread_limit)
        if use_trade_price:
            mkt_price = float(row["trade_price"])
            ptype = "trade"
        else:
            mkt_price = float(row["mid"]) if quote_valid else np.nan
            ptype = "mid"

        if T <= 0:
            iv = np.nan
        else:
            if (not use_trade_price) and (row["spread"] > spread_limit):
                # spread 过大且未使用成交价：认为盘口不可信，沿用上一时刻 iv
                iv = last_iv.get(sym, np.nan)
            else:
                if (not np.isfinite(mkt_price)) or (mkt_price <= 0):
                    iv = last_iv.get(sym, np.nan)
                else:
                    sigma0 = last_iv.get(sym, 0.2)
                    iv = implied_vol_newton(F, float(row["K"]), T, r, mkt_price, row["cp"], sigma0=sigma0)

        if np.isfinite(iv):
            last_iv[sym] = iv

        iv_used = last_iv.get(sym, np.nan)
        vega_val = black76_vega(F, float(row["K"]), T, r, iv_used) if np.isfinite(iv_used) else np.nan
        delta_val = black76_delta(F, float(row["K"]), T, r, iv_used, row["cp"]) if np.isfinite(iv_used) else np.nan

        F_used.append(F); T_list.append(T); iv_list.append(iv_used); vega_list.append(vega_val); delta_list.append(delta_val)
        input_price.append(mkt_price); input_type.append(ptype)

    df["F_used"] = F_used
    df["T"] = T_list
    df["iv"] = iv_list
    df["vega"] = vega_list
    df["delta"] = delta_list

    df["vega_1pct"] = df["vega"] / 100.0
    df["iv_input_type"] = input_type
    df["iv_input_price"] = input_price
    df["contract_multiplier"] = multiplier

    # ========= 成交方向 & 成交 vega（只期权）=========
    out = df[df["is_option"] | df["is_future"]].copy().sort_values("dt_utc")
    out["mid_lag2"] = out.groupby("symbol")["mid"].shift(2)

    cond_trade = out["is_option"] & out["has_trade"] & out["mid_lag2"].notna() & out["trade_volume_lots"].notna()
    out["trade_sign"] = np.where(
        cond_trade,
        np.where(out["trade_price"] > out["mid_lag2"], 1.0, -1.0),
        np.nan
    )

    out["traded_vega"] = np.where(
        cond_trade & np.isfinite(out["vega_1pct"]) & np.isfinite(out["trade_volume_lots"]) & np.isfinite(out["contract_multiplier"]),
        out["vega_1pct"] * out["trade_volume_lots"] * out["contract_multiplier"],
        0.0
    )
    out["traded_vega_signed"] = np.where(cond_trade, out["traded_vega"] * out["trade_sign"], 0.0)

    keep_cols = [
        "dt_exch","dt_utc","symbol","underlying","is_option","is_future","cp","K",
        "mid","mid_lag2","spread","fut_price","F_used","T",
        "iv","vega","vega_1pct","delta",
        "iv_input_type","iv_input_price",
        "has_trade","trade_price",
        "d_totalvaluetraded","d_volume","trade_lot_x_mult","trade_volume_lots",
        "contract_multiplier","trade_sign","traded_vega","traded_vega_signed",
        "bidprice1","askprice1","bidvol1","askvol1","lastprice","totalvaluetraded","volume"
    ]

    # 去 tz（兼容 parquet & app）
    # - dt_exch: 保留“上海本地墙上时间”（naive）
    # - dt_utc : 保留 UTC（naive）
    if "dt_exch" in out.columns:
        out["dt_exch"] = pd.to_datetime(out["dt_exch"], errors="coerce")
        if getattr(out["dt_exch"].dt, "tz", None) is not None:
            out["dt_exch"] = out["dt_exch"].dt.tz_convert(tz_exchange).dt.tz_localize(None)
        out["dt_exch"] = out["dt_exch"].dt.floor("us")

    if "dt_utc" in out.columns:
        out["dt_utc"] = pd.to_datetime(out["dt_utc"], errors="coerce")
        if getattr(out["dt_utc"].dt, "tz", None) is not None:
            out["dt_utc"] = out["dt_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        out["dt_utc"] = out["dt_utc"].dt.floor("us")

    print(out["dt_exch"].dtype, out["dt_utc"].dtype)

    # 1) 全量 parquet
    out.to_parquet(out_path, index=False, engine="pyarrow")

    # 2) 预览 csv（默认前 3000 行）
    if out_csv_preview_path:
        out[keep_cols].head(int(csv_preview_n)).to_csv(out_csv_preview_path, index=False, encoding="utf-8-sig")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build modified-v4 option/futures dataset.")
    parser.add_argument("--csv-path", default=str(RAW_DATA_DIR / "PL26.csv"))
    parser.add_argument("--underlying", default="PL605")
    parser.add_argument("--expiry-date", default="2026-04-13")
    parser.add_argument("--spread-limit", type=float, default=25.0)
    parser.add_argument("--r", type=float, default=0.0)
    parser.add_argument("--day-count", type=float, default=365.0)
    parser.add_argument("--tz-exchange", default="Asia/Shanghai")
    parser.add_argument("--out-path", default=str(DERIVED_DATA_DIR / "PL60526.parquet"))
    parser.add_argument("--out-csv-preview-path", default=str(DERIVED_DATA_DIR / "PL60526preview5000.csv"))
    parser.add_argument("--csv-preview-n", type=int, default=5000)
    args = parser.parse_args()

    out = run_pl603_iv_traded_v4(
        csv_path=args.csv_path,
        underlying=args.underlying,
        expiry_date=args.expiry_date,
        spread_limit=args.spread_limit,
        r=args.r,
        day_count=args.day_count,
        tz_exchange=args.tz_exchange,
        out_path=args.out_path,
        out_csv_preview_path=args.out_csv_preview_path,
        csv_preview_n=args.csv_preview_n,
    )
    print("done ->" + str(args.out_path) + f" ({len(out)} rows)")
