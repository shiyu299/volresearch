# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 17:14:09 2026

@author: admin
"""

import re
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
# e.g. PL603C6800 -> underlying=PL603, cp=C, K=6800
_opt_re = re.compile(r"^(?P<under>.*?)(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)$")

def parse_option_symbol(sym: str):
    m = _opt_re.match(sym)
    if not m:
        return None
    return m.group("under"), m.group("cp"), float(m.group("strike"))


# ========= main =========
def run_pl603_iv_traded_v4(
    csv_path="pl.csv",
    underlying="PL603",
    expiry_date="2026-02-11",   # 只填日期，时间固定 15:00
    spread_limit=15.0,
    r=0.0,
    day_count=365.0,
    tz_exchange="Asia/Shanghai",
    out_path="PL603_option_iv_vega_traded_v4.csv",
):
    df = pd.read_csv(csv_path)
    df["symbol"] = df["symbol"].astype(str)

    # 1) 只取包含 PL603 的行（期货 + 期权）
    df = df[df["symbol"].str.contains(re.escape(underlying), na=False)].copy()

    # 2) 时间：timestamp 是 ns
    df["dt_utc"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
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

    # 5) 期货行：symbol == PL603
    df["is_future"] = df["symbol"].eq(underlying)

    # 期货 F：优先盘口 mid，否则 lastprice
    df["fut_price"] = np.where(
        df["is_future"] & (df["askprice1"] > 0) & (df["bidprice1"] > 0),
        df["mid"],
        np.where(df["is_future"], df["lastprice"], np.nan),
    )

    # ========= 成交识别：用累计值增量 =========
    # totalvaluetraded / volume 都按“累计”处理
    df["d_totalvaluetraded"] = df.groupby("symbol")["totalvaluetraded"].diff()
    df["d_volume"] = df.groupby("symbol")["volume"].diff()

    # 清洗 NaN/负值（累计会重置或乱跳时，至少不误判成交）
    for col in ["d_totalvaluetraded", "d_volume"]:
        df[col] = df[col].where(df[col].notna(), 0.0)
        df[col] = df[col].where(df[col] > 0, 0.0)

    # 有成交：累计成交额发生增量 + lastprice>0（只对期权）
    df["has_trade"] = (df["is_option"]) & (df["d_totalvaluetraded"] > 0) & (df["lastprice"] > 0)
    df["trade_price"] = np.where(df["has_trade"], df["lastprice"], np.nan)

    # 你定义：手数*合约乘数（lot * multiplier）
    df["trade_lot_x_mult"] = np.where(df["has_trade"], df["d_totalvaluetraded"] / df["trade_price"], 0.0)

    # 合约乘数：median(d_value/(d_volume*lastprice)) 并 round
    mask_mult = df["has_trade"] & (df["d_volume"] > 0) & (df["trade_price"] > 0)
    if mask_mult.any():
        df.loc[mask_mult, "mult_est"] = df.loc[mask_mult, "d_totalvaluetraded"] / (
            df.loc[mask_mult, "d_volume"] * df.loc[mask_mult, "trade_price"]
        )
        multiplier = float(round(float(np.nanmedian(df.loc[mask_mult, "mult_est"]))))
    else:
        multiplier = np.nan

    # 成交手数 lots：优先 d_volume，否则 lot_x_mult/multiplier 兜底
    df["trade_volume_lots"] = np.where(
        df["has_trade"] & (df["d_volume"] > 0),
        df["d_volume"],
        np.where(
            df["has_trade"] & np.isfinite(multiplier) & (multiplier > 0),
            df["trade_lot_x_mult"] / multiplier,
            np.nan,
        )
    )

    # ========= 反算 IV / Vega：按时间序列更新最新期货价 =========
    df = df.sort_values("dt_utc").reset_index(drop=True)

    exp_dt = pd.Timestamp(expiry_date).tz_localize(tz_exchange) + pd.Timedelta(hours=15)

    latest_F = np.nan
    last_iv = {}

    F_used, T_list, iv_list, vega_list = [], [], [], []
    input_price, input_type = [], []

    for _, row in df.iterrows():
        if row["is_future"]:
            latest_F = row["fut_price"]
            F_used.append(np.nan); T_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue

        if not row["is_option"]:
            F_used.append(np.nan); T_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue

        F = latest_F
        if not np.isfinite(F) or F <= 0:
            F_used.append(F); T_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue

        T = (exp_dt - row["dt_exch"]).total_seconds() / (day_count * 24 * 3600)
        sym = row["symbol"]

        # 有成交就用成交价（无视盘口宽），否则用 mid
        has_trade = bool(row["has_trade"])
        if has_trade:
            mkt_price = float(row["trade_price"])
            ptype = "trade"
        else:
            mkt_price = float(row["mid"])
            ptype = "mid"

        if T <= 0:
            iv = np.nan
        else:
            # 只有“没有成交”时，才因为 spread>limit 而跳过
            if (not has_trade) and (row["spread"] > spread_limit):
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

        F_used.append(F); T_list.append(T); iv_list.append(iv_used); vega_list.append(vega_val)
        input_price.append(mkt_price); input_type.append(ptype)

    df["F_used"] = F_used
    df["T"] = T_list
    df["iv"] = iv_list
    df["vega"] = vega_list

    # vega per 1 vol point (0.01)
    df["vega_1pct"] = df["vega"] / 100.0
    df["iv_input_type"] = input_type
    df["iv_input_price"] = input_price
    df["contract_multiplier"] = multiplier

    # ========= 成交方向 & 成交 vega（只期权）=========
    opt = df[df["is_option"]].copy().sort_values("dt_utc")
    opt["mid_lag2"] = opt.groupby("symbol")["mid"].shift(2)

    cond_trade = opt["has_trade"] & opt["mid_lag2"].notna() & opt["trade_volume_lots"].notna()
    opt["trade_sign"] = np.where(cond_trade, np.where(opt["trade_price"] > opt["mid_lag2"], 1.0, -1.0), np.nan)

    opt["traded_vega"] = np.where(cond_trade, opt["vega_1pct"] * opt["trade_volume_lots"] * multiplier, 0.0)
    opt["traded_vega_signed"] = np.where(cond_trade, opt["traded_vega"] * opt["trade_sign"], 0.0)

    # 输出列
    keep_cols = [
        "dt_exch","dt_utc","symbol","underlying","cp","K",
        "mid","mid_lag2","spread","F_used","T",
        "iv","vega","vega_1pct",
        "iv_input_type","iv_input_price",
        "has_trade","trade_price",
        "d_totalvaluetraded","d_volume","trade_lot_x_mult","trade_volume_lots",
        "contract_multiplier","trade_sign","traded_vega","traded_vega_signed",
        "bidprice1","askprice1","lastprice","totalvaluetraded","volume"
    ]
    for c in ["dt_exch", "dt_utc"]:
        if c in opt.columns:
            opt[c] = pd.to_datetime(opt[c], errors="coerce")
            if getattr(opt[c].dt, "tz", None) is not None:
                opt[c] = opt[c].dt.tz_convert("UTC").dt.tz_localize(None)
            opt[c] = opt[c].dt.floor("us")
    print(opt["dt_exch"].dtype, opt["dt_utc"].dtype)
    opt[keep_cols].to_parquet(out_path, index=False, engine="pyarrow")
    return opt


if __name__ == "__main__":
    run_pl603_iv_traded_v4(
        csv_path="pl.csv",
        underlying="PL603",
        expiry_date="2026-02-11",
        spread_limit=15.0,
        r=0.0,
        day_count=365.0,
        tz_exchange="Asia/Shanghai",
        out_path="PL603_option_iv_vega_traded_v4.parquet",
    )
    print("done -> PL603_option_iv_vega_traded_v4.parquet")