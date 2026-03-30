from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


SIMPLE_ROOT = Path(__file__).resolve().parent
_OPT_RE = re.compile(r"^(?P<under>[A-Z]+\d+)(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)$")


def _resolve_local_path(path_like) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (SIMPLE_ROOT / path)


def normalize_contract_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(symbol).upper())


def parse_option_symbol(symbol: str):
    m = _OPT_RE.match(normalize_contract_symbol(symbol))
    if not m:
        return None
    return m.group("under"), m.group("cp"), float(m.group("strike"))


def black76_price(f, k, t, r, sigma, cp):
    if not np.isfinite(f) or not np.isfinite(k) or not np.isfinite(t) or not np.isfinite(sigma) or f <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return np.nan
    srt = sigma * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * sigma * sigma * t) / srt
    d2 = d1 - srt
    disc = np.exp(-r * t)
    if str(cp).upper().startswith("C"):
        return disc * (f * norm.cdf(d1) - k * norm.cdf(d2))
    return disc * (k * norm.cdf(-d2) - f * norm.cdf(-d1))


def black76_vega(f, k, t, r, sigma):
    if not np.isfinite(f) or not np.isfinite(k) or not np.isfinite(t) or not np.isfinite(sigma) or f <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return np.nan
    srt = sigma * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * sigma * sigma * t) / srt
    return np.exp(-r * t) * f * norm.pdf(d1) * np.sqrt(t)


def black76_delta(f, k, t, r, sigma, cp):
    if not np.isfinite(f) or not np.isfinite(k) or not np.isfinite(t) or not np.isfinite(sigma) or f <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return np.nan
    srt = sigma * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * sigma * sigma * t) / srt
    disc = np.exp(-r * t)
    if str(cp).upper().startswith("C"):
        return disc * norm.cdf(d1)
    return disc * (norm.cdf(d1) - 1.0)


def implied_vol_newton(f, k, t, r, price, cp, sigma0=0.2, max_iter=100, tol=1e-8):
    if not np.isfinite(price) or price <= 0 or not np.isfinite(f) or not np.isfinite(k) or not np.isfinite(t) or f <= 0 or k <= 0 or t <= 0:
        return np.nan
    sigma = max(float(sigma0), 1e-4)
    for _ in range(max_iter):
        px = black76_price(f, k, t, r, sigma, cp)
        vg = black76_vega(f, k, t, r, sigma)
        if not np.isfinite(px) or not np.isfinite(vg) or vg <= 1e-12:
            break
        step = (px - price) / vg
        sigma_new = min(max(sigma - step, 1e-4), 5.0)
        if abs(sigma_new - sigma) < tol:
            sigma = sigma_new
            break
        sigma = sigma_new
    return sigma if np.isfinite(sigma) else np.nan


def estimate_tick_size(fut_df: pd.DataFrame) -> float:
    if fut_df is None or fut_df.empty or "lastprice" not in fut_df.columns:
        return np.nan
    px = pd.to_numeric(fut_df["lastprice"], errors="coerce").dropna().sort_values().unique()
    if len(px) < 2:
        return np.nan
    diffs = np.diff(px)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.min(diffs)) if len(diffs) else np.nan


def _session_code(dt: pd.Timestamp) -> str:
    hm = dt.hour * 60 + dt.minute
    if 9 * 60 <= hm <= 11 * 60 + 30:
        return "DAY_AM"
    if 13 * 60 + 30 <= hm <= 15 * 60:
        return "DAY_PM"
    if 21 * 60 <= hm <= 23 * 60:
        return "NIGHT"
    return "OFF"


def run_pl603_iv_traded_v4(
    csv_path="data/raw/pl.csv",
    underlying="PL603",
    expiry_date="2026-02-11",
    spread_limit=15.0,
    r=0.0,
    day_count=365.0,
    tz_exchange="Asia/Shanghai",
    out_path="data/derived/PL603_option_iv_vega_traded_v4.parquet",
    out_csv_preview_path=None,
    csv_preview_n=3000,
):
    csv_path = _resolve_local_path(csv_path)
    out_path = _resolve_local_path(out_path)
    out_csv_preview_path = _resolve_local_path(out_csv_preview_path) if out_csv_preview_path else None
    normalized_underlying = normalize_contract_symbol(underlying)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_csv_preview_path is not None:
        out_csv_preview_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["symbol"] = df["symbol"].astype(str)
    df["symbol_normalized"] = df["symbol"].map(normalize_contract_symbol)
    df = df[df["symbol_normalized"].str.contains(re.escape(normalized_underlying), na=False)].copy()

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
    df = df.sort_values(["symbol", "dt_utc"]).reset_index(drop=True)

    df["mid"] = (df["askprice1"] + df["bidprice1"]) / 2.0
    df["spread"] = df["askprice1"] - df["bidprice1"]
    parsed = df["symbol"].apply(parse_option_symbol)
    df["is_option"] = parsed.notna()
    df.loc[df["is_option"], "underlying"] = parsed[df["is_option"]].apply(lambda x: x[0])
    df.loc[df["is_option"], "cp"] = parsed[df["is_option"]].apply(lambda x: x[1])
    df.loc[df["is_option"], "K"] = parsed[df["is_option"]].apply(lambda x: x[2])
    df["is_future"] = df["symbol_normalized"].eq(normalized_underlying)

    fut_df = df[df["is_future"]].copy()
    tick = estimate_tick_size(fut_df)
    denom = df["bidvol1"].fillna(0.0) + df["askvol1"].fillna(0.0)
    microprice = np.where(
        denom > 0,
        (df["askprice1"] * df["bidvol1"].fillna(0.0) + df["bidprice1"] * df["askvol1"].fillna(0.0)) / denom,
        df["mid"],
    )
    use_mid_when_wide = np.isfinite(tick) and (df["spread"] >= 2.0 * tick)
    df["fut_price"] = np.where(
        df["is_future"] & (df["askprice1"] > 0) & (df["bidprice1"] > 0),
        np.where(use_mid_when_wide, df["mid"], microprice),
        np.where(df["is_future"], df["lastprice"], np.nan),
    )

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
        df.loc[mask_mult, "mult_est"] = df.loc[mask_mult, "d_totalvaluetraded"] / (df.loc[mask_mult, "d_volume"] * df.loc[mask_mult, "trade_price"])
        multiplier = float(round(float(np.nanmedian(df.loc[mask_mult, "mult_est"]))))
    else:
        multiplier = np.nan

    df["trade_volume_lots"] = np.where(
        df["has_trade"] & (df["d_volume"] > 0),
        df["d_volume"],
        np.where(df["has_trade"] & np.isfinite(multiplier) & (multiplier > 0), df["trade_lot_x_mult"] / multiplier, np.nan),
    )

    df = df.sort_values("dt_utc").reset_index(drop=True)
    exp_ts = pd.Timestamp(expiry_date)
    if exp_ts.hour == 0 and exp_ts.minute == 0 and exp_ts.second == 0 and exp_ts.nanosecond == 0:
        exp_ts = exp_ts + pd.Timedelta(hours=15)
    exp_dt = exp_ts.tz_localize(tz_exchange)

    latest_f = np.nan
    last_iv = {}
    prev_sess = None
    f_used, t_list, iv_list, vega_list, delta_list = [], [], [], [], []
    input_price, input_type = [], []

    for _, row in df.iterrows():
        if row["is_future"]:
            latest_f = row["fut_price"]
            f_used.append(latest_f); t_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan); delta_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue
        if not row["is_option"]:
            f_used.append(latest_f); t_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan); delta_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue

        cur_sess = _session_code(row["dt_exch"])
        if prev_sess is not None and cur_sess != prev_sess:
            last_iv = {}
        prev_sess = cur_sess

        f_value = latest_f
        if not np.isfinite(f_value) or f_value <= 0:
            f_used.append(f_value); t_list.append(np.nan); iv_list.append(np.nan); vega_list.append(np.nan); delta_list.append(np.nan)
            input_price.append(np.nan); input_type.append(None)
            continue

        t = (exp_dt - row["dt_exch"]).total_seconds() / (day_count * 24 * 3600)
        sym = row["symbol"]
        use_trade_price = bool(row["has_trade"]) and np.isfinite(row["spread"]) and (row["spread"] > spread_limit)
        quote_valid = np.isfinite(row["bidprice1"]) and np.isfinite(row["askprice1"]) and (row["bidprice1"] > 0) and (row["askprice1"] > 0) and np.isfinite(row["spread"]) and (row["spread"] >= 0)
        if use_trade_price:
            mkt_price = float(row["trade_price"])
            ptype = "trade"
        else:
            mkt_price = float(row["mid"]) if quote_valid else np.nan
            ptype = "mid"

        if t <= 0:
            iv = np.nan
        elif (not use_trade_price) and (row["spread"] > spread_limit):
            iv = last_iv.get(sym, np.nan)
        elif (not np.isfinite(mkt_price)) or (mkt_price <= 0):
            iv = last_iv.get(sym, np.nan)
        else:
            iv = implied_vol_newton(f_value, float(row["K"]), t, r, mkt_price, row["cp"], sigma0=last_iv.get(sym, 0.2))

        if np.isfinite(iv):
            last_iv[sym] = iv

        iv_used = last_iv.get(sym, np.nan)
        vega_val = black76_vega(f_value, float(row["K"]), t, r, iv_used) if np.isfinite(iv_used) else np.nan
        delta_val = black76_delta(f_value, float(row["K"]), t, r, iv_used, row["cp"]) if np.isfinite(iv_used) else np.nan

        f_used.append(f_value); t_list.append(t); iv_list.append(iv_used); vega_list.append(vega_val); delta_list.append(delta_val)
        input_price.append(mkt_price); input_type.append(ptype)

    df["F_used"] = f_used
    df["T"] = t_list
    df["iv"] = iv_list
    df["vega"] = vega_list
    df["delta"] = delta_list
    df["vega_1pct"] = df["vega"] / 100.0
    df["iv_input_type"] = input_type
    df["iv_input_price"] = input_price
    df["contract_multiplier"] = multiplier

    out = df[df["is_option"] | df["is_future"]].copy().sort_values("dt_utc")
    out["mid_lag2"] = out.groupby("symbol")["mid"].shift(2)
    cond_trade = out["is_option"] & out["has_trade"] & out["mid_lag2"].notna() & out["trade_volume_lots"].notna()
    out["trade_sign"] = np.where(cond_trade, np.where(out["trade_price"] > out["mid_lag2"], 1.0, -1.0), np.nan)
    out["traded_vega"] = np.where(
        cond_trade & np.isfinite(out["vega_1pct"]) & np.isfinite(out["trade_volume_lots"]) & np.isfinite(out["contract_multiplier"]),
        out["vega_1pct"] * out["trade_volume_lots"] * out["contract_multiplier"],
        0.0,
    )
    out["traded_vega_signed"] = np.where(cond_trade, out["traded_vega"] * out["trade_sign"], 0.0)

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

    out.to_parquet(out_path, index=False, engine="pyarrow")
    if out_csv_preview_path:
        out.head(int(csv_preview_n)).to_csv(out_csv_preview_path, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run modified-v4 option/futures IV calculation.")
    parser.add_argument("--csv-path", default="data/raw/pl.csv")
    parser.add_argument("--underlying", default="PL603")
    parser.add_argument("--expiry-date", default="2026-02-11")
    parser.add_argument("--spread-limit", type=float, default=15.0)
    parser.add_argument("--r", type=float, default=0.0)
    parser.add_argument("--day-count", type=float, default=365.0)
    parser.add_argument("--tz-exchange", default="Asia/Shanghai")
    parser.add_argument("--out-path", default="data/derived/PL603_option_iv_vega_traded_v4.parquet")
    parser.add_argument("--out-csv-preview-path", default=None)
    parser.add_argument("--csv-preview-n", type=int, default=3000)
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
    print("done ->" + str(_resolve_local_path(args.out_path)) + f" ({len(out)} rows)")
