"""Load real data from volresearch and build 1min datasets for IV forecasting.

Outputs:
- data/real/mainpool_1m.parquet
- data/real/top3_contract_1m.parquet

新增：期货单边市过滤
- 若某分钟内期货不存在双边报价（buy/sell一侧缺失），该分钟全部剔除：
  - 不参与期权池化
  - 不参与训练/评估
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


REQ_COLS = [
    "dt_exch", "symbol", "is_option", "is_future", "F_used", "iv", "vega",
    "traded_vega", "traded_vega_signed", "d_volume", "volume", "spread", "cp", "K",
    "bidprice1", "askprice1",
]


def _pick_pool_symbols(meta: pd.DataFrame, f_now: float, n: int, otm_atm_only: bool = True) -> list[str]:
    if meta.empty or not np.isfinite(f_now):
        return []
    ks = meta["K"].astype(float).to_numpy()
    atm_k = float(meta.iloc[np.argmin(np.abs(ks - f_now))]["K"])
    atm_rows = meta[meta["K"].astype(float) == atm_k]
    picked = list(atm_rows["symbol"].astype(str).unique())
    if len(picked) >= n:
        return picked[:n]

    rest = meta[~meta["symbol"].isin(picked)].copy()
    if otm_atm_only and "cp" in rest.columns:
        cp = rest["cp"].astype(str).str.upper()
        is_call = cp.str.startswith("C")
        is_put = cp.str.startswith("P")
        k = rest["K"].astype(float)
        rest = rest[((is_call) & (k >= f_now)) | ((is_put) & (k <= f_now))]

    rest["dist"] = (rest["K"].astype(float) - f_now).abs()
    rest = rest.sort_values(["dist", "K"])
    for sym in rest["symbol"].astype(str).tolist():
        if sym not in picked:
            picked.append(sym)
        if len(picked) >= n:
            break
    return picked


def _grace_weight(elapsed_minutes: float) -> float:
    if not np.isfinite(elapsed_minutes):
        return 0.0
    if elapsed_minutes >= 5.0:
        return 0.0
    if elapsed_minutes >= 4.0:
        return 0.25
    if elapsed_minutes >= 2.0:
        return 0.5
    return 1.0


def _build_mainpool_1m(raw: pd.DataFrame, atm_n: int) -> pd.DataFrame:
    opt = raw[raw["is_option"] == True].copy()
    opt = opt.dropna(subset=["F_used", "K"]).copy()
    if opt.empty:
        return pd.DataFrame(columns=["dt_exch", "iv_pool", "vega_signed_1m", "vega_abs_1m", "spread_pool_1m", "F_used"])

    opt["m"] = opt["dt_exch"].dt.floor("1min")
    meta_cols = ["symbol", "K"]
    if "cp" in opt.columns:
        meta_cols.append("cp")
    meta = opt[meta_cols].drop_duplicates(subset=["symbol"]).copy()

    last_quote = (
        opt.sort_values("dt_exch")
        .groupby(["m", "symbol"], as_index=False)
        .tail(1)
        .set_index(["m", "symbol"])
    )

    f_series = (
        raw[["dt_exch", "F_used"]]
        .dropna()
        .assign(m=lambda d: d["dt_exch"].dt.floor("1min"))
        .groupby("m")["F_used"]
        .last()
        .sort_index()
    )
    minute_index = pd.DatetimeIndex(sorted(opt["m"].dropna().unique()))
    f_series = f_series.reindex(minute_index).ffill()

    rows = []
    pool_syms: list[str] = []
    pool_exit_since: dict[str, pd.Timestamp] = {}

    for bt in minute_index:
        f_now = float(f_series.loc[bt]) if bt in f_series.index else np.nan
        if not np.isfinite(f_now):
            continue

        prev_pool_set = set(pool_syms)
        pool_syms = _pick_pool_symbols(meta, f_now, int(atm_n), True)
        cur_pool_set = set(pool_syms)

        for sym in cur_pool_set:
            pool_exit_since.pop(sym, None)
        for sym in prev_pool_set - cur_pool_set:
            if sym not in pool_exit_since:
                pool_exit_since[sym] = pd.Timestamp(bt)

        expired = []
        for sym, exit_ts in pool_exit_since.items():
            if _grace_weight((pd.Timestamp(bt) - pd.Timestamp(exit_ts)).total_seconds() / 60.0) <= 0.0:
                expired.append(sym)
        for sym in expired:
            pool_exit_since.pop(sym, None)

        active_syms = list(pool_syms)
        for sym, exit_ts in pool_exit_since.items():
            mult = _grace_weight((pd.Timestamp(bt) - pd.Timestamp(exit_ts)).total_seconds() / 60.0)
            if mult > 0.0 and sym not in active_syms:
                active_syms.append(sym)

        used_iv = []
        used_w = []
        used_spread = []
        flow_sum = 0.0
        flow_abs = 0.0
        for sym in active_syms:
            key = (bt, sym)
            if key not in last_quote.index:
                continue
            row = last_quote.loc[key]
            iv_val = float(row.get("iv", np.nan))
            vega_val = float(row.get("vega", np.nan))
            decay_mult = 1.0 if sym in pool_syms else _grace_weight((pd.Timestamp(bt) - pd.Timestamp(pool_exit_since.get(sym))).total_seconds() / 60.0)
            if not np.isfinite(iv_val) or iv_val <= 0 or not np.isfinite(vega_val) or vega_val <= 0 or decay_mult <= 0.0:
                continue
            used_iv.append(iv_val)
            used_w.append(vega_val * decay_mult)
            used_spread.append(float(row.get("spread", np.nan)))
            flow_val = float(row.get("traded_vega_signed", 0.0))
            if np.isfinite(flow_val):
                flow_sum += flow_val
                flow_abs += abs(flow_val)

        if len(used_iv) == 0:
            continue
        w = np.asarray(used_w, dtype=float)
        iv_pool = float(np.average(np.asarray(used_iv, dtype=float), weights=w)) if np.nansum(w) > 1e-12 else float(np.mean(used_iv))
        rows.append(
            {
                "dt_exch": pd.Timestamp(bt),
                "iv_pool": iv_pool,
                "vega_signed_1m": float(flow_sum),
                "vega_abs_1m": float(flow_abs),
                "spread_pool_1m": float(np.nanmean(used_spread)) if len(used_spread) else np.nan,
                "F_used": float(f_now),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="/Users/shiyu/.openclaw/workspace/volresearch/data/derived")
    p.add_argument("--output-dir", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real")
    p.add_argument("--topn", type=int, default=3)
    p.add_argument("--atm-n", type=int, default=20)
    p.add_argument("--enforce-two-sided-future", action="store_true", default=True)
    return p.parse_args()


def _pick_top3(df: pd.DataFrame, topn: int) -> list[str]:
    opt = df[df["is_option"] == True].copy()
    if "d_volume" in opt.columns:
        vol = opt.groupby("symbol", dropna=True)["d_volume"].sum().sort_values(ascending=False)
    else:
        vol = opt.groupby("symbol", dropna=True)["volume"].agg(lambda s: s.iloc[-1] - s.iloc[0]).sort_values(ascending=False)
    return vol.head(topn).index.tolist()


def _future_two_sided_minutes(raw: pd.DataFrame) -> pd.Series:
    """Return valid minute index where future quote is two-sided."""
    if not {"is_future", "bidprice1", "askprice1", "dt_exch"}.issubset(raw.columns):
        return pd.Series(True, index=pd.DatetimeIndex(raw["dt_exch"]).floor("1min").unique())

    fut = raw[raw["is_future"] == True].copy()
    if fut.empty:
        return pd.Series(True, index=pd.DatetimeIndex(raw["dt_exch"]).floor("1min").unique())

    fut["m"] = fut["dt_exch"].dt.floor("1min")
    good = (
        fut.groupby("m").apply(
            lambda g: bool(((g["bidprice1"].fillna(0) > 0) & (g["askprice1"].fillna(0) > 0)).any())
        )
    )
    return good


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet in {in_dir}")

    dfs = []
    for fp in files:
        dfx = pd.read_parquet(fp)
        for c in REQ_COLS:
            if c not in dfx.columns:
                dfx[c] = np.nan
        dfs.append(dfx[REQ_COLS])

    raw = pd.concat(dfs, ignore_index=True)
    raw["dt_exch"] = pd.to_datetime(raw["dt_exch"])
    raw = raw.sort_values("dt_exch")

    # --- filter one-sided future market minutes ---
    if args.enforce_two_sided_future:
        valid_min = _future_two_sided_minutes(raw)
        raw["m"] = raw["dt_exch"].dt.floor("1min")
        before = len(raw)
        raw = raw[raw["m"].map(valid_min).fillna(False)].copy()
        after = len(raw)
        print(f"[FILTER] one-sided future minutes removed: rows {before} -> {after}")

    # --- Route A: mainpool (ATM strike forced in, non-ATM OTM only, 5min exit grace decay) ---
    mainpool = _build_mainpool_1m(raw, args.atm_n)
    mainpool = mainpool.dropna(subset=["iv_pool"]).copy()
    mainpool["f_ret_1m"] = np.log(mainpool["F_used"] / mainpool["F_used"].shift(1))
    mainpool.to_parquet(out_dir / "mainpool_1m.parquet", index=False)

    # --- Route B: top3 contracts ---
    top3 = _pick_top3(raw, args.topn)
    top = raw[raw["symbol"].isin(top3)].copy()
    top1m = top.groupby([pd.Grouper(key="dt_exch", freq="1min"), "symbol"]).agg(
        iv=("iv", "last"),
        F_used=("F_used", "last"),
        vega_signed_1m=("traded_vega_signed", "sum"),
        vega_abs_1m=("traded_vega_signed", lambda s: s.fillna(0).abs().sum()),
        spread_1m=("spread", "mean"),
        d_volume_1m=("d_volume", "sum"),
    ).reset_index()
    top1m = top1m.sort_values(["symbol", "dt_exch"])
    top1m["f_ret_1m"] = top1m.groupby("symbol")["F_used"].transform(lambda s: np.log(s / s.shift(1)))
    top1m.to_parquet(out_dir / "top3_contract_1m.parquet", index=False)

    print(f"[OK] mainpool rows={len(mainpool)} -> {out_dir / 'mainpool_1m.parquet'}")
    print(f"[OK] top3 symbols={top3} rows={len(top1m)} -> {out_dir / 'top3_contract_1m.parquet'}")


if __name__ == "__main__":
    main()
