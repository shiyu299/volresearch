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

    # --- Route A: mainpool (ATM proxied by near-F strikes) ---
    opt = raw[raw["is_option"] == True].copy()
    opt = opt.dropna(subset=["iv", "F_used", "K"])  # keep calculable rows
    opt["dist"] = (opt["K"] - opt["F_used"]).abs()
    opt = opt.sort_values(["dt_exch", "dist"]).groupby("dt_exch", as_index=False).head(args.atm_n)

    def _safe_weighted_iv(g: pd.DataFrame) -> float:
        w = g["vega"].fillna(0).clip(lower=0)
        if w.sum() <= 1e-12:
            return float(g["iv"].mean())
        return float(np.average(g["iv"], weights=w))

    mainpool = opt.groupby(pd.Grouper(key="dt_exch", freq="1min")).apply(
        lambda g: pd.Series({
            "iv_pool": _safe_weighted_iv(g) if len(g) else np.nan,
            "vega_signed_1m": g["traded_vega_signed"].fillna(0).sum(),
            "vega_abs_1m": g["traded_vega_signed"].fillna(0).abs().sum(),
            "spread_pool_1m": g["spread"].mean(),
            "F_used": g["F_used"].dropna().iloc[-1] if g["F_used"].notna().any() else np.nan,
        })
    ).reset_index()
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
