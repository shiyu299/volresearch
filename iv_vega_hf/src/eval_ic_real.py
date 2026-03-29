"""Compute factor IC on real mainpool/top3 IV datasets."""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mainpool", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/mainpool_1m.parquet")
    p.add_argument("--top3", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/top3_contract_1m.parquet")
    p.add_argument("--horizons", default="1,5,15")
    p.add_argument("--iv-rsi-window", type=int, default=14)
    return p.parse_args()


def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    up_ema = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    dn_ema = dn.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = up_ema / dn_ema.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ic_table(df: pd.DataFrame, factors: list[str], target: str) -> list[tuple[str, float, float]]:
    out = []
    d = df.dropna(subset=[target]).copy()
    for f in factors:
        z = d[[f, target]].dropna()
        if len(z) < 30:
            out.append((f, np.nan, np.nan))
            continue
        pear = z[f].corr(z[target], method="pearson")
        spear = z[f].corr(z[target], method="spearman")
        out.append((f, float(pear), float(spear)))
    out.sort(key=lambda x: abs(x[2]) if pd.notna(x[2]) else -1, reverse=True)
    return out


def main():
    args = parse_args()
    hs = [int(x) for x in args.horizons.split(",")]

    m = pd.read_parquet(args.mainpool).sort_values("dt_exch")
    m["iv_rsi"] = calc_rsi(m["iv_pool"], args.iv_rsi_window)
    m["iv_rsi_centered"] = (m["iv_rsi"] - 50.0) / 50.0
    m_factors = ["vega_signed_1m", "vega_abs_1m", "spread_pool_1m", "f_ret_1m", "iv_rsi_centered"]

    print("[MAINPOOL IC]")
    for h in hs:
        y = f"y_{h}m"
        m[y] = m["iv_pool"].shift(-h) - m["iv_pool"]
        rows = ic_table(m, m_factors, y)
        print(f"h={h}")
        for f, p, s in rows:
            print(f"  {f:18s} pearson={p:+.5f} spearman={s:+.5f}")

    t = pd.read_parquet(args.top3).sort_values(["symbol", "dt_exch"])
    t["iv_rsi"] = t.groupby("symbol")["iv"].transform(lambda s: calc_rsi(s, args.iv_rsi_window))
    t["iv_rsi_centered"] = (t["iv_rsi"] - 50.0) / 50.0
    t_factors = ["vega_signed_1m", "vega_abs_1m", "spread_1m", "d_volume_1m", "f_ret_1m", "iv_rsi_centered"]

    print("\n[TOP3 IC | avg across symbols]")
    for h in hs:
        y = f"y_{h}m"
        t[y] = t.groupby("symbol")["iv"].shift(-h) - t["iv"]
        sym_tables = []
        for _, g in t.groupby("symbol"):
            tab = ic_table(g, t_factors, y)
            sym_tables.append({f: s for f, _, s in tab})
        print(f"h={h}")
        for f in t_factors:
            vals = [d.get(f, np.nan) for d in sym_tables]
            v = float(np.nanmean(vals))
            print(f"  {f:18s} spearman={v:+.5f}")


if __name__ == "__main__":
    main()
