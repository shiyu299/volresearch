# -*- coding: utf-8 -*-
import argparse
import pandas as pd

from preprocess_iv_features import preprocess
from factors.iv_price_factors import build_iv_price_factors
from factors.tradevega_volume_factors import build_tradevega_factors
from factors.orderbook_factors import build_orderbook_factors


def main(inp: str, out: str):
    df = pd.read_parquet(inp) if inp.lower().endswith(".parquet") else pd.read_csv(inp)
    x = preprocess(df)
    x = build_iv_price_factors(x, iv_col="iv")
    x = build_tradevega_factors(x)
    x = build_orderbook_factors(x)

    # target: future iv change (10 steps)
    x = x.sort_values("dt_exch")
    x["target_iv_ret_fwd10"] = x["iv"].shift(-10) / x["iv"] - 1.0

    cols_keep = [
        "dt_exch", "symbol", "is_option", "is_future", "iv", "F_used",
        "iv_ret_1", "iv_ret_5", "iv_mom_30", "iv_vol_60",
        "fut_ret_1", "fut_ret_5", "fut_micro_dev",
        "tv_signed", "tv_abs", "tv_imbalance_30", "tv_z_60", "dvol_30",
        "spread_abs", "spread_rel", "book_imbalance", "book_pressure_ema",
        "bid_iv", "ask_iv", "mid_iv", "target_iv_ret_fwd10",
    ]
    cols_keep = [c for c in cols_keep if c in x.columns]
    y = x[cols_keep]
    y.to_parquet(out, index=False)
    print(f"saved {out}, rows={len(y)}, cols={len(y.columns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(args.input, args.output)
