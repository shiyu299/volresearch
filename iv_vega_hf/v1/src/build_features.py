"""Build IV/Vega microstructure features (skeleton v1).

Usage:
  python src/build_features.py \
    --input data/dataset_1m.parquet \
    --output data/features_1m.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build IV/Vega features")
    p.add_argument("--input", required=True, help="Aligned dataset parquet/csv")
    p.add_argument("--output", required=True, help="Output features parquet")
    p.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    feat = pd.DataFrame(index=df.index)

    # --- Skeleton features (computed only if required columns exist) ---
    if {"buy_vega", "sell_vega"}.issubset(df.columns):
        total = df["buy_vega"].fillna(0) + df["sell_vega"].fillna(0)
        feat["vega_imbalance_1m"] = (df["buy_vega"].fillna(0) - df["sell_vega"].fillna(0)) / np.maximum(total, args.eps)

    if "signed_vega" in df.columns:
        feat["vega_signed_1m"] = df["signed_vega"].fillna(0)
        feat["vega_signed_5m"] = feat["vega_signed_1m"].rolling(5, min_periods=1).sum()
        feat["vega_signed_15m_ewm"] = feat["vega_signed_1m"].ewm(span=15, adjust=False).mean()

    if "iv_atm" in df.columns:
        feat["iv_delta_1m"] = df["iv_atm"].diff(1)
        roll_mean = df["iv_atm"].rolling(30, min_periods=10).mean()
        roll_std = df["iv_atm"].rolling(30, min_periods=10).std()
        feat["iv_zscore_30m"] = (df["iv_atm"] - roll_mean) / roll_std.replace(0, np.nan)

    if {"trade_count", "quote_update_count"}.issubset(df.columns):
        feat["micro_trade_quote_ratio_1m"] = df["trade_count"].fillna(0) / np.maximum(df["quote_update_count"].fillna(0), 1)

    if "underlying_mid" in df.columns:
        feat["underlying_ret_1m"] = np.log(df["underlying_mid"] / df["underlying_mid"].shift(1))

    feat.to_parquet(out_path)
    print(f"[OK] wrote features: {out_path} cols={len(feat.columns)} rows={len(feat)}")


if __name__ == "__main__":
    main()
