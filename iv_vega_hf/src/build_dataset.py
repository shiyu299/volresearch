"""Build aligned intraday dataset (skeleton v1).

Usage:
  python src/build_dataset.py \
    --input data/raw_quotes_trades.csv \
    --output data/dataset_1m.parquet \
    --freq 1min
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build aligned intraday dataset")
    p.add_argument("--input", required=True, help="Input raw csv/parquet path")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument("--freq", default="1min", help="Target bar frequency")
    p.add_argument("--timestamp-col", default="timestamp", help="Timestamp column")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # NOTE: keep imports local so script can still show clear error if pandas missing.
    import pandas as pd

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    if args.timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {args.timestamp_col}")

    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=False)
    df = df.sort_values(args.timestamp_col).set_index(args.timestamp_col)

    # Skeleton alignment: keep last observation per bar as point-in-time snapshot.
    aligned = df.resample(args.freq).last().dropna(how="all")

    aligned.to_parquet(out_path)
    print(f"[OK] wrote dataset: {out_path} rows={len(aligned)} freq={args.freq}")


if __name__ == "__main__":
    main()
