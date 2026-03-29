"""Build Level-A labels for IV change horizons.

Usage:
  python src/build_labels.py \
    --input data/dataset_1m.parquet \
    --output data/labels_1m.parquet \
    --iv-col iv_atm \
    --horizons 1,5,15
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build IV delta labels")
    p.add_argument("--input", required=True, help="Aligned dataset parquet/csv")
    p.add_argument("--output", required=True, help="Output labels parquet")
    p.add_argument("--iv-col", default="iv_atm", help="IV column name")
    p.add_argument("--horizons", default="1,5,15", help="Comma-separated minute horizons")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    if args.iv_col not in df.columns:
        raise ValueError(f"missing IV column: {args.iv_col}")

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    labels = pd.DataFrame(index=df.index)
    for h in horizons:
        labels[f"y_{h}m"] = df[args.iv_col].shift(-h) - df[args.iv_col]

    labels.to_parquet(out_path)
    print(f"[OK] wrote labels: {out_path} cols={len(labels.columns)} rows={len(labels)}")


if __name__ == "__main__":
    main()
