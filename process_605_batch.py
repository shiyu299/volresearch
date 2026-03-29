from __future__ import annotations

import argparse
import re
from pathlib import Path

from timeseries.marketvolseries_modified_v4 import run_pl603_iv_traded_v4
from timeseries.iv_inspector_refactor_toggle.iv_inspector_refactor.iv_inspector.data import load_data
from timeseries.iv_inspector_refactor_toggle.iv_inspector_refactor.iv_inspector.feature_store import (
    factor_store_path,
    precompute_factor_materials,
)


REPO_ROOT = Path(__file__).resolve().parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
DERIVED_ROOT = REPO_ROOT / "data" / "derived"
EXPIRY_605 = "2026-04-13"
CSV_RE = re.compile(r"^(?P<product>[A-Za-z]+)(?P<series>\d{6})\.csv$")


def discover_csvs(raw_root: Path, day_dirs: list[str] | None, products: list[str] | None) -> list[Path]:
    selected_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.isdigit()])
    if day_dirs:
        allow = set(day_dirs)
        selected_dirs = [p for p in selected_dirs if p.name in allow]

    files: list[Path] = []
    for day_dir in selected_dirs:
        for fp in sorted(day_dir.glob("*.csv")):
            m = CSV_RE.match(fp.name)
            if not m:
                continue
            product = m.group("product").upper()
            if products and product not in products:
                continue
            files.append(fp)
    return files


def infer_underlying(csv_path: Path) -> tuple[str, str]:
    m = CSV_RE.match(csv_path.name)
    if not m:
        raise ValueError(f"Unsupported csv filename: {csv_path.name}")
    product = m.group("product").upper()
    series = m.group("series")
    underlying = f"{product}{series[:-3]}"
    return product, underlying


def process_one(csv_path: Path, base_rule: str, spread_limit: float) -> tuple[Path, Path]:
    product, underlying = infer_underlying(csv_path)
    out_dir = DERIVED_ROOT / product
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}.parquet"

    run_pl603_iv_traded_v4(
        csv_path=csv_path,
        underlying=underlying,
        expiry_date=EXPIRY_605,
        spread_limit=spread_limit,
        out_path=out_path,
        out_csv_preview_path=None,
    )

    df_raw = load_data(str(out_path))
    factor_path = precompute_factor_materials(df_raw, base_rule, data_path=out_path)
    return out_path, factor_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process 605-series raw CSVs into derived parquet and factor materials.")
    parser.add_argument("--day-dirs", default="605310,605311,605312,605313", help="Comma-separated raw subdirectories under data/raw.")
    parser.add_argument("--products", default="FG,MA,PL,SH,TA", help="Comma-separated product prefixes to process.")
    parser.add_argument("--base-rule", default="1S", help="Factor base rule, default 1S.")
    parser.add_argument("--spread-limit", type=float, default=15.0, help="Spread limit for modified v4.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files whose derived parquet and factor parquet already exist.")
    args = parser.parse_args()

    day_dirs = [x.strip() for x in args.day_dirs.split(",") if x.strip()]
    products = [x.strip().upper() for x in args.products.split(",") if x.strip()]

    csvs = discover_csvs(RAW_ROOT, day_dirs=day_dirs, products=products)
    if not csvs:
        print("No matching CSV files found.")
        return 1

    print(f"Found {len(csvs)} CSV files.")
    for idx, csv_path in enumerate(csvs, start=1):
        product, underlying = infer_underlying(csv_path)
        out_path = DERIVED_ROOT / product / f"{csv_path.stem}.parquet"
        fac_path = factor_store_path(out_path, args.base_rule)
        if args.skip_existing and out_path.exists() and fac_path.exists():
            print(f"[{idx}/{len(csvs)}] skip existing -> {csv_path}")
            continue
        print(f"[{idx}/{len(csvs)}] {csv_path} -> product={product}, underlying={underlying}")
        out_path, factor_path = process_one(csv_path, base_rule=args.base_rule, spread_limit=args.spread_limit)
        print(f"  derived -> {out_path}")
        print(f"  factors -> {factor_path}")

    print("Batch processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
