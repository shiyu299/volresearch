from __future__ import annotations

import argparse
import re
from pathlib import Path

from marketvolseries_modified_v4 import run_pl603_iv_traded_v4

SIMPLE_ROOT = Path(__file__).resolve().parent
RAW_ROOT = SIMPLE_ROOT / "data" / "raw"
DERIVED_ROOT = SIMPLE_ROOT / "data" / "derived"
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


def process_one(csv_path: Path, expiry_date: str, spread_limit: float) -> Path:
    product, underlying = infer_underlying(csv_path)
    out_dir = DERIVED_ROOT / product
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}.parquet"

    run_pl603_iv_traded_v4(
        csv_path=csv_path,
        underlying=underlying,
        expiry_date=expiry_date,
        spread_limit=spread_limit,
        out_path=out_path,
        out_csv_preview_path=None,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple raw CSV -> derived parquet pipeline without factor materials.")
    parser.add_argument("--csv-path", help="Single raw CSV path. If omitted, batch mode scans simple/data/raw/<day_dir>/*.csv.")
    parser.add_argument("--expiry-date", required=True, help="Expiry date used by modified v4, for example 2026-04-13.")
    parser.add_argument("--spread-limit", type=float, default=15.0, help="Spread limit for modified v4.")
    parser.add_argument("--day-dirs", default="", help="Comma-separated raw subdirectories under simple/data/raw.")
    parser.add_argument("--products", default="", help="Comma-separated product prefixes, for example FG,MA,SH,TA.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip output parquet if it already exists.")
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = (SIMPLE_ROOT / csv_path).resolve()
        out_path = process_one(csv_path, expiry_date=args.expiry_date, spread_limit=args.spread_limit)
        print(f"derived -> {out_path}")
        return 0

    day_dirs = [x.strip() for x in args.day_dirs.split(",") if x.strip()]
    products = [x.strip().upper() for x in args.products.split(",") if x.strip()]
    csvs = discover_csvs(RAW_ROOT, day_dirs=day_dirs or None, products=products or None)
    if not csvs:
        print("No matching CSV files found.")
        return 1

    print(f"Found {len(csvs)} CSV files.")
    for idx, csv_path in enumerate(csvs, start=1):
        product, _ = infer_underlying(csv_path)
        out_path = DERIVED_ROOT / product / f"{csv_path.stem}.parquet"
        if args.skip_existing and out_path.exists():
            print(f"[{idx}/{len(csvs)}] skip existing -> {csv_path}")
            continue
        print(f"[{idx}/{len(csvs)}] {csv_path}")
        out_path = process_one(csv_path, expiry_date=args.expiry_date, spread_limit=args.spread_limit)
        print(f"  derived -> {out_path}")
    print("Batch processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
