from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def dataframe_digest(df: pd.DataFrame) -> str:
    row_hash = pd.util.hash_pandas_object(df, index=True)
    return hashlib.md5(row_hash.to_numpy().tobytes()).hexdigest()


def summarize_detailed_diff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    print("\nDetailed comparison")
    if "time" in df_a.columns and "time" in df_b.columns:
        print(f"A time min/max: {df_a['time'].min()} -> {df_a['time'].max()}")
        print(f"B time min/max: {df_b['time'].min()} -> {df_b['time'].max()}")
        print(f"A unique time count: {df_a['time'].nunique()}")
        print(f"B unique time count: {df_b['time'].nunique()}")

    key_cols = [
        c
        for c in ["time", "lastprice", "volume", "totalvaluetraded", "openinterest", "bidprice1", "askprice1", "bidvol1", "askvol1"]
        if c in df_a.columns and c in df_b.columns
    ]
    if not key_cols:
        print("No common key columns available for detailed diff.")
        return

    hash_a = pd.util.hash_pandas_object(df_a[key_cols], index=False).value_counts()
    hash_b = pd.util.hash_pandas_object(df_b[key_cols], index=False).value_counts()
    extra_a = (hash_a.sub(hash_b, fill_value=0)).clip(lower=0)
    extra_b = (hash_b.sub(hash_a, fill_value=0)).clip(lower=0)
    print(f"Extra row multiplicity in A over B: {int(extra_a.sum())}")
    print(f"Extra row multiplicity in B over A: {int(extra_b.sum())}")
    print(f"B is subset of A on key cols: {bool(extra_b.sum() == 0)}")

    merged = df_a[key_cols].merge(df_b[key_cols].drop_duplicates(), on=key_cols, how="left", indicator=True)
    diff = merged[merged["_merge"] == "left_only"]
    print(f"Rows in A but not B on key cols: {len(diff)}")
    if not diff.empty and "time" in diff.columns:
        print(f"Diff time min/max: {diff['time'].min()} -> {diff['time'].max()}")
        print("Last 10 differing timestamps:")
        print(diff["time"].value_counts().sort_index().tail(10).to_string())


def compare_parquet(path_a: Path, path_b: Path, detailed: bool = True) -> int:
    print(f"A: {path_a}")
    print(f"B: {path_b}")

    if not path_a.exists() or not path_b.exists():
        print("At least one file does not exist.")
        return 2

    stat_a = path_a.stat()
    stat_b = path_b.stat()
    print(f"File size A: {stat_a.st_size}")
    print(f"File size B: {stat_b.st_size}")

    md5_a = file_md5(path_a)
    md5_b = file_md5(path_b)
    print(f"File MD5 A: {md5_a}")
    print(f"File MD5 B: {md5_b}")

    if md5_a == md5_b:
        print("Binary-identical files: YES")
        return 0

    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    print(f"Shape A: {df_a.shape}")
    print(f"Shape B: {df_b.shape}")
    print(f"Columns equal: {list(df_a.columns) == list(df_b.columns)}")

    if df_a.shape != df_b.shape:
        print("Exactly duplicated data: NO (shape mismatch)")
        if detailed:
            summarize_detailed_diff(df_a, df_b)
        return 1

    if list(df_a.columns) != list(df_b.columns):
        print("Exactly duplicated data: NO (column mismatch)")
        if detailed:
            summarize_detailed_diff(df_a, df_b)
        return 1

    digest_a = dataframe_digest(df_a)
    digest_b = dataframe_digest(df_b)
    print(f"Data digest A: {digest_a}")
    print(f"Data digest B: {digest_b}")

    if digest_a == digest_b and df_a.equals(df_b):
        print("Exactly duplicated data: YES")
        return 0

    diff_rows = (df_a != df_b).any(axis=1).sum()
    print(f"Exactly duplicated data: NO")
    print(f"Rows with any difference: {diff_rows}")
    if detailed:
        summarize_detailed_diff(df_a, df_b)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether two parquet files are identical.")
    parser.add_argument("path_a")
    parser.add_argument("path_b")
    parser.add_argument("--no-detailed", action="store_true", help="Skip detailed diff summary.")
    args = parser.parse_args()
    return compare_parquet(Path(args.path_a), Path(args.path_b), detailed=not args.no_detailed)


if __name__ == "__main__":
    raise SystemExit(main())
