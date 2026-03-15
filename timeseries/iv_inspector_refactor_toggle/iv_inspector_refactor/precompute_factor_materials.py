# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

from iv_inspector.data import load_data
from iv_inspector.feature_store import precompute_factor_materials


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute factor materials for iv_inspector.")
    parser.add_argument("--input", required=True, help="Source dataset under data/derived or absolute path.")
    parser.add_argument("--base-rule", required=True, help="Base resample rule, e.g. 1S / 500ms / 2S.")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol whitelist.")
    args = parser.parse_args()

    df_raw = load_data(args.input)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out_path = precompute_factor_materials(df_raw, args.base_rule, data_path=args.input, symbols=symbols or None)
    print(f"factor materials -> {out_path}")


if __name__ == "__main__":
    main()
