from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from sc_factor_lib import add_targets, build_sc_factors
from sc_train_eval import wf_logit


DEFAULT_FEATURES = [
    "flow",
    "iv_dev_ema5_ratio",
    "iv_mom3",
    "iv_willr10",
    "resid_z",
]
SECOND_BATCH = ["flow_ema10", "shockF"]


def load_sc_concat(sc_file: str, out_file: str):
    df = pd.read_parquet(sc_file)
    if "symbol" in df.columns:
        m = df["symbol"].astype(str).str.contains("sc2604", case=False, na=False)
        df = df[m].copy()
    df.to_parquet(out_file, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sc-raw", default="/Users/shiyu/.openclaw/workspace/volresearch/data/derived/sc2604_iv_20260313.parquet")
    p.add_argument("--workdir", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/sc_pipeline")
    p.add_argument("--conf-thr", type=float, default=0.15)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--atm-n", type=int, default=6)
    p.add_argument("--report", default="/Users/shiyu/.openclaw/workspace/iv_vega_hf/reports/SC_SECOND_BATCH_PIPELINE_RESULT.json")
    args = p.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)

    sc_filtered = wd / "sc_filtered.parquet"
    load_sc_concat(args.sc_raw, str(sc_filtered))

    # Reuse existing dataset builder for 1min mainpool/top3
    subprocess.check_call([
        "python3",
        "/Users/shiyu/.openclaw/workspace/iv_vega_hf/src/load_volresearch_data.py",
        "--input-dir",
        "/Users/shiyu/.openclaw/workspace/volresearch/data/derived",
        "--output-dir",
        "/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real",
        "--topn", "3", "--atm-n", str(args.atm_n),
    ])

    mainpool = pd.read_parquet("/Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/mainpool_1m.parquet")
    fac = build_sc_factors(mainpool)
    fac = add_targets(fac, horizons=(1, 3, 5))
    fac_path = wd / "sc_factors_targets.parquet"
    fac.to_parquet(fac_path, index=False)

    ycol = f"y_{args.horizon}m"
    base = wf_logit(fac, ycol, DEFAULT_FEATURES, conf_thr=args.conf_thr)

    add_results = {}
    for f in SECOND_BATCH:
        feats = DEFAULT_FEATURES + [f]
        add_results[f] = wf_logit(fac, ycol, feats, conf_thr=args.conf_thr)

    keep = DEFAULT_FEATURES + ["flow_ema10", "shockF"]
    final_metrics = wf_logit(fac, ycol, keep, conf_thr=args.conf_thr)

    out = {
        "horizon": args.horizon,
        "conf_thr": args.conf_thr,
        "base_features": DEFAULT_FEATURES,
        "second_batch_candidates": SECOND_BATCH,
        "base_metrics": base,
        "add_one_metrics": add_results,
        "final_keep": keep,
        "final_metrics": final_metrics,
        "artifacts": {
            "factors_targets": str(fac_path),
        },
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
