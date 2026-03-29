from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit_gd(X: np.ndarray, y: np.ndarray, lr=0.05, steps=200, l2=1e-3):
    w = np.zeros(X.shape[1], dtype=float)
    for _ in range(steps):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / len(y)
        grad += l2 * w
        grad[0] -= l2 * w[0]  # no reg bias
        w -= lr * grad
    return w


def wf_logit(df: pd.DataFrame, ycol: str, features: list[str], conf_thr=0.15, min_train=240):
    d = df.dropna(subset=[ycol]).copy()
    d = d.dropna(subset=features)
    if len(d) <= min_train + 20:
        return None

    y_raw = d[ycol].to_numpy(float)
    y_cls = (y_raw > 0).astype(float)
    X = d[features].to_numpy(float)

    probs, pred_sign, true_sign, trig = [], [], [], []

    for t in range(min_train, len(d)):
        xtr = X[:t]
        ytr = y_cls[:t]
        xte = X[t:t+1]

        mu = np.nanmean(xtr, axis=0)
        sd = np.nanstd(xtr, axis=0)
        sd[sd < 1e-12] = 1.0
        xtr = np.clip((np.nan_to_num(xtr) - mu) / sd, -8, 8)
        xte = np.clip((np.nan_to_num(xte) - mu) / sd, -8, 8)

        xtr = np.c_[np.ones(len(xtr)), xtr]
        xte = np.c_[np.ones(len(xte)), xte]

        w = fit_logit_gd(xtr, ytr)
        p = float(sigmoid((xte @ w)[0]))
        conf = abs(p - 0.5)

        probs.append(p)
        pred_sign.append(1.0 if p >= 0.5 else -1.0)
        true_sign.append(1.0 if y_raw[t] > 0 else -1.0)
        trig.append(conf >= conf_thr)

    probs = np.array(probs)
    pred_sign = np.array(pred_sign)
    true_sign = np.array(true_sign)
    trig = np.array(trig, dtype=bool)
    y_oos = y_raw[min_train:]

    all_hit = float(np.mean(pred_sign == true_sign))
    if trig.any():
        triggered_hit = float(np.mean(pred_sign[trig] == true_sign[trig]))
        avg_vol_decimal = float(np.mean(pred_sign[trig] * y_oos[trig]))
    else:
        triggered_hit = math.nan
        avg_vol_decimal = math.nan

    return {
        "n": int(len(y_oos)),
        "n_sig": int(trig.sum()),
        "all_hit": all_hit,
        "triggered_hit": triggered_hit,
        "coverage": float(trig.mean()),
        "avg_vol_decimal": avg_vol_decimal,
        "avg_vol_points": float(avg_vol_decimal * 100) if avg_vol_decimal == avg_vol_decimal else math.nan,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--conf-thr", type=float, default=0.15)
    p.add_argument("--features", default="flow,iv_dev_ema5_ratio,iv_mom3,iv_willr10,resid_z")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.input).sort_values("dt_exch")
    ycol = f"y_{args.horizon}m"
    features = [x.strip() for x in args.features.split(",") if x.strip()]

    res = wf_logit(df, ycol, features, conf_thr=args.conf_thr)
    out = {
        "input": args.input,
        "horizon": args.horizon,
        "conf_thr": args.conf_thr,
        "features": features,
        "metrics": res,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
