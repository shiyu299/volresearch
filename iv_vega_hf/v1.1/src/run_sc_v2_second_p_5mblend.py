from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit_gd(X: np.ndarray, y: np.ndarray, lr=0.03, steps=120, l2=1e-3):
    w = np.zeros(X.shape[1], dtype=float)
    for _ in range(steps):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / max(1, len(y))
        grad += l2 * w
        grad[0] -= l2 * w[0]
        w -= lr * grad
    return w


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='/Users/shiyu/.openclaw/workspace/volresearch/data/derived/sc2604_iv_20260313.parquet')
    p.add_argument('--atm-n', type=int, default=15)
    p.add_argument('--conf-thr', type=float, default=0.2)
    p.add_argument('--min-train', type=int, default=3600)
    p.add_argument('--out-json', default='/Users/shiyu/.openclaw/workspace/iv_vega_hf/v2/reports/SC_V2_SECOND_P_5MBLEND.json')
    return p.parse_args()


def build_second_panel(raw: pd.DataFrame, atm_n: int) -> pd.DataFrame:
    raw = raw.sort_values('dt_exch').copy()

    # single-side future filter (second-level)
    fut = raw[raw['is_future'] == True].copy()
    fut['sec'] = fut['dt_exch'].dt.floor('1s')
    two_sided = fut.groupby('sec').apply(lambda g: bool(((g['bidprice1'].fillna(0) > 0) & (g['askprice1'].fillna(0) > 0)).any()))

    raw['sec'] = raw['dt_exch'].dt.floor('1s')
    raw = raw[raw['sec'].map(two_sided).fillna(False)].copy()

    opt = raw[raw['is_option'] == True].copy()
    opt = opt.dropna(subset=['iv', 'F_used', 'K'])
    opt['dist'] = (opt['K'] - opt['F_used']).abs()

    nearest = opt.sort_values(['sec', 'dist']).groupby('sec', as_index=False).head(atm_n)

    def _agg(g: pd.DataFrame):
        w = np.clip(g['vega'].fillna(0), 0, None)
        if len(g) == 0:
            return pd.Series({'iv_pool': np.nan, 'flow': 0.0, 'F_used': np.nan})
        iv = float(np.average(g['iv'], weights=w)) if w.sum() > 0 else float(g['iv'].mean())
        return pd.Series({
            'iv_pool': iv,
            'flow': g['traded_vega_signed'].fillna(0).sum(),
            'F_used': g['F_used'].dropna().iloc[-1] if g['F_used'].notna().any() else np.nan,
        })

    sec = nearest.groupby('sec').apply(_agg).reset_index().rename(columns={'sec': 'dt_exch'})
    sec = sec.sort_values('dt_exch').dropna(subset=['iv_pool'])

    # second-level features, but 5-min information blend (300s windows)
    iv = sec['iv_pool']
    flow = sec['flow'].fillna(0)

    ema300 = iv.ewm(span=300, adjust=False).mean()
    sec['iv_dev_ema5m_ratio'] = (iv - ema300) / ema300.abs().replace(0, np.nan)
    sec['iv_mom_5m'] = iv - iv.shift(300)

    hh = iv.rolling(300, min_periods=120).max()
    ll = iv.rolling(300, min_periods=120).min()
    sec['iv_willr_5m'] = -100.0 * (hh - iv) / (hh - ll).replace(0, np.nan)

    sec['flow_5m_sum'] = flow.rolling(300, min_periods=60).sum()
    sec['flow_5m_ema'] = flow.ewm(span=300, adjust=False).mean()

    dF = np.log(sec['F_used'] / sec['F_used'].shift(1))
    shockF = dF + 0.5 * dF.shift(1) + 0.25 * dF.shift(2)
    dIV = iv.diff(1)
    beta = dIV.rolling(600, min_periods=240).cov(shockF) / shockF.rolling(600, min_periods=240).var().replace(0, np.nan)
    resid = dIV - beta * shockF
    sec['resid_z_5mctx'] = (resid - resid.rolling(600, min_periods=180).mean()) / resid.rolling(600, min_periods=180).std().replace(0, np.nan)
    sec['shockF'] = shockF

    # future validation: mean iv in [t+3m, t+5m]
    sec['future_iv_mean_3_5m'] = (iv.shift(-180).rolling(121, min_periods=80).mean().shift(-120))
    sec['y_3_5m_mean'] = sec['future_iv_mean_3_5m'] - iv

    return sec


def run_second_trigger(df: pd.DataFrame, conf_thr: float, min_train: int):
    feats = ['flow_5m_sum', 'iv_dev_ema5m_ratio', 'iv_mom_5m', 'iv_willr_5m', 'resid_z_5mctx', 'flow_5m_ema', 'shockF']
    d = df.dropna(subset=feats + ['y_3_5m_mean']).copy()
    if len(d) <= min_train + 100:
        return None

    X = d[feats].to_numpy(float)
    y_raw = d['y_3_5m_mean'].to_numpy(float)
    y = (y_raw > 0).astype(float)

    rec = []
    for t in range(min_train, len(d)):
        xtr = X[:t]
        ytr = y[:t]
        xte = X[t:t+1]

        mu = np.nanmean(xtr, axis=0)
        sd = np.nanstd(xtr, axis=0)
        sd[sd < 1e-12] = 1
        xtr = np.clip((np.nan_to_num(xtr) - mu) / sd, -8, 8)
        xte = np.clip((np.nan_to_num(xte) - mu) / sd, -8, 8)
        xtr = np.c_[np.ones(len(xtr)), xtr]
        xte = np.c_[np.ones(len(xte)), xte]

        w = fit_logit_gd(xtr, ytr)
        p = float(sigmoid((xte @ w)[0]))
        pred_sign = 1 if p >= 0.5 else -1
        conf = abs(p - 0.5)

        rec.append({
            'dt_exch': d.iloc[t]['dt_exch'],
            'p': p,
            'conf': conf,
            'triggered': conf >= conf_thr,
            'pred_sign': pred_sign,
            'y': float(y_raw[t]),
            'true_sign': 1 if y_raw[t] > 0 else -1,
        })

    out = pd.DataFrame(rec)
    n = len(out)
    trig = out['triggered'] == True
    n_sig = int(trig.sum())
    return {
        'n': n,
        'n_sig': n_sig,
        'all_hit': float((out['pred_sign'] == out['true_sign']).mean()),
        'triggered_hit': float((out.loc[trig, 'pred_sign'] == out.loc[trig, 'true_sign']).mean()) if n_sig else np.nan,
        'coverage': float(n_sig / n) if n else np.nan,
        'avg_vol_decimal': float((out.loc[trig, 'pred_sign'] * out.loc[trig, 'y']).mean()) if n_sig else np.nan,
        'avg_vol_points': float((out.loc[trig, 'pred_sign'] * out.loc[trig, 'y']).mean() * 100) if n_sig else np.nan,
    }


def main():
    args = parse_args()
    raw = pd.read_parquet(args.input)
    raw['dt_exch'] = pd.to_datetime(raw['dt_exch'])

    sec = build_second_panel(raw, args.atm_n)
    met = run_second_trigger(sec, args.conf_thr, args.min_train)

    out = {
        'config': {
            'atm_n': args.atm_n,
            'conf_thr': args.conf_thr,
            'min_train': args.min_train,
            'trigger_freq': '1s',
            'feature_context': 'past 5min blended from second data',
            'label': 'mean(IV[t+3m:t+5m]) - IV[t]',
            'filter': 'remove one-sided future seconds'
        },
        'metrics': met,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
