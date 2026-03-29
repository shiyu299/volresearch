from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit_gd(X: np.ndarray, y: np.ndarray, lr=0.05, steps=250, l2=1e-3):
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
    p.add_argument('--conf-thr', type=float, default=0.15)
    p.add_argument('--min-train', type=int, default=30)
    p.add_argument('--out-json', default='/Users/shiyu/.openclaw/workspace/iv_vega_hf/v2/reports/SC_V2_SECOND_TRIGGER_5M_FACTOR.json')
    return p.parse_args()


def build_5m_dataset(raw: pd.DataFrame, atm_n=15) -> pd.DataFrame:
    raw = raw.sort_values('dt_exch').copy()

    # filter one-sided future seconds
    fut = raw[raw['is_future'] == True].copy()
    fut['sec'] = fut['dt_exch'].dt.floor('1s')
    two_sided_sec = fut.groupby('sec').apply(lambda g: bool(((g['bidprice1'].fillna(0) > 0) & (g['askprice1'].fillna(0) > 0)).any()))
    raw['sec'] = raw['dt_exch'].dt.floor('1s')
    raw = raw[raw['sec'].map(two_sided_sec).fillna(False)].copy()

    opt = raw[raw['is_option'] == True].copy()
    opt = opt.dropna(subset=['iv', 'F_used', 'K'])
    opt['dist'] = (opt['K'] - opt['F_used']).abs()

    # 1min pool first
    opt1 = opt.sort_values(['dt_exch', 'dist']).groupby('dt_exch', as_index=False).head(atm_n)
    m1 = opt1.groupby(pd.Grouper(key='dt_exch', freq='1min')).apply(
        lambda g: pd.Series({
            'iv_pool': float(np.average(g['iv'], weights=np.clip(g['vega'].fillna(0), 0, None))) if len(g) and np.clip(g['vega'].fillna(0), 0, None).sum() > 0 else (float(g['iv'].mean()) if len(g) else np.nan),
            'flow': g['traded_vega_signed'].fillna(0).sum(),
            'F_used': g['F_used'].dropna().iloc[-1] if g['F_used'].notna().any() else np.nan,
        })
    ).reset_index().dropna(subset=['iv_pool'])

    # aggregate to 5min for factor construction
    m5 = m1.groupby(pd.Grouper(key='dt_exch', freq='5min')).agg(
        iv_pool=('iv_pool', 'last'),
        flow=('flow', 'sum'),
        F_used=('F_used', 'last'),
    ).reset_index().dropna(subset=['iv_pool'])

    e5 = m5['iv_pool'].ewm(span=5, adjust=False).mean()
    m5['iv_dev_ema5_ratio'] = (m5['iv_pool'] - e5) / e5.abs().replace(0, np.nan)
    m5['iv_mom3'] = m5['iv_pool'] - m5['iv_pool'].shift(3)

    hh = m5['iv_pool'].rolling(10, min_periods=10).max()
    ll = m5['iv_pool'].rolling(10, min_periods=10).min()
    m5['iv_willr10'] = -100 * (hh - m5['iv_pool']) / (hh - ll).replace(0, np.nan)

    m5['flow_ema10'] = m5['flow'].ewm(span=10, adjust=False).mean()

    dF = np.log(m5['F_used'] / m5['F_used'].shift(1))
    shockF = dF + 0.5 * dF.shift(1) + 0.25 * dF.shift(2)
    dIV = m5['iv_pool'].diff(1)
    beta = dIV.rolling(30, min_periods=20).cov(shockF) / shockF.rolling(30, min_periods=20).var().replace(0, np.nan)
    resid = dIV - beta * shockF
    rz = (resid - resid.rolling(30, min_periods=10).mean()) / resid.rolling(30, min_periods=10).std().replace(0, np.nan)

    m5['shockF'] = shockF
    m5['resid_z'] = rz
    m5['y_5m'] = m5['iv_pool'].shift(-1) - m5['iv_pool']

    return m5, two_sided_sec


def train_and_infer_5m(m5: pd.DataFrame, min_train=120, conf_thr=0.15):
    feats = ['flow', 'iv_dev_ema5_ratio', 'iv_mom3', 'iv_willr10', 'resid_z', 'flow_ema10', 'shockF']
    d = m5.dropna(subset=feats + ['y_5m']).copy()
    X = d[feats].to_numpy(float)
    y = (d['y_5m'].to_numpy(float) > 0).astype(float)
    y_raw = d['y_5m'].to_numpy(float)

    rows = []
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

        rows.append({
            'dt_5m': d.iloc[t]['dt_exch'],
            'p': p,
            'pred_sign': pred_sign,
            'conf': conf,
            'triggered': conf >= conf_thr,
            'y_5m': float(y_raw[t]),
            'true_sign': 1 if y_raw[t] > 0 else -1,
        })

    return pd.DataFrame(rows)


def expand_to_seconds(pred5: pd.DataFrame, two_sided_sec: pd.Series) -> pd.DataFrame:
    if pred5 is None or pred5.empty:
        return pd.DataFrame()
    # each 5m signal applies to [dt_5m, dt_5m+5m)
    secs = pd.DataFrame({'sec': two_sided_sec.index[two_sided_sec.values]})
    secs = secs.sort_values('sec')
    pred5 = pred5.sort_values('dt_5m').copy()
    pred5['end'] = pred5['dt_5m'] + pd.Timedelta(minutes=5)

    merged = pd.merge_asof(secs, pred5, left_on='sec', right_on='dt_5m', direction='backward')
    merged = merged[merged['sec'] < merged['end']].copy()
    return merged


def summarize(df: pd.DataFrame):
    n = len(df)
    if n == 0:
        return None
    all_hit = float((df['pred_sign'] == df['true_sign']).mean())
    trig = df['triggered'] == True
    n_sig = int(trig.sum())
    if n_sig > 0:
        triggered_hit = float((df.loc[trig, 'pred_sign'] == df.loc[trig, 'true_sign']).mean())
        avg_vol_decimal = float((df.loc[trig, 'pred_sign'] * df.loc[trig, 'y_5m']).mean())
    else:
        triggered_hit = float('nan')
        avg_vol_decimal = float('nan')
    return {
        'n': n,
        'n_sig': n_sig,
        'all_hit': all_hit,
        'triggered_hit': triggered_hit,
        'coverage': float(n_sig / n) if n > 0 else float('nan'),
        'avg_vol_decimal': avg_vol_decimal,
        'avg_vol_points': float(avg_vol_decimal * 100) if avg_vol_decimal == avg_vol_decimal else float('nan'),
    }


def main():
    args = parse_args()
    raw = pd.read_parquet(args.input)
    raw['dt_exch'] = pd.to_datetime(raw['dt_exch'])

    m5, two_sided_sec = build_5m_dataset(raw, atm_n=args.atm_n)
    pred5 = train_and_infer_5m(m5, min_train=args.min_train, conf_thr=args.conf_thr)
    sec_eval = expand_to_seconds(pred5, two_sided_sec)

    out = {
        'config': {
            'input': args.input,
            'atm_n': args.atm_n,
            'conf_thr': args.conf_thr,
            'min_train': args.min_train,
            'factor_freq': '5min',
            'trigger_freq': '1s',
            'label': 'iv_pool(t+5m)-iv_pool(t)',
            'filter': 'exclude one-sided future seconds'
        },
        'metrics_second_trigger': summarize(sec_eval),
        'metrics_5m_trigger': summarize(pred5),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
