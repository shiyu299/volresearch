from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "flow_5m_sum",
    "iv_dev_ema5m_ratio",
    "iv_mom_5m",
    "iv_willr_5m",
    "resid_z_5mctx",
    "flow_5m_ema",
    "shockF",
]


TIME_SERIES_COLS = [
    "dt_exch",
    "trade_date",
    "product",
    "source_file",
    "session_id",
    "session_open",
    "session_close",
    "iv_pool",
    "flow",
    "cum_flow_trade_date",
    "cum_abs_flow_trade_date",
    "cum_flow_session",
    "cum_abs_flow_session",
    "F_used",
    "pool_n",
    "used_n",
    "grace_n",
    "mins_from_open",
    "mins_to_close",
    "train_allowed",
    "predict_allowed",
    "iv_dev_ema5m_ratio",
    "iv_mom_5m",
    "iv_willr_5m",
    "flow_5m_sum",
    "flow_5m_ema",
    "resid_z_5mctx",
    "shockF",
    "future_iv_mean_3_5m",
    "y_3_5m_mean",
    "y",
    "p",
    "conf",
    "triggered",
    "pred_sign",
    "true_sign",
    "n_train_seen",
]


DETAIL_COLS = [
    "dt_exch",
    "trade_date",
    "product",
    "source_file",
    "session_id",
    "session_open",
    "session_close",
    "symbol",
    "cp",
    "K",
    "in_pool",
    "in_grace",
    "decay_mult",
    "iv_contract",
    "vega_contract",
    "vega_weight_used",
    "traded_vega_signed",
    "spread",
    "iv_pool",
    "F_used",
]


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.03,
    steps: int = 120,
    l2: float = 1e-3,
    w0: np.ndarray | None = None,
) -> np.ndarray:
    w = np.zeros(X.shape[1], dtype=float) if w0 is None else np.array(w0, dtype=float, copy=True)
    for _ in range(steps):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / max(1, len(y))
        grad += l2 * w
        grad[0] -= l2 * w[0]
        w -= lr * grad
    return w


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="data/derived/TA")
    p.add_argument("--glob", default="*.parquet")
    p.add_argument("--atm-n", type=int, default=15)
    p.add_argument(
        "--trading-session",
        action="append",
        dest="trading_sessions",
        help="Trading session in HH:MM:SS-HH:MM:SS format; repeat this flag to add more sessions.",
    )
    p.add_argument("--pool-refresh-seconds", type=int, default=120)
    p.add_argument("--pool-refresh-fut-move", type=float, default=0.0)
    p.add_argument("--min-train-minutes-after-open", type=float, default=1.0)
    p.add_argument("--min-predict-minutes-after-open", type=float, default=1.0)
    p.add_argument("--min-predict-minutes-before-close", type=float, default=5.0)
    p.add_argument("--conf-thr", type=float, default=0.20)
    p.add_argument("--min-train", type=int, default=3600)
    p.add_argument("--fit-steps", type=int, default=120)
    p.add_argument("--online-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--out-dir", default="iv_vega_hf/v1.5/output/ta")
    args = p.parse_args()
    if not args.trading_sessions:
        args.trading_sessions = [
            "21:00:00-23:00:00",
            "09:00:00-10:15:00",
            "10:30:00-11:30:00",
            "13:30:00-15:00:00",
        ]
    return args


class SessionSpec(NamedTuple):
    start_seconds: int
    end_seconds: int
    raw: str


def _hhmmss_to_seconds(text: str) -> int:
    parts = text.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid time '{text}', expected HH:MM:SS")
    hh, mm, ss = (int(x) for x in parts)
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        raise ValueError(f"invalid time '{text}', out of range")
    return hh * 3600 + mm * 60 + ss


def parse_trading_sessions(values: list[str]) -> list[SessionSpec]:
    sessions: list[SessionSpec] = []
    for value in values:
        if "-" not in value:
            raise ValueError(f"invalid session '{value}', expected HH:MM:SS-HH:MM:SS")
        start_text, end_text = value.split("-", 1)
        sessions.append(
            SessionSpec(
                start_seconds=_hhmmss_to_seconds(start_text),
                end_seconds=_hhmmss_to_seconds(end_text),
                raw=value,
            )
        )
    return sessions


def _resolve_session_window(ts: pd.Timestamp, sessions: list[SessionSpec]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    ts = pd.Timestamp(ts)
    day = ts.normalize()
    tod = ts.hour * 3600 + ts.minute * 60 + ts.second
    for spec in sessions:
        start = spec.start_seconds
        end = spec.end_seconds
        if start <= end:
            if start <= tod <= end:
                return day + pd.Timedelta(seconds=start), day + pd.Timedelta(seconds=end)
            continue
        if tod >= start:
            return day + pd.Timedelta(seconds=start), day + pd.Timedelta(days=1, seconds=end)
        if tod <= end:
            return day - pd.Timedelta(days=1) + pd.Timedelta(seconds=start), day + pd.Timedelta(seconds=end)
    return None


def infer_trade_date(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.hour >= 21:
        return (ts + pd.Timedelta(days=1)).normalize()
    return ts.normalize()


def minutes_from_session_open(ts: pd.Timestamp, sessions: list[SessionSpec]) -> float:
    window = _resolve_session_window(ts, sessions)
    if window is None:
        return np.nan
    open_ts, _ = window
    return (pd.Timestamp(ts) - open_ts).total_seconds() / 60.0


def minutes_to_session_close(ts: pd.Timestamp, sessions: list[SessionSpec]) -> float:
    window = _resolve_session_window(ts, sessions)
    if window is None:
        return np.nan
    _, close_ts = window
    return (close_ts - pd.Timestamp(ts)).total_seconds() / 60.0


def _build_future_mean_label(iv: pd.Series, session_id: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=iv.index, dtype=float)
    for _, grp in iv.groupby(session_id):
        future_mean = grp.shift(-180).rolling(121, min_periods=80).mean().shift(-120)
        out.loc[grp.index] = future_mean
    return out


def _pick_pool_symbols(meta: pd.DataFrame, f_now: float, n: int) -> list[str]:
    if meta.empty or not np.isfinite(f_now):
        return []
    ks = meta["K"].astype(float).to_numpy()
    atm_k = float(meta.iloc[np.argmin(np.abs(ks - f_now))]["K"])
    atm_rows = meta[meta["K"].astype(float) == atm_k]
    picked = list(atm_rows["symbol"].astype(str).unique())
    if len(picked) >= n:
        return picked[:n]

    rest = meta[~meta["symbol"].isin(picked)].copy()
    if "cp" in rest.columns:
        cp = rest["cp"].astype(str).str.upper()
        is_call = cp.str.startswith("C")
        is_put = cp.str.startswith("P")
        k = rest["K"].astype(float)
        rest = rest[((is_call) & (k >= f_now)) | ((is_put) & (k <= f_now))]

    rest["dist"] = (rest["K"].astype(float) - f_now).abs()
    rest = rest.sort_values(["dist", "K"])
    for sym in rest["symbol"].astype(str).tolist():
        if sym not in picked:
            picked.append(sym)
        if len(picked) >= n:
            break
    return picked


def _grace_weight(elapsed_seconds: float) -> float:
    if not np.isfinite(elapsed_seconds):
        return 0.0
    if elapsed_seconds >= 300.0:
        return 0.0
    if elapsed_seconds >= 240.0:
        return 0.25
    if elapsed_seconds >= 120.0:
        return 0.5
    return 1.0


def build_second_panel(
    raw: pd.DataFrame,
    atm_n: int,
    product: str,
    source_name: str,
    pool_refresh_seconds: int,
    pool_refresh_fut_move: float,
    sessions: list[SessionSpec],
    min_train_minutes_after_open: float,
    min_predict_minutes_after_open: float,
    min_predict_minutes_before_close: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.sort_values("dt_exch").copy()
    raw["session_window"] = raw["dt_exch"].map(lambda ts: _resolve_session_window(ts, sessions))
    raw = raw[raw["session_window"].notna()].copy()
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    raw["session_open"] = raw["session_window"].map(lambda x: x[0])
    raw["session_close"] = raw["session_window"].map(lambda x: x[1])
    raw["session_id"] = raw["session_open"].astype(str) + "->" + raw["session_close"].astype(str)

    fut = raw[raw["is_future"] == True].copy()
    fut["sec"] = fut["dt_exch"].dt.floor("1s")
    two_sided = fut.groupby("sec").apply(
        lambda g: bool(((g["bidprice1"].fillna(0) > 0) & (g["askprice1"].fillna(0) > 0)).any())
    )

    raw["sec"] = raw["dt_exch"].dt.floor("1s")
    raw = raw[raw["sec"].map(two_sided).fillna(False)].copy()

    opt = raw[raw["is_option"] == True].copy()
    opt = opt.dropna(subset=["F_used", "K"]).copy()
    if opt.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta_cols = ["symbol", "K"]
    if "cp" in opt.columns:
        meta_cols.append("cp")
    meta = opt[meta_cols].drop_duplicates(subset=["symbol"]).copy()

    last_quote = (
        opt.sort_values("dt_exch")
        .groupby(["sec", "symbol"], as_index=False)
        .tail(1)
        .set_index(["sec", "symbol"])
    )

    sec_index = pd.DatetimeIndex(sorted(raw["sec"].dropna().unique()))
    f_series = (
        raw[["sec", "F_used"]]
        .dropna()
        .groupby("sec")["F_used"]
        .last()
        .sort_index()
        .reindex(sec_index)
        .ffill()
    )
    sec_meta = (
        raw[["sec", "session_open", "session_close", "session_id"]]
        .drop_duplicates(subset=["sec"])
        .sort_values("sec")
        .set_index("sec")
        .reindex(sec_index)
        .ffill()
    )

    refresh_td = pd.Timedelta(seconds=max(int(pool_refresh_seconds), 0))
    next_refresh_time = None
    last_pool_f = np.nan
    pool_syms: list[str] = []
    pool_exit_since: dict[str, pd.Timestamp] = {}
    rows: list[dict] = []
    detail_rows: list[dict] = []

    for bt in sec_index:
        f_now = float(f_series.loc[bt]) if bt in f_series.index else np.nan
        if not np.isfinite(f_now):
            continue

        refresh_by_time = (next_refresh_time is None) or (int(pool_refresh_seconds) == 0) or (bt >= next_refresh_time)
        refresh_by_f_move = (
            np.isfinite(float(pool_refresh_fut_move))
            and float(pool_refresh_fut_move) > 0.0
            and np.isfinite(float(last_pool_f))
            and abs(float(f_now) - float(last_pool_f)) >= float(pool_refresh_fut_move)
        )
        if refresh_by_time or refresh_by_f_move:
            prev_pool_set = set(pool_syms)
            pool_syms = _pick_pool_symbols(meta, f_now, int(atm_n))
            cur_pool_set = set(pool_syms)

            for sym in cur_pool_set:
                pool_exit_since.pop(sym, None)
            for sym in prev_pool_set - cur_pool_set:
                if sym not in pool_exit_since:
                    pool_exit_since[sym] = pd.Timestamp(bt)

            expired = []
            for sym, exit_ts in pool_exit_since.items():
                if _grace_weight((pd.Timestamp(bt) - pd.Timestamp(exit_ts)).total_seconds()) <= 0.0:
                    expired.append(sym)
            for sym in expired:
                pool_exit_since.pop(sym, None)

            next_refresh_time = bt + refresh_td if int(pool_refresh_seconds) > 0 else pd.Timestamp.max
            last_pool_f = float(f_now)

        active_syms = list(pool_syms)
        for sym, exit_ts in pool_exit_since.items():
            mult = _grace_weight((pd.Timestamp(bt) - pd.Timestamp(exit_ts)).total_seconds())
            if mult > 0.0 and sym not in active_syms:
                active_syms.append(sym)

        used_iv = []
        used_w = []
        flow_sum = 0.0
        n_used = 0
        n_grace = 0
        second_details: list[dict] = []
        for sym in active_syms:
            key = (bt, sym)
            if key not in last_quote.index:
                continue
            row = last_quote.loc[key]
            iv_val = float(row.get("iv", np.nan))
            vega_val = float(row.get("vega", np.nan))
            decay_mult = 1.0 if sym in pool_syms else _grace_weight((pd.Timestamp(bt) - pd.Timestamp(pool_exit_since.get(sym))).total_seconds())
            if not np.isfinite(iv_val) or iv_val <= 0 or not np.isfinite(vega_val) or vega_val <= 0 or decay_mult <= 0.0:
                continue
            used_iv.append(iv_val)
            used_weight = vega_val * decay_mult
            used_w.append(used_weight)
            flow_val = float(row.get("traded_vega_signed", 0.0))
            if np.isfinite(flow_val):
                flow_sum += flow_val
            n_used += 1
            if decay_mult < 0.999:
                n_grace += 1
            second_details.append(
                {
                    "dt_exch": pd.Timestamp(bt),
                    "trade_date": pd.Timestamp(infer_trade_date(bt)),
                    "product": product,
                    "source_file": source_name,
                    "session_id": str(sec_meta.loc[bt, "session_id"]),
                    "session_open": pd.Timestamp(sec_meta.loc[bt, "session_open"]),
                    "session_close": pd.Timestamp(sec_meta.loc[bt, "session_close"]),
                    "symbol": str(sym),
                    "cp": row.get("cp"),
                    "K": float(row.get("K", np.nan)) if pd.notna(row.get("K", np.nan)) else np.nan,
                    "in_pool": bool(sym in pool_syms),
                    "in_grace": bool(sym not in pool_syms),
                    "decay_mult": float(decay_mult),
                    "iv_contract": float(iv_val),
                    "vega_contract": float(vega_val),
                    "vega_weight_used": float(used_weight),
                    "traded_vega_signed": float(flow_val) if np.isfinite(flow_val) else np.nan,
                    "spread": float(row.get("spread", np.nan)) if pd.notna(row.get("spread", np.nan)) else np.nan,
                    "F_used": float(f_now),
                }
            )

        if len(used_iv) == 0:
            continue

        w = np.asarray(used_w, dtype=float)
        iv = float(np.average(np.asarray(used_iv, dtype=float), weights=w)) if np.nansum(w) > 1e-12 else float(np.mean(used_iv))
        for d in second_details:
            d["iv_pool"] = float(iv)
        detail_rows.extend(second_details)
        rows.append(
            {
                "dt_exch": pd.Timestamp(bt),
                "iv_pool": iv,
                "flow": float(flow_sum),
                "F_used": float(f_now),
                "pool_n": int(len(pool_syms)),
                "used_n": int(n_used),
                "grace_n": int(n_grace),
                "session_open": pd.Timestamp(sec_meta.loc[bt, "session_open"]),
                "session_close": pd.Timestamp(sec_meta.loc[bt, "session_close"]),
                "session_id": str(sec_meta.loc[bt, "session_id"]),
            }
        )

    sec = pd.DataFrame(rows).sort_values("dt_exch").dropna(subset=["iv_pool"]).copy()
    if sec.empty:
        return sec, pd.DataFrame(detail_rows)

    iv = sec["iv_pool"]
    flow = sec["flow"].fillna(0.0)

    ema300 = iv.ewm(span=300, adjust=False).mean()
    sec["iv_dev_ema5m_ratio"] = (iv - ema300) / ema300.abs().replace(0, np.nan)
    sec["iv_mom_5m"] = iv - iv.shift(300)

    hh = iv.rolling(300, min_periods=120).max()
    ll = iv.rolling(300, min_periods=120).min()
    sec["iv_willr_5m"] = -100.0 * (hh - iv) / (hh - ll).replace(0, np.nan)

    sec["flow_5m_sum"] = flow.rolling(300, min_periods=60).sum()
    sec["flow_5m_ema"] = flow.ewm(span=300, adjust=False).mean()

    dF = np.log(sec["F_used"] / sec["F_used"].shift(1))
    shockF = dF + 0.5 * dF.shift(1) + 0.25 * dF.shift(2)
    dIV = iv.diff(1)
    beta = dIV.rolling(600, min_periods=240).cov(shockF) / shockF.rolling(600, min_periods=240).var().replace(0, np.nan)
    resid = dIV - beta * shockF
    sec["resid_z_5mctx"] = (
        (resid - resid.rolling(600, min_periods=180).mean())
        / resid.rolling(600, min_periods=180).std().replace(0, np.nan)
    )
    sec["shockF"] = shockF

    sec["future_iv_mean_3_5m"] = _build_future_mean_label(iv, sec["session_id"])
    sec["y_3_5m_mean"] = sec["future_iv_mean_3_5m"] - iv
    sec["product"] = product
    sec["source_file"] = source_name
    sec["trade_date"] = sec["dt_exch"].map(infer_trade_date)
    sec["mins_from_open"] = sec["dt_exch"].map(lambda ts: minutes_from_session_open(ts, sessions))
    sec["mins_to_close"] = sec["dt_exch"].map(lambda ts: minutes_to_session_close(ts, sessions))
    sec["train_allowed"] = (
        sec["mins_from_open"].ge(float(min_train_minutes_after_open))
        & sec["future_iv_mean_3_5m"].notna()
    )
    sec["predict_allowed"] = (
        sec["mins_from_open"].ge(float(min_predict_minutes_after_open))
        & sec["mins_to_close"].gt(float(min_predict_minutes_before_close))
        & sec["future_iv_mean_3_5m"].notna()
    )
    sec["label_ready_ts"] = sec["dt_exch"] + pd.Timedelta(minutes=5)
    details_df = pd.DataFrame(detail_rows)
    if not details_df.empty:
        details_df = details_df.sort_values(["dt_exch", "symbol"]).reset_index(drop=True)
    return sec, details_df


@dataclass
class OnlineLogitState:
    feature_cols: list[str]
    lr: float
    l2: float
    fit_steps: int
    online_steps: int
    min_train: int
    w: np.ndarray | None = None
    mean_: np.ndarray | None = None
    m2_: np.ndarray | None = None
    n_stats: int = 0
    n_train: int = 0
    ready: bool = False
    bootstrap_X: list[np.ndarray] | None = None
    bootstrap_y: list[float] | None = None

    def __post_init__(self) -> None:
        if self.bootstrap_X is None:
            self.bootstrap_X = []
        if self.bootstrap_y is None:
            self.bootstrap_y = []

    def _std(self) -> np.ndarray:
        if self.mean_ is None or self.m2_ is None or self.n_stats <= 1:
            return np.ones(len(self.feature_cols), dtype=float)
        var = self.m2_ / max(1, self.n_stats - 1)
        std = np.sqrt(np.clip(var, 0.0, None))
        std[std < 1e-12] = 1.0
        return std

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            mu = np.zeros(len(self.feature_cols), dtype=float)
            sd = np.ones(len(self.feature_cols), dtype=float)
        else:
            mu = self.mean_
            sd = self._std()
        return np.clip((np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mu) / sd, -8, 8)

    def _update_stats(self, x: np.ndarray) -> None:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.mean_ is None:
            self.mean_ = np.array(x, dtype=float, copy=True)
            self.m2_ = np.zeros_like(self.mean_)
            self.n_stats = 1
            return
        self.n_stats += 1
        delta = x - self.mean_
        self.mean_ += delta / self.n_stats
        delta2 = x - self.mean_
        self.m2_ += delta * delta2

    def add_training_sample(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float)
        y = float(y)
        if not self.ready:
            self.bootstrap_X.append(x)
            self.bootstrap_y.append(y)
            self.n_train += 1
            if self.n_train >= self.min_train:
                X = np.asarray(self.bootstrap_X, dtype=float)
                yv = np.asarray(self.bootstrap_y, dtype=float)
                self.mean_ = np.nanmean(X, axis=0)
                X0 = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                diff = X0 - self.mean_
                self.m2_ = np.sum(diff * diff, axis=0)
                self.n_stats = len(X0)
                Xs = np.clip((X0 - self.mean_) / self._std(), -8, 8)
                Xb = np.c_[np.ones(len(Xs)), Xs]
                self.w = fit_logit_gd(Xb, yv, lr=self.lr, steps=self.fit_steps, l2=self.l2)
                self.ready = True
                self.bootstrap_X = []
                self.bootstrap_y = []
            return

        xs = self._transform(x)
        xb = np.r_[1.0, xs][None, :]
        yb = np.array([y], dtype=float)
        self.w = fit_logit_gd(xb, yb, lr=self.lr, steps=self.online_steps, l2=self.l2, w0=self.w)
        self._update_stats(x)
        self.n_train += 1

    def predict_prob(self, x: np.ndarray) -> float:
        if not self.ready or self.w is None:
            return np.nan
        xs = self._transform(np.asarray(x, dtype=float))
        xb = np.r_[1.0, xs]
        return float(sigmoid(xb @ self.w))

    def save(self, path: Path, as_of: pd.Timestamp | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_cols=np.array(self.feature_cols, dtype=object),
            w=np.array([]) if self.w is None else self.w,
            mean=np.array([]) if self.mean_ is None else self.mean_,
            m2=np.array([]) if self.m2_ is None else self.m2_,
            n_stats=np.array([self.n_stats], dtype=np.int64),
            n_train=np.array([self.n_train], dtype=np.int64),
            ready=np.array([int(self.ready)], dtype=np.int64),
            as_of=np.array(["" if as_of is None else str(pd.Timestamp(as_of))], dtype=object),
        )


def summarize_predictions(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n": 0,
            "n_sig": 0,
            "all_hit": np.nan,
            "triggered_hit": np.nan,
            "coverage": np.nan,
            "avg_vol_decimal": np.nan,
            "avg_vol_points": np.nan,
        }
    trig = df["triggered"] == True
    n = int(len(df))
    n_sig = int(trig.sum())
    all_hit = float((df["pred_sign"] == df["true_sign"]).mean())
    if n_sig > 0:
        triggered_hit = float((df.loc[trig, "pred_sign"] == df.loc[trig, "true_sign"]).mean())
        avg_vol_decimal = float((df.loc[trig, "pred_sign"] * df.loc[trig, "y"]).mean())
    else:
        triggered_hit = np.nan
        avg_vol_decimal = np.nan
    return {
        "n": n,
        "n_sig": n_sig,
        "all_hit": all_hit,
        "triggered_hit": triggered_hit,
        "coverage": float(n_sig / n) if n else np.nan,
        "avg_vol_decimal": avg_vol_decimal,
        "avg_vol_points": float(avg_vol_decimal * 100) if avg_vol_decimal == avg_vol_decimal else np.nan,
    }


def _json_ready(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def augment_visualization_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values("dt_exch").copy()
    flow = pd.to_numeric(out.get("flow"), errors="coerce").fillna(0.0)
    out["cum_flow_trade_date"] = flow.groupby(out["trade_date"]).cumsum()
    out["cum_abs_flow_trade_date"] = flow.abs().groupby(out["trade_date"]).cumsum()
    out["cum_flow_session"] = flow.groupby(out["session_id"]).cumsum()
    out["cum_abs_flow_session"] = flow.abs().groupby(out["session_id"]).cumsum()
    return out


def export_prediction_artifacts(base_dir: Path, time_series_df: pd.DataFrame, details_df: pd.DataFrame) -> None:
    if time_series_df.empty:
        return

    export_root = base_dir / "data" / "prediction" / "logistic"
    export_root.mkdir(parents=True, exist_ok=True)

    for (product, trade_date), grp in time_series_df.groupby(["product", "trade_date"], dropna=False):
        if pd.isna(product) or pd.isna(trade_date):
            continue
        trade_date_ts = pd.Timestamp(trade_date)
        day_dir = export_root / str(product) / str(trade_date_ts.date())
        day_dir.mkdir(parents=True, exist_ok=True)

        ts_out = grp.sort_values("dt_exch").copy()
        keep_cols = [c for c in TIME_SERIES_COLS if c in ts_out.columns]
        ts_out = ts_out[keep_cols]
        ts_out.to_parquet(day_dir / "timeseries.parquet", index=False)
        ts_out.to_parquet(day_dir / "visualization.parquet", index=False)
        ts_out.to_csv(day_dir / "timeseries.csv", index=False)
        day_details = pd.DataFrame()
        if not details_df.empty:
            day_details = details_df[
                (details_df["product"] == product)
                & (details_df["trade_date"] == trade_date_ts)
            ].sort_values(["dt_exch", "symbol"]).copy()
            keep_detail_cols = [c for c in DETAIL_COLS if c in day_details.columns]
            day_details = day_details[keep_detail_cols]
            day_details.to_parquet(day_dir / "details.parquet", index=False)
            day_details.to_csv(day_dir / "details.csv", index=False)

        pred_slice = ts_out[ts_out["p"].notna()].copy() if "p" in ts_out.columns else pd.DataFrame()
        eval_payload = {
            "product": str(product),
            "trade_date": str(trade_date_ts.date()),
            "time_series_rows": int(len(ts_out)),
            "prediction_rows": int(len(pred_slice)),
            "detail_rows": int(len(day_details)),
            "metrics": {k: _json_ready(v) for k, v in summarize_predictions(pred_slice).items()},
            "files": {
                "timeseries_parquet": str(day_dir / "timeseries.parquet"),
                "visualization_parquet": str(day_dir / "visualization.parquet"),
                "timeseries_csv": str(day_dir / "timeseries.csv"),
                "details_parquet": str(day_dir / "details.parquet"),
                "details_csv": str(day_dir / "details.csv"),
            },
        }
        (day_dir / "evaluation.json").write_text(
            json.dumps(eval_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def run_online_backtest(
    panels: list[tuple[pd.DataFrame, pd.DataFrame]],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    state = OnlineLogitState(
        feature_cols=FEATURE_COLS,
        lr=args.lr,
        l2=args.l2,
        fit_steps=args.fit_steps,
        online_steps=args.online_steps,
        min_train=args.min_train,
    )
    pending: list[dict] = []
    predictions: list[dict] = []
    time_series_panels: list[pd.DataFrame] = []
    detail_panels: list[pd.DataFrame] = []
    daily_rows: list[dict] = []
    model_dir = out_dir / "models"
    latest_ts: pd.Timestamp | None = None

    for panel, panel_details in panels:
        panel = panel.copy()
        panel_details = panel_details.copy()
        panel["p"] = np.nan
        panel["conf"] = np.nan
        panel["triggered"] = False
        panel["pred_sign"] = np.nan
        panel["y"] = panel["y_3_5m_mean"]
        panel["true_sign"] = np.where(panel["y_3_5m_mean"] > 0, 1, np.where(panel["y_3_5m_mean"].notna(), -1, np.nan))
        panel["n_train_seen"] = np.nan
        file_name = str(panel["source_file"].iloc[0]) if not panel.empty else ""
        for idx, row in panel.iterrows():
            ts = pd.Timestamp(row.dt_exch)
            latest_ts = ts

            still_pending: list[dict] = []
            for sample in pending:
                if sample["label_ready_ts"] <= ts:
                    state.add_training_sample(sample["x"], sample["y"])
                else:
                    still_pending.append(sample)
            pending = still_pending

            x = np.asarray([row[c] for c in FEATURE_COLS], dtype=float)
            y_raw = float(row.y_3_5m_mean)
            if np.isfinite(y_raw) and bool(row.train_allowed) and np.all(np.isfinite(x)):
                pending.append(
                    {
                        "label_ready_ts": pd.Timestamp(row.label_ready_ts),
                        "x": x,
                        "y": 1.0 if y_raw > 0 else 0.0,
                    }
                )

            if state.ready and bool(row.predict_allowed) and np.all(np.isfinite(x)) and np.isfinite(y_raw):
                p = state.predict_prob(x)
                pred_sign = 1 if p >= 0.5 else -1
                conf = abs(p - 0.5)
                panel.at[idx, "p"] = p
                panel.at[idx, "conf"] = conf
                panel.at[idx, "triggered"] = conf >= args.conf_thr
                panel.at[idx, "pred_sign"] = pred_sign
                panel.at[idx, "n_train_seen"] = state.n_train
                predictions.append(
                    {
                        "dt_exch": ts,
                        "trade_date": pd.Timestamp(row["trade_date"]),
                        "product": row["product"],
                        "source_file": file_name,
                        "p": p,
                        "conf": conf,
                        "triggered": conf >= args.conf_thr,
                        "pred_sign": pred_sign,
                        "y": y_raw,
                        "true_sign": 1 if y_raw > 0 else -1,
                        "n_train_seen": state.n_train,
                    }
                )
        time_series_panels.append(panel)
        if not panel_details.empty:
            detail_panels.append(panel_details)

        if not panel.empty:
            file_pred = pd.DataFrame([r for r in predictions if r["source_file"] == file_name])
            daily = file_pred.groupby("trade_date", dropna=False).apply(summarize_predictions)
            if isinstance(daily, pd.Series):
                for trade_date, metrics in daily.items():
                    row = {"source_file": file_name, "trade_date": str(pd.Timestamp(trade_date).date())}
                    row.update(metrics)
                    daily_rows.append(row)
                    state.save(model_dir / f"TA_model_{pd.Timestamp(trade_date).date()}.npz", pd.Timestamp(trade_date))

    pred_df = pd.DataFrame(predictions)
    time_series_df = pd.concat(time_series_panels, ignore_index=True) if time_series_panels else pd.DataFrame()
    time_series_df = augment_visualization_frame(time_series_df)
    details_df = pd.concat(detail_panels, ignore_index=True) if detail_panels else pd.DataFrame()
    state.save(model_dir / "TA_model_latest.npz", latest_ts)
    return pred_df, time_series_df, details_df, {
        "overall": summarize_predictions(pred_df),
        "daily": daily_rows,
    }


def load_panels(
    input_dir: Path,
    pattern: str,
    atm_n: int,
    pool_refresh_seconds: int,
    pool_refresh_fut_move: float,
    sessions: list[SessionSpec],
    min_train_minutes_after_open: float,
    min_predict_minutes_after_open: float,
    min_predict_minutes_before_close: float,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    files = sorted(input_dir.rglob(pattern))
    panels: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fp in files:
        raw = pd.read_parquet(fp)
        raw["dt_exch"] = pd.to_datetime(raw["dt_exch"])
        panels.append(
            build_second_panel(
                raw,
                atm_n=atm_n,
                product=fp.parent.name,
                source_name=fp.name,
                pool_refresh_seconds=pool_refresh_seconds,
                pool_refresh_fut_move=pool_refresh_fut_move,
                sessions=sessions,
                min_train_minutes_after_open=min_train_minutes_after_open,
                min_predict_minutes_after_open=min_predict_minutes_after_open,
                min_predict_minutes_before_close=min_predict_minutes_before_close,
            )
        )
    return panels


def main() -> None:
    args = parse_args()
    sessions = parse_trading_sessions(args.trading_sessions)
    repo_root = Path(__file__).resolve().parents[3]
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = load_panels(
        input_dir,
        args.glob,
        args.atm_n,
        args.pool_refresh_seconds,
        args.pool_refresh_fut_move,
        sessions,
        args.min_train_minutes_after_open,
        args.min_predict_minutes_after_open,
        args.min_predict_minutes_before_close,
    )
    pred_df, time_series_df, details_df, summary = run_online_backtest(panels, args, out_dir)

    pred_path = out_dir / "ta_v15_predictions.parquet"
    daily_path = out_dir / "ta_v15_daily_summary.parquet"
    ts_path = out_dir / "ta_v15_timeseries.parquet"
    summary_path = out_dir / "ta_v15_summary.json"

    pred_df.to_parquet(pred_path, index=False)
    pd.DataFrame(summary["daily"]).to_parquet(daily_path, index=False)
    time_series_df.to_parquet(ts_path, index=False)
    export_prediction_artifacts(repo_root, time_series_df, details_df)

    out = {
        "config": {
            "input_dir": str(input_dir),
            "glob": args.glob,
            "atm_n": args.atm_n,
            "trading_sessions": args.trading_sessions,
            "pool_refresh_seconds": args.pool_refresh_seconds,
            "pool_refresh_fut_move": args.pool_refresh_fut_move,
            "min_train_minutes_after_open": args.min_train_minutes_after_open,
            "min_predict_minutes_after_open": args.min_predict_minutes_after_open,
            "min_predict_minutes_before_close": args.min_predict_minutes_before_close,
            "conf_thr": args.conf_thr,
            "min_train": args.min_train,
            "fit_steps": args.fit_steps,
            "online_steps": args.online_steps,
            "lr": args.lr,
            "l2": args.l2,
            "label": "mean(IV[t+3m:t+5m]) - IV[t]",
            "train_gate": "training rows must be inside configured trading sessions, at least min_train_minutes_after_open after session open, and have an in-session 3m-5m future label",
            "predict_gate": "prediction rows must be inside configured trading sessions, at least min_predict_minutes_after_open after session open, more than min_predict_minutes_before_close before session close, and have an in-session 3m-5m future label",
            "warmup_policy": "first run requires min_train labeled samples; later days continue with persisted in-memory model",
            "filter": "remove one-sided future seconds",
            "pool_rule": "ATM strike forced in, non-ATM OTM only, refresh by seconds or |ΔF|, 5min exit grace decay",
        },
        "metrics": summary["overall"],
        "daily_metrics": summary["daily"],
        "artifacts": {
            "predictions": str(pred_path),
            "daily_summary": str(daily_path),
            "time_series": str(ts_path),
            "prediction_export_root": str(repo_root / "data" / "prediction" / "logistic"),
            "models_dir": str(out_dir / "models"),
        },
    }
    summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
