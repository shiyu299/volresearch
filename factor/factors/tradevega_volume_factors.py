import pandas as pd


def build_tradevega_factors(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("dt_exch")
    tv = x.get("traded_vega_signed", x.get("traded_vega", 0)).fillna(0.0)
    x["tv_signed"] = tv
    x["tv_abs"] = tv.abs()
    x["tv_imbalance_30"] = tv.rolling(30).sum() / (tv.abs().rolling(30).sum() + 1e-9)
    x["tv_z_60"] = (tv - tv.rolling(60).mean()) / (tv.rolling(60).std() + 1e-9)
    x["dvol_30"] = x.get("d_volume", 0).fillna(0.0).rolling(30).sum()
    return x
