import pandas as pd


def build_orderbook_factors(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("dt_exch")
    bp = x.get("bidprice1")
    ap = x.get("askprice1")
    x["spread_abs"] = (ap - bp)
    x["spread_rel"] = (ap - bp) / ((ap + bp) / 2.0)
    x["book_imbalance"] = x.get("fut_book_imbalance")
    x["book_pressure_ema"] = x["book_imbalance"].ewm(span=30, adjust=False).mean()
    return x
