import pandas as pd
import numpy as np


def build_iv_price_factors(df: pd.DataFrame, iv_col: str = "iv") -> pd.DataFrame:
    x = df.copy().sort_values("dt_exch")
    x["iv_ret_1"] = x[iv_col].pct_change()
    x["iv_ret_5"] = x[iv_col].pct_change(5)
    x["iv_mom_30"] = x[iv_col] - x[iv_col].rolling(30).mean()
    x["iv_vol_60"] = x["iv_ret_1"].rolling(60).std()
    x["fut_ret_1"] = x["F_used"].pct_change()
    x["fut_ret_5"] = x["F_used"].pct_change(5)
    x["fut_micro_dev"] = (x["fut_microprice"] - x["F_used"]) / x["F_used"]
    return x
