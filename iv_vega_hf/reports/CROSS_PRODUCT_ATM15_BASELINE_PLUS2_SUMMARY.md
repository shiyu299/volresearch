# TA/MA/FG/SH 结果汇总（ATM n=15, h=5m, conf>=0.15）

说明：用户提到 SA，本地数据口径中 SA 对应 SH。

## Baseline（core）
- TA: all_hit=0.5152, triggered_hit=0.7385, coverage=0.0991, avg_vol_points=0.6471
- MA: all_hit=0.6003, triggered_hit=0.7147, coverage=0.2627, avg_vol_points=0.4442
- FG: all_hit=0.6707, triggered_hit=0.7649, coverage=0.5807, avg_vol_points=0.6802
- SH: all_hit=0.5822, triggered_hit=0.6250, coverage=0.1374, avg_vol_points=0.1810

## Baseline + flow_ema10
- TA: all_hit=0.5091, triggered_hit=0.7188, coverage=0.0976, avg_vol_points=0.6406
- MA: all_hit=0.6003, triggered_hit=0.7073, coverage=0.2643, avg_vol_points=0.4324
- FG: all_hit=0.6691, triggered_hit=0.7684, coverage=0.5791, avg_vol_points=0.6872
- SH: all_hit=0.5879, triggered_hit=0.6000, coverage=0.1594, avg_vol_points=0.1678

## Baseline + shockF
- TA: all_hit=0.5396, triggered_hit=0.6510, coverage=0.2271, avg_vol_points=0.3903
- MA: all_hit=0.6003, triggered_hit=0.7108, coverage=0.2981, avg_vol_points=0.4322
- FG: all_hit=0.6707, triggered_hit=0.7670, coverage=0.5791, avg_vol_points=0.6816
- SH: all_hit=0.5814, triggered_hit=0.6243, coverage=0.1545, avg_vol_points=0.2430

## 两个都加（flow_ema10 + shockF）
- TA: all_hit=0.5457, triggered_hit=0.6312, coverage=0.2149, avg_vol_points=0.3516
- MA: all_hit=0.5923, triggered_hit=0.7166, coverage=0.3014, avg_vol_points=0.4431
- FG: all_hit=0.6667, triggered_hit=0.7681, coverage=0.5783, avg_vol_points=0.6861
- SH: all_hit=0.5846, triggered_hit=0.5982, coverage=0.1832, avg_vol_points=0.1942

原始明细 JSON：
- `iv_vega_hf/reports/CROSS_PRODUCT_ATM15_BASELINE_PLUS2.json`
