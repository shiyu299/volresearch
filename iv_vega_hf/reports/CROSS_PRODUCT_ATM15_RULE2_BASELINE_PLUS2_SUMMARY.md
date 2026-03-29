# 新规则重跑结果（剔除期货单边市分钟）

口径：ATM n=15, h=5m, conf>=0.15
规则：期货分钟内若无双边报价（bid/ask 任一侧缺失或<=0），该分钟全量剔除（不训练、不评估、不触发）。

## SC
- baseline: all_hit=0.5842, trig_hit=0.7308, cov=0.0932, avg_vol_pts=1.7366
- +flow_ema10: all_hit=0.5914, trig_hit=0.5806, cov=0.2222, avg_vol_pts=1.4156
- +shockF: all_hit=0.5914, trig_hit=0.7568, cov=0.1326, avg_vol_pts=2.5931
- +both: all_hit=0.5842, trig_hit=0.6406, cov=0.2294, avg_vol_pts=1.7069

## TA
- baseline: all_hit=0.5389, trig_hit=0.6379, cov=0.0759, avg_vol_pts=0.6697
- +flow_ema10: 0.5304, 0.6364, 0.1079, 0.7157
- +shockF: 0.5428, 0.6387, 0.1014, 0.7504
- +both: 0.5324, 0.6477, 0.1262, 0.5676

## MA
- baseline: all_hit=0.5795, trig_hit=0.7030, cov=0.1471, avg_vol_pts=0.6254
- +flow_ema10: 0.5799, 0.6763, 0.1631, 0.5651
- +shockF: 0.5799, 0.6950, 0.1540, 0.5730
- +both: 0.5788, 0.6812, 0.1667, 0.5457

## FG
- baseline: all_hit=0.6738, trig_hit=0.7722, cov=0.4748, avg_vol_pts=0.8369
- +flow_ema10: 0.6745, 0.7780, 0.4791, 0.8518
- +shockF: 0.6728, 0.7692, 0.4701, 0.8369
- +both: 0.6738, 0.7799, 0.4783, 0.8657

## SH（SA 对应）
- baseline: all_hit=0.5835, trig_hit=0.7218, cov=0.0969, avg_vol_pts=0.6758
- +flow_ema10: 0.5805, 0.6795, 0.1330, 0.4764
- +shockF: 0.5824, 0.6894, 0.1173, 0.6328
- +both: 0.5798, 0.6789, 0.1487, 0.5089

原始文件：
- `iv_vega_hf/reports/CROSS_PRODUCT_ATM15_RULE2_BASELINE_PLUS2.json`
