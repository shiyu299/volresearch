# IV / Vega / 期货联动深度分析报告 v3（PL26+PL27联合样本）

## 1. 本版口径（v3）
- 事件定义改为“非零成交vega样本”的 μ+3σ / μ+4σ 双阈值。
- PL26 与 PL27 合并为 PL联合样本，用于信号挖掘（训练/发现）。
- 检测评估统一在“所有样本秒”（含无成交秒）进行，报告 precision / recall / triggers per hour / FPR proxy。
- 信号按 lift、precision、噪声(FPR)综合分级 A/B/C；主文仅保留 A/B。

## 2. 事件阈值（非零样本）
- PL联合样本：μ=1486.8770, σ=4766.9441, μ+3σ=15787.7094, μ+4σ=20554.6535
- 联合样本基准事件率：event_3s=0.000127, event_4s=0.000093（分母=所有秒）
- 详见 tables_v3/event_thresholds_mu_sigma.csv

## 3. PL联合样本信号挖掘结果（主文仅A/B）
- 本次阈值下未出现A/B信号（详见附录C级）。
- 双阈值（μ+3σ / μ+4σ）对比见 tables_v3/summary_union_main_signals.csv

## 4. 检测评估（所有样本秒）
- 评估对象：PL26、PL27、PL联合样本；均包含无成交秒。
- 指标：precision / recall / triggers per hour / non-event trigger rate(FPR proxy)。
- 全量结果：tables_v3/signal_eval_all_seconds.csv
- 主文信号表（A/B）：tables_v3/signal_main_ab_only.csv

## 5. no_trade_move 指标完整解释
### 5.1 交易员口径
- 当过去60秒期货价格出现明显变化，但成交/主动成交没有同步放量时，视作“无量波动”风险升高。
### 5.2 数学定义
- no_trade_move_60 = |fut_ret_60s| / (fut_dvol_60s + 1)
- 其中 fut_ret_60s 为60秒收益率，fut_dvol_60s 为60秒内期货增量成交量。+1 用于避免分母为0。
### 5.3 真实时间戳样例
- [PL26] 2026-02-26 06:34:30: no_trade_move_60=0.000635, fut_ret_60s=0.000635, fut_dvol_60s=0.00, lots=0.00, abs_vega_1s=0.0000
- [PL26] 2026-02-26 06:34:31: no_trade_move_60=0.000556, fut_ret_60s=0.000556, fut_dvol_60s=0.00, lots=0.00, abs_vega_1s=0.0000
- [PL26] 2026-02-26 06:34:32: no_trade_move_60=0.000476, fut_ret_60s=0.000476, fut_dvol_60s=0.00, lots=0.00, abs_vega_1s=0.0000
- [PL27] 2026-02-27 06:22:27: no_trade_move_60=0.000403, fut_ret_60s=-0.000403, fut_dvol_60s=0.00, lots=0.00, abs_vega_1s=0.0000
- [PL27] 2026-02-27 06:22:26: no_trade_move_60=0.000403, fut_ret_60s=-0.000403, fut_dvol_60s=0.00, lots=0.00, abs_vega_1s=0.0000
- [PL27] 2026-02-27 03:05:53: no_trade_move_60=0.000402, fut_ret_60s=-0.000402, fut_dvol_60s=0.00, lots=0.00, abs_vega_1s=0.0000
- 样例表：tables_v3/no_trade_move_real_timestamp_examples.csv

## 6. 附录（C级信号简述）
- sig_no_trade_move_60: 等级C（lift=1.82, precision=0.0002, FPR=0.1997），噪声或区分度不足。
- sig_opt_spread_60s: 等级C（lift=0.00, precision=0.0000, FPR=0.0007），噪声或区分度不足。
- sig_iv_f_div_60: 等级C（lift=0.56, precision=0.0001, FPR=0.8152），噪声或区分度不足。
- sig_fut_imb: 等级C（lift=1.81, precision=0.0002, FPR=0.0502），噪声或区分度不足。
- sig_combo_core: 等级C（lift=0.00, precision=0.0000, FPR=0.0004），噪声或区分度不足。
- C级详细数值：tables_v3/signal_appendix_c_only.csv

## 7. 结论（简明）
- 用非零样本 μ+3σ/μ+4σ 定义事件后，可把“极端vega秒”与普通无成交秒在同一检测框架下评估噪声。
- PL联合样本训练可提升信号稳定性，避免单合约样本偏差。
- 主文仅保留A/B信号，降低监控冗余和误报负担。

为什么q99口径不适配当前目标：q99是分位截断，只保证样本内比例，不反映非零成交分布的波动结构（μ、σ），因此不利于在“全秒级噪声控制”目标下统一比较 precision 与 FPR。