# Core Factor Skeleton v1 (SC-only, ATM-mainline)

更新时间：2026-03-21
状态：active baseline

## 核心骨架（固定保留）

1. `flow`
- 定义：当bar内 `traded_vega_signed` 求和
- 角色：需求压力主因子

2. `iv_dev_ema5_ratio`
- 定义：`(IV - EMA5(IV)) / abs(EMA5(IV))`
- 角色：短期偏离/均值回复主因子（ratio 版本为主）

3. `iv_mom3`
- 定义：`IV_t - IV_{t-3}`
- 角色：短动量主因子

4. `iv_willr10`
- 定义：Williams %R on IV, window=10
- 角色：短窗口位置/过热过冷主因子

5. `resid_z`（F-IV 唯一交易因子）
- 定义：`dIV - beta_t * shockF` 的滚动标准化残差
- 角色：F-IV 偏离交易信号

## F-IV 过滤条件（不并入主打分）

- `betaF` 稳定性
- `R2_percentile`（相对分位）
- `CUSUM/regime` 报警

## 使用口径

- 目标：预测 `ΔIV_h`
- 频率：分钟级主线（秒级辅助）
- 评估：walk-forward + 触发后 hit + coverage + avg vol points
