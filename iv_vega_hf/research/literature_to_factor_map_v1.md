# Literature → Factor Map (v1)

> 目标：把文献中的可复现信号，映射成 `iv_vega_hf` 可直接实现的特征。

## A. 订单流 / 需求压力（Option demand pressure）

### 代表文献
- Bollen, N. & Whaley, R. (2004): Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?
- Garleanu, Pedersen, Poteshman (2009): Demand-based option pricing

### 可落地因子
- `flow_pressure = vega_signed_1m / (vega_abs_1m + eps)`
- `flow_pressure_ewm_{5,15,30}`
- `signed_flow_regime`（滚动符号一致性）
- `shock_x_flow = vega_shock_z30 * flow_pressure`

---

## B. IV 动量 / 均值回复 / 技术指标

### 代表文献方向
- IV 动态与均值回复（大量实证文献）
- 技术指标在波动率序列上的应用（经验研究）

### 可落地因子
- `iv_mom_{5,15,30}`
- `iv_roc_{5,10,20}`
- `iv_rsi_{6,14,21}`
- `iv_bb_z20`（布林带 zscore）
- `iv_kalman_like_resid`（后续可加）

---

## C. IV-RV 关系（HAR-RV/nowcasting 系）

### 代表文献
- Corsi (2009): HAR-RV
- Andersen et al. 高频 realized volatility 系列

### 可落地因子（当前用期货替代）
- `f_rv_5`, `f_rv_15`, `f_rv_30`（期货 1m 收益滚动波动）
- `iv_minus_frv = iv_pool - f_rv_15`
- `spread_x_rv = spread_pool_1m * f_rv_15`

---

## D. 流动性与微观结构

### 代表文献方向
- Bid-ask / liquidity / transaction cost 对可预测性的影响

### 可落地因子
- `spread_pool_1m`
- `liquidity_stress = spread_pool_1m * vega_abs_1m`
- `d_spread_1m`, `spread_ewm_10`

---

## E. 非传统（跳出传统）

### 可落地因子（先做 proxy）
- regime 因子：
  - `regime_vol = qcut(f_rv_15)`
  - `regime_liq = qcut(spread_pool_1m)`
- 交互因子：
  - `mom_x_pressure = iv_mom_5 * flow_pressure_ewm15`
  - `shock_x_rsi = vega_shock_z30 * iv_rsi14`
- 分层模型：按 regime 训练子模型（高波 / 低波）

---

## 下一步实现清单（直接进代码）
1. 把 `iv_minus_frv`, `d_spread_1m`, `spread_ewm_10` 加入 zoo。
2. 增加 regime 分层评估（先二分：高波/低波）。
3. 对每个 horizon 输出：
   - 全样本 top IC
   - regime 内 top IC
   - greedy 最终特征集合
4. 仅保留“全样本 + regime 内都稳定”的因子进 v2 白名单。
