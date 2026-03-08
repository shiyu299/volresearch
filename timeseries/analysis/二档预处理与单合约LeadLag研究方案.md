# 二档预处理与单合约 Lead-Lag 研究方案（先文档，不改代码）

> 版本：v1（供明天评审）
> 
> 目标：
> 1) 基于你现有 `marketvolseries_modified_v4.py` 设计一版“包含二档信息”的预处理产物；
> 2) 先把“单合约是否领先全曲面”的研究方法定清楚（不急着写代码）。

---

## 0. 你当前基础（已存在）

当前脚本：`volresearch/timeseries/marketvolseries_modified_v4.py`

已有能力（你这版已做得不错）：
- Black-76 IV / Vega / Delta
- 期货价格融合（microprice + mid fallback）
- 成交识别（累计值差分）
- `spread_limit` 过滤
- 期权/期货统一事件流

这版方案是在你当前逻辑上“加层”，不推翻。

---

## 1. 新预处理目标（“二档vol”版）

产出一份新 parquet（建议命名）：
- `*_option_iv_vega_traded_v5_l2.parquet`

核心变化：在 v4 基础上新增“二档+订单簿微结构”字段，后续因子可直接复用。

### 1.1 必增字段（最小集合）

#### A) 二档价格与价差
- `bidprice2`, `askprice2`（原始）
- `mid_l1 = (ask1+bid1)/2`
- `mid_l2 = (ask2+bid2)/2`
- `spread_l1 = ask1-bid1`
- `spread_l2 = ask2-bid2`
- `depth_spread_ratio = spread_l2 / max(spread_l1, eps)`

#### B) 二档深度与不平衡
- `depth_l1 = bidvol1 + askvol1`
- `depth_l2 = bidvol2 + askvol2`
- `depth_total_2 = depth_l1 + depth_l2`
- `obi_l1 = (bidvol1-askvol1)/(bidvol1+askvol1+eps)`
- `obi_l2 = (bidvol2-askvol2)/(bidvol2+askvol2+eps)`
- `obi_2lvl = ((bid1+bid2)-(ask1+ask2))/((bid1+bid2)+(ask1+ask2)+eps)`（这里 bid1 等代表对应挂单量）

#### C) 价格重心与层间斜率
- `micro_l1 = (ask1*bidvol1 + bid1*askvol1)/(bidvol1+askvol1+eps)`
- `micro_l2 = (ask2*bidvol2 + bid2*askvol2)/(bidvol2+askvol2+eps)`
- `micro_shift_12 = micro_l2 - micro_l1`
- `book_slope_bid = (bidprice1-bidprice2)/max(bidvol1+bidvol2, eps)`
- `book_slope_ask = (askprice2-askprice1)/max(askvol1+askvol2, eps)`

#### D) 与现有事件字段联动（用于后续因子）
- 保留：`traded_vega`, `traded_vega_signed`, `trade_volume_lots`, `iv`, `delta`, `F_used`
- 新增滚动窗口特征（1s panel 再生成）：
  - `obi_2lvl_30s_mean`
  - `depth_total_2_30s_mean`
  - `micro_shift_12_30s_mean`
  - `depth_spread_ratio_30s_med`

---

## 2. 数据清洗与稳健规则（建议强制）

1) **盘口有效性**：
- 要求 `ask1>=bid1`，`ask2>=bid2`；否则该时刻二档字段置空。

2) **层级一致性**：
- 正常情况下 `bid1>=bid2`、`ask1<=ask2`；不满足时标记 `book_cross_flag=1`。

3) **极值裁剪**：
- 对 `obi_*`、`micro_shift_12`、`depth_spread_ratio` 进行 winsorize（如 0.1%~99.9%）。

4) **缺失处理**：
- 二档缺失不回填到很远，只允许短窗 ffill（<=2秒）；超窗置空。

5) **会话边界重置**：
- 早盘/午盘/夜盘切换时，不跨会话继承 rolling 状态。

---

## 3. 单合约 Lead-Lag：先定义问题，再写代码

你提的问题可以精确定义为：

> “某单合约成交量（或成交vega）异常放大，是否在统计上领先全曲面关键状态变化？”

### 3.1 被解释变量（“全曲面状态”）

先定 2 个主目标，避免泛化过头：

- `Y1`: 全曲面 ATM-IV 变动（如 30s/60s）
- `Y2`: 全曲面风险指标（如曲面斜率变化、skew变化、abs(traded_vega)_panel）

建议先以 `Y1` 起步，验证最快。

### 3.2 解释变量（“单合约冲击”）

针对每个候选合约 i：
- `X_i1 = trade_volume_lots_i`（或其 zscore）
- `X_i2 = abs(traded_vega_i)`
- `X_i3 = signed_traded_vega_i`
- `X_i4 = volume_spike_i`（是否超过该合约滚动Q95）

### 3.3 Lead-Lag 检验框架

#### 方法A：滞后相关（先做）
- 计算 `corr(X_i(t), Y(t+lag))`, lag ∈ [-120s, +120s]
- 若在 `lag>0` 区间峰值显著，支持“X领先Y”。

#### 方法B：事件窗（最直观）
- 触发事件：`X_i` 超过 Q95/Q99
- 观察窗口：[-60s, +180s]
- 看 `Y` 在 0 后的累计变化是否显著偏离 0

#### 方法C：回归/Granger（第二阶段）
- `Y_t = a + Σ b_k X_{t-k} + controls + e_t`
- 检验 `b_k (k>0)` 联合显著性

---

## 4. 显著性与“真领先”判定标准（先定规矩）

为防止“看图说话”，建议提前定判据：

1) 峰值相关 lag 在 `+5s ~ +60s` 且稳定出现（跨日一致）
2) 事件窗中 `post(0~60s)` 的 Y 累计变化显著（bootstrap CI 不含0）
3) 去掉高波动时段后结论不反转（稳健性）
4) 至少 3 天样本重复

满足 2 条以上再称“有领先证据”。

---

## 5. 控制变量（避免伪领先）

必须控制：
- 同期期货收益/成交量
- 时段效应（早/午/夜）
- 合约 moneyness 与到期剩余时间
- 盘口整体流动性（spread/depth）

否则容易把“共同受市场冲击”误判成“单合约领先”。

---

## 6. 明天落地执行清单（不改代码版）

### Step 1：确认口径（你拍板）
- Lead-Lag 的 Y 用哪一个（先 ATM-IV 还是先曲面斜率）
- 事件阈值用 Q95 还是 Q99
- 窗口长度（建议 [-60,+180]）

### Step 2：确认 v5_l2 字段清单
- 是否按上面 A/B/C/D 全量
- 是否先只做 TA/SC 两个品种试点

### Step 3：产物约定
建议固定输出：
- `tables_leadlag/<symbol>_lag_corr.csv`
- `tables_leadlag/<symbol>_event_window.csv`
- `tables_leadlag/<symbol>_leadlag_summary.md`

---

## 7. 你关心的“通用接口兼容未来因子”怎么实现（设计层）

和 app 的思路一致，预处理层也做注册式：

- `FactorInputBuilder`：只负责构建统一底表（含二档字段）
- `FactorRegistry`：每个因子只注册 `id/compute/threshold_default`
- `TriggerEngine`：统一输出 `signal/trigger_time/threshold`

这样后面新增因子只加一段 compute，不改主流程。

---

## 8. 结论（给明天的你）

- 先别急着直接上代码，当前最重要是把**口径和判据先定死**；
- 二档预处理能明显提升你后续“领先性”研究的解释力；
- Lead-Lag 第一版建议用“滞后相关 + 事件窗”双轨，快且稳。

---

如果你明天确认这个方案，我会按这个文档直接拆成：
1) v5_l2 预处理实现清单；
2) lead-lag 实验脚本清单；
3) 报告模板（自动落表 + 自动图）。
