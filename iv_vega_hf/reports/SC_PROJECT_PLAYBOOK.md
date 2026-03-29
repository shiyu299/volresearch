# SC 项目完整流程说明（IV 预测与交易信号）

更新时间：2026-03-21（Asia/Shanghai）
适用范围：`SC2604` 期权链，分钟级主线（秒级作为辅助）

---

## 0. 项目目标

目标不是预测价格本身，而是预测 **IV 的未来变化**，并将预测转成可交易信号。

- 预测目标：`delta_iv_h = IV(t+h) - IV(t)`
- 核心评估：
  - all_hit（全样本命中率）
  - triggered_hit（触发后命中率）
  - coverage（触发覆盖率）
  - avg_vol（每笔平均 vol 变化，decimal 与 vol points）

---

## 1. 数据来源与范围

数据根目录：`volresearch/data/derived/`

本项目先做 `SC`：
- 过滤规则：`symbol` 包含 `sc2604`
- 主线口径：ATM 池化（与 app 主图思路一致）
- 对照口径：单合约（成交量最大合约）

> 说明：之前只扫根目录会漏子目录，后已确认子目录存在 TA/FG/MA/PL/SH 数据，后续按品种可复用同流程。

---

## 2. 特征构建流程

### 2.1 ATM 池化序列构造（分钟级）

1) 从 `sc2604` 子集里筛 option 行
2) 每个时刻按 `|K - F_used|` 取 ATM 附近 `n` 个合约（当前常用 n=20）
3) 计算池化 IV：
- 优先 vega 加权平均
- 无有效权重时退化为简单均值
4) 同步聚合：
- `flow = sum(traded_vega_signed)`
- `spread = mean(spread)`
- `F = last(F_used)`

### 2.2 基础技术因子（核心）

- `iv_dev_ema5_ratio = (IV - EMA5(IV)) / abs(EMA5(IV))`
- `iv_mom3 = IV_t - IV_{t-3}`
- `iv_willr10 = -100 * (HH10 - IV) / (HH10 - LL10)`
- `flow`（有符号 vega 净流）

### 2.3 F-IV 因子（已确定策略）

- `shockF = dF_t + 0.5*dF_{t-1} + 0.25*dF_{t-2}`
- rolling `betaF` 估计 dIV 对 shockF 的敏感度
- `resid = dIV - betaF * shockF`
- `resid_z = zscore(resid)`

使用原则：
- `resid_z` 作为交易因子（可入模）
- `betaF` / `R2分位` / 稳定性 / CUSUM 作为过滤与体检（不一定并入主打分）

---

## 3. 模型流程（主线）

模型：walk-forward Logistic Regression（二分类）

- 标签：`up/down = sign(delta_iv_h) > 0`
- 特征：核心因子 + 扩展候选因子
- 训练方式：每个时点只用历史窗口训练，再预测下一时点（防未来泄露）
- 输出：上涨概率 `p`

信号强度定义：
- 置信度：`conf = |p - 0.5|`
- 触发条件：`conf >= threshold`（例如 0.10 / 0.15）

---

## 4. 指标定义（飞书纯文本口径）

- all_hit：所有 OOS 预测点的方向命中率
- triggered_hit：仅在触发点（conf 超阈值）上的命中率
- coverage：触发点数量 / 总预测点数量
- avg_vol_decimal：mean(sign(pred) * delta_iv)
- avg_vol_points：`avg_vol_decimal * 100`

示例：
- `avg_vol_decimal = 0.02` 表示 2 vol points

---

## 5. 因子筛选策略

### 5.1 v1 核心骨架（固定）

- `flow`
- `iv_dev_ema5_ratio`
- `iv_mom3`
- `iv_willr10`
- `resid_z`

### 5.2 扩展筛选（分批）

候选包含（示例）：
- `flow_ema3`, `flow_ema10`, `flow_lag1`
- `spread_chg1`, `spread_ema10`
- `iv_roc3`, `iv_rsi6`, `bb_z20`
- `shockF`, `shockF_abs`, `resid_z_abs`, `flow_x_dev`

筛选标准：
1) 对比基线模型的边际提升（triggered_hit, avg_vol, coverage）
2) 看是否稳定提升而非一次性偶然
3) 覆盖率过低/收益不稳则降权或淘汰

---

## 6. 当前已确认的中间结论（SC）

1) 分钟级明显比秒级稳
2) ATM 池化口径优于单合约口径
3) `iv_dev_ema5_ratio` 比 zscore 版更稳（总体）
4) 第一批扩展里：
- `flow_ema10`：较均衡（hit 与 avg_vol 兼顾）
- `shockF`：hit 提升明显，但需关注 avg_vol 的权衡

---

## 7. 运行顺序（建议执行 SOP）

1) 数据准备：SC 子集 + ATM 池化
2) 构建核心因子 + F-IV 因子
3) 训练 baseline logit（core）
4) 加入 `resid_z` 做对照
5) 扩展因子逐个/分批加入，记录边际变化
6) 形成 v2 保留清单（keep/drop + 理由）
7) 固化报告：
- 排名表
- 指标全表
- 推荐阈值与可交易解释

---

## 8. 本地文件索引

- 核心骨架：`iv_vega_hf/spec/core_factor_skeleton_v1.md`
- 状态检查：`iv_vega_hf/reports/SC_V2_RESULTS_STATUS.md`
- 过程说明：`iv_vega_hf/reports/SC_V2_PROCESS_NOTES.md`
- 小时日志：`iv_vega_hf/PROGRESS_LOG.md`

---

## 9. 后续扩展（在 SC 完整收口后）

按品种复跑同流程：TA / FG / MA / PL / SH

每个品种输出同一模板：
- 因子排名
- v2 组合
- all_hit / triggered_hit / coverage / avg_vol_points
- 口径一致，便于横向比较
