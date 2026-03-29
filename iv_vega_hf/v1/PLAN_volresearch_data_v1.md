# PLAN v1 — 基于 `volresearch/` 真实数据预测后续 IV 变化

更新时间：2026-03-20

## 0. 我已核对到的可用数据与入口

- 数据目录：`volresearch/data/derived/`
- 已见样本文件：
  - `PL26_iv.parquet`（约 20 万行）
  - `sc2604_iv_20260313.parquet`（约 205 万行）
- 核心字段（已存在）：
  - 时间：`dt_exch`
  - 期货价格主图可用：`F_used`
  - IV：`iv`
  - 成交 vega：`traded_vega`、`traded_vega_signed`
  - 合约信息：`symbol`、`is_option`、`cp`、`K`
  - 其他：`vega`、`delta`、`volume`、`d_volume`、`spread` 等
- 可视化入口（你说的 app）：`volresearch/timeseries/iv_inspector_refactor_toggle/appclaude.py`
  - 主图本质：期货（F_used）+ IV K线 + 成交 vega 柱。

---

## 1. 目标定义（锁死）

**目标：预测未来 IV 变化**，先做 Level A：
- `y_h = iv_{t+h} - iv_t`
- `h ∈ {1, 5, 15}`（分钟）

输出两套可比较的数据建模口径：

### 路线 A（主图口径）
聚合“ATM附近合约池”的 IV 与成交 vega，形成时间序列做预测。

### 路线 B（三大成交合约口径）
每日选成交量最大的 3 个期权合约，分别建模并做组合/汇总评估。

> 先并行做 A/B，再决定后续主线。

---

## 2. 数据方案

## 2.1 路线 A：主图口径数据构造

以 `dt_exch` 按 1min 重采样，构建：
- `iv_pool_t`：当分钟合约池（ATM±n，默认 n=20~30）的加权 IV
  - 备选权重：`vega` 权重、等权
- `vega_signed_1m`：当分钟 `traded_vega_signed` 总和
- `vega_abs_1m`：当分钟 `abs(traded_vega_signed)` 总和
- `f_ret_1m`：`log(F_used_t/F_used_{t-1})`
- `spread_pool_1m`：池内平均 spread

标签：
- `y_1m/y_5m/y_15m = iv_pool_{t+h} - iv_pool_t`

## 2.2 路线 B：Top3 合约口径数据构造

按交易日 D：
1. 在日内按 `symbol` 聚合成交量（优先 `d_volume` 或 `volume` 增量）
2. 选 Top3 期权合约（`is_option=True`）
3. 每个合约独立构造 1min 序列与特征：
   - `iv_t`, `iv_delta_1m`
   - `traded_vega_signed_1m`, `traded_vega_abs_1m`
   - `f_ret_1m`（同 underlying 的 `F_used`）
   - `spread_1m`
4. 生成每个合约的 `y_h`

评估层可做两种汇总：
- Top3 等权平均指标
- 按合约成交量加权指标

---

## 3. 特征与模型

## 3.1 第一批特征（两路线共用）
- IV 自身：`iv_delta_1m`, `iv_delta_5m`, `iv_zscore_30m`
- vega 流：`vega_signed_1m/5m/15m`, `vega_abs_1m/5m`, `vega_imbalance_5m`
- 微观结构：`spread_1m`, `trade_count_1m`（可得则上）
- 期货联动：`f_ret_1m`, `f_ret_5m`
- 时间特征：`minute_of_session`

## 3.2 模型顺序
1. Naive：`y_hat=0`
2. Ridge（稳定基线）
3. LightGBM（如果真实样本充足）

---

## 4. 验证与对比

统一使用 walk-forward（expanding window）：
- 指标：MAE、RMSE、Sign Hit Ratio
- 维度：
  - 路线 A vs 路线 B
  - horizon（1/5/15m）
  - 不同交易时段（开盘前30m、常规时段、尾盘）

判定标准（v1）：
- 若某路线在 ≥2 个 horizon 上稳定优于 naive，则进入下一轮（交易映射）
- 否则回到特征工程迭代

---

## 5. 具体执行步骤（可直接开工）

### Step 1（今晚）数据落地
- 新建脚本：`iv_vega_hf/src/load_volresearch_data.py`
- 读取 `volresearch/data/derived/*.parquet`
- 产出：
  - `iv_vega_hf/data/real/mainpool_1m.parquet`（路线A）
  - `iv_vega_hf/data/real/top3_contract_1m.parquet`（路线B）

### Step 2（随后）标签与特征
- 复用并扩展现有：`build_features.py` / `build_labels.py`
- 产出：
  - `features_mainpool_1m.parquet`, `labels_mainpool_1m.parquet`
  - `features_top3_1m.parquet`, `labels_top3_1m.parquet`

### Step 3（随后）评估
- 复用 `eval_walkforward.py`（先 ridge + naive）
- 报告：`iv_vega_hf/reports/model_eval_real_v1.md`
  - A/B 对比表
  - horizon 分项结果
  - 初步结论与下一步建议

---

## 6. 风险与处理

- 风险1：不同文件的交易时段/字段口径有差异
  - 处理：先做字段映射 + 时段过滤配置
- 风险2：Top3 会日内切换，产生样本不连续
  - 处理：先按“日级 Top3 固定”版本，后续再做滚动 Top3
- 风险3：单日/少日样本导致结论不稳定
  - 处理：先跑通，再扩多日做稳定性检验

---

## 7. 本计划对应你的要求

- ✅ 已从 `volresearch` 文件夹入手
- ✅ 已按你说的两条线设计：
  - 主图口径（IV K线 + 成交 vega）
  - 当天成交量最大的三个合约口径
- ✅ 目标明确：预测后续 IV 变化

如果你确认这个计划，我下一步就直接开始 Step 1，先把两套真实数据样本产出来给你验收。