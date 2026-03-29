# 项目计划（iv_vega_hf）

## 总目标
构建一套日内可平仓的高频预测框架：以期权侧 `vega + IV + 微观结构` 为核心，预测短周期波动率信号。

## 里程碑

### M1 研究与设计（D1-D2）
**交付：**
- `spec/spec_v1.md`
- `research/literature_queue.md`

**验收：**
- 目标标签（A/B/C 层）定义清晰
- 时间切分与防泄露规则可执行
- 特征字典初版形成

### M2 数据与特征管线（D3-D4）
**交付：**
- `src/build_dataset.py`（占位，后续实现）
- `src/build_features.py`（占位，后续实现）

**验收：**
- 可输出统一 1min 粒度样本
- 可复现实验输入 parquet

### M3 建模与验证（D5）
**交付：**
- `src/train_baseline.py`
- `src/train_lgbm.py`
- `reports/model_eval_v1.md`

**验收：**
- Walk-forward OOS 指标可复现
- 至少 1 个模型超过 baseline

### M4 交易映射与成本回测（D6）
**交付：**
- `src/backtest_intraday.py`
- `reports/backtest_v1.md`

**验收：**
- 成本后绩效与风险指标完整
- 日内平仓约束与容量限制纳入

### M5 复盘与迭代（D7）
**交付：**
- `reports/review_v1.md`

**验收：**
- 有效特征/失效场景/下一轮实验计划明确
