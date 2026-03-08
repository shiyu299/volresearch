# TAoil（TA605）论文指标 ↔ 代码对应索引

> 目标：把论文里用到的核心指标，映射到实际 Python 实现位置，便于你复核与引用。

## 一、主脚本位置

- 主脚本：`volresearch/timeseries/analysis/run_iv_vega_futures_analysis.py`
- TA 本次调用入口（参数化执行）：
  - 函数：`run_iv_pipeline(...)`（第 112 行）
  - 本次参数：`underlying=TA605, expiry_date=2026-04-13, spread_limit=25.0, out_prefix=TAoil_TA605`

---

## 二、指标与代码映射（论文常用）

### 1) 事件定义：`abs_traded_vega` 与 Q99 阈值

- 绝对成交vega：
  - `x['abs_traded_vega'] = x['traded_vega'].abs()`（第 332 行）
- 事件阈值：
  - `thr = x['abs_traded_vega'].quantile(0.99)`（第 333 行）
- 事件标记：
  - `x['is_event'] = x['abs_traded_vega'] >= thr`（第 334 行）
- 函数位置：`pl_event_study(panel, name)`（第 330 行）

### 2) 先兆指标：`no_trade_move_60`

- 定义：`|fut_ret_60s| / (fut_dvol_60s + 1)`
- 代码：`panel['no_trade_move_60'] = ...`（第 325 行）
- 构建函数：`build_1s_panel(df)`（第 276 行）

### 3) 先兆指标：`opt_spread_60s`

- 来源：期权秒级中位价差 `opt_spread_med` 的 60 秒 rolling median
- 计算在 `build_1s_panel` 的窗口循环里（约 318-324 行段）
- 在事件统计中作为特征之一（第 347 行）

### 4) 先兆指标：`iv_f_div_60`

- 定义：`iv_chg_60s * sign(fut_ret_60s)`
- 代码：`panel['iv_f_div_60'] = ...`（第 326 行）
- 在事件统计中作为特征之一（第 347 行）

### 5) 盘口不平衡：`fut_imb`

- 先构造：`imbalance=(bidvol1-askvol1)/(bidvol1+askvol1+1e-9)`
- 1 秒聚合列重命名：`imbalance -> fut_imb`（第 287 行）
- 在事件统计中作为特征之一（第 347 行）

### 6) Lift 统计（Q80）

- 核心逻辑：
  - `q80 = base[f].quantile(0.8)`（第 353 行）
  - `P(event|sig)`、`P(event)`、`lift` 的计算（第 354-358 行）
- 输出文件：`{name}_signal_lift_stats.csv`（第 412 行）

### 7) 分位事件率

- 对 `no_trade_move_60 / opt_spread_60s / iv_f_div_60` 做 `qcut` 分组（第 363 行开始）
- 输出文件：`{name}_quantile_event_rate.csv`（第 413 行）

### 8) 分时段统计（早/午/夜）

- 会话定义：`x['session'] = np.where(...)`（第 378 行）
- 聚合输出：`event_rate / avg_abs_vega / n`（第 379 行）
- 输出文件：`{name}_session_compare.csv`（第 414 行）

### 9) 事件明细表

- 输出字段：`abs_traded_vega, fut_ret_60s, iv_chg_60s, no_trade_move_60, opt_spread_60s, iv_f_div_60`
- 输出代码：第 415 行
- 输出文件：`{name}_events_detail.csv`

### 10) 图形输出（阈值图与事件路径图）

- 阈值图：`{name}_abs_traded_vega_threshold.png`（第 397 行）
- 事件窗口路径图：`{name}_event_window_path.png`（第 409 行）

---

## 三、本次 TAoil 对应产物（与你论文直接相关）

目录：`volresearch/timeseries/analysis/tables_taoil/`

- `TAoil_TA605_signal_lift_stats.csv` ← 对应“先兆Lift”章节
- `TAoil_TA605_quantile_event_rate.csv` ← 对应“分位事件率”章节
- `TAoil_TA605_session_compare.csv` ← 对应“分时段比较”章节
- `TAoil_TA605_events_detail.csv` ← 对应“事件样本明细”附录
- `TAoil_TA605_sec_panel_1s.csv` ← 对应“指标构建底表”附录

图目录：`volresearch/timeseries/analysis/plots_taoil/`

- `TAoil_TA605_abs_traded_vega_threshold.png`
- `TAoil_TA605_event_window_path.png`

---

## 四、引用建议（写论文时）

- 方法章节建议写法：
  - “事件以 `abs_traded_vega` 的样本内 Q99 定义；先兆特征包括 `no_trade_move_60`、`opt_spread_60s`、`iv_f_div_60` 与 `fut_imb`；通过 Q80 条件概率提升（Lift）评估预测能力。”
- 结果章节建议与以下文件一一对应引用：
  - Lift → `TAoil_TA605_signal_lift_stats.csv`
  - 分位对比 → `TAoil_TA605_quantile_event_rate.csv`
  - 时段差异 → `TAoil_TA605_session_compare.csv`
  - 事件回溯 → `TAoil_TA605_events_detail.csv`
