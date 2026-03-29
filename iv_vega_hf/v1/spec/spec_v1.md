# Spec v1 — Intraday IV/Vega HF Framework

## 1. 任务层级

### Level A（先落地）
预测 `y_h = IV_{t+h} - IV_t`，h in {1, 5, 15} minutes。

### Level B（第二阶段）
预测 `future_RV - current_IV` 的短时修复幅度。

### Level C（第三阶段）
直接预测可交易信号（方向+强度），用于日内开平仓。

## 2. 样本与切分
- 粒度：1min（主），10s/30s 作为增强候选。
- 切分：walk-forward（训练窗口滚动，测试窗口前推）。
- 严禁随机 KFold。

## 3. 防泄露规范
- 仅使用 point-in-time 可见字段。
- 对齐规则：特征在 t 结束时可计算，标签从 t+1 开始。
- 任何未来报价/成交不进入当前样本。

## 4. 特征字典（v1）

### 4.1 Vega Flow 核心
- `signed_vega_1m`: 当前分钟方向化成交vega总和
- `signed_vega_5m`: 过去5分钟累积
- `vega_imbalance_1m`: (buy_vega - sell_vega) / total_vega
- `vega_ewm_15m`: 15分钟指数衰减累积

### 4.2 IV 动态
- `d_iv_1m`, `d_iv_5m`
- `iv_zscore_30m`
- `term_spread_near_next`

### 4.3 微观结构
- `opt_spread_bps`
- `depth_imbalance`
- `trade_quote_ratio_1m`

### 4.4 控制变量
- `underlying_ret_1m`
- `underlying_rv_5m`
- `minute_of_session`

## 5. 验证指标
- 回归：MAE/RMSE
- 排序：RankIC/ICIR
- 方向：Sign hit ratio
- 稳定性：分时段、分波动率 regime 评估

## 6. 标签细则（Level A 先行）

### 6.1 预测目标与 horizon
- 主任务：`y_h = iv_atm_t_plus_h - iv_atm_t`
- `horizon_set = {1, 5, 15}`（单位：分钟）
- `iv_atm` 默认使用主交易合约 ATM IV（如有多来源，优先 mid-IV 口径）

### 6.2 时间对齐与 lag（防泄露）
- 样本时间戳 `t` 表示 **[t-1min, t]** 收盘时刻。
- 特征只允许使用 `<= t` 可见信息。
- 标签从 `t+1` 开始，禁止使用 `t` 之后 quote/trade 构造特征。
- 建议统一实现：
  - `X_t = features(raw[:t])`
  - `y_h(t) = iv(t+h) - iv(t)`

### 6.3 可交易标签映射（用于 Level C / 回测前置）
给定模型输出 `pred_h`（对 `y_h` 的预测），定义三分类动作：
- `LONG_VOL`：`pred_h >= theta_long_h`
- `SHORT_VOL`：`pred_h <= -theta_short_h`
- `FLAT`：其余区间

阈值初始化（后续按训练集分位数自适应）：
- `theta_long_h = q_{0.70}(|pred_h|)`
- `theta_short_h = q_{0.70}(|pred_h|)`
- 若需不对称，可单独设定多空阈值。

### 6.4 样本过滤规则
- 剔除：停牌分钟、无有效双边报价分钟、异常跳点（按当日 MAD/分位规则）。
- 开收盘窗口单独打标：`is_open_window`, `is_close_window`。
- 低流动性分钟可仅用于训练、回测时禁交易（`tradable_flag=0`）。

### 6.5 评估标签补充
- 回归：`MAE_h`, `RMSE_h`
- 方向：`sign_hit_h = mean(sign(pred_h)==sign(y_h))`
- 可交易命中：`trade_hit_h`（仅在非 FLAT 样本上）

## 7. 特征命名规范（v1）
统一格式：`<domain>_<metric>_<window><unit>[_<variant>]`

- `domain`：`vega|iv|micro|underlying|time`
- `metric`：含义简洁、可复用（如 `imbalance`, `zscore`, `ret`, `spread`）
- `window`：整数窗口长度
- `unit`：`s|m|h|d`（秒/分/时/日）
- `variant`：可选（如 `ewm`, `raw`, `clip`, `near_next`）

示例：
- `vega_signed_1m`
- `vega_imbalance_5m`
- `vega_signed_15m_ewm`
- `iv_delta_1m`
- `iv_zscore_30m`
- `micro_trade_quote_ratio_1m`
- `underlying_ret_1m`
- `time_minute_of_session`

命名约束：
- 全小写、下划线分隔；禁止中英文混用与隐式缩写。
- 同一指标跨窗口只改 window/unit，不改 metric 语义。
- 特征字典中必须登记：`name`, `formula`, `inputs`, `latency`, `null_policy`。

## 8. 回测映射（预留）
- 阈值开仓：`score > θ_long`, `score < -θ_short`
- 强制日内平仓
- 成本：点差+手续费+滑点
