# Feature Dictionary v1 (Template)

> 目的：统一特征定义，保证可复现、可审计、防泄露。

## 字段说明
- `name`: 特征名（遵循 `<domain>_<metric>_<window><unit>[_<variant>]`）
- `formula`: 计算公式（明确分子分母/聚合方式）
- `inputs`: 原始输入字段列表（point-in-time）
- `latency`: 可用延迟（如 `t_close`, `t+1min`）
- `null_policy`: 缺失处理（ffill/zero/drop/clip）
- `winsorize`: 截尾规则（如 p1/p99）
- `notes`: 口径备注（合约筛选、交易时段等）

## 模板表

| name | formula | inputs | latency | null_policy | winsorize | notes |
|---|---|---|---|---|---|---|
| vega_signed_1m | sum(sign_i * traded_vega_i) over 1m | trade_sign, traded_vega | t_close | zero_if_no_trade | p1/p99 | sign from aggressor side |
| vega_imbalance_5m | (buy_vega - sell_vega) / max(total_vega, eps) over 5m | buy_vega, sell_vega | t_close | zero_if_total_zero | p1/p99 | eps=1e-8 |
| vega_signed_15m_ewm | EWM(alpha=2/(15+1)) of vega_signed_1m | vega_signed_1m | t_close | ffill_then_zero | none | alpha fixed |
| iv_delta_1m | iv_atm_t - iv_atm_t-1 | iv_atm | t_close | drop_if_missing | p1/p99 | atm by near-month |
| iv_zscore_30m | (iv_atm_t - mean_30m) / std_30m | iv_atm | t_close | drop_if_std_zero | clip(-8,8) | rolling window=30 |
| micro_trade_quote_ratio_1m | trade_count / max(quote_update_count,1) | trade_count, quote_update_count | t_close | zero_if_both_zero | p1/p99 | microstructure intensity |
| underlying_ret_1m | ln(px_t/px_t-1) | underlying_mid | t_close | drop_if_missing | p1/p99 | control variable |
| time_minute_of_session | minute index from session open | timestamp, calendar | t_close | not_null | none | [0, session_len) |

## 校验清单
- [ ] 所有 `inputs` 均为 point-in-time 可见字段
- [ ] `latency` 明确且不晚于标签起点
- [ ] 缺失处理策略可复现且已记录
- [ ] 与标签对齐：`X_t` 对 `y_h(t)=iv(t+h)-iv(t)`
- [ ] 训练/验证/回测口径一致
