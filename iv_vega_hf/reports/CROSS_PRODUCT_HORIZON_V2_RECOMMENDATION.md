# 分品种推荐 Horizon 与 v2 因子保留清单（基于 SC 方法复跑）

更新时间：2026-03-22（Asia/Shanghai）
输入依据：
- `iv_vega_hf/reports/CROSS_PRODUCT_SUMMARY.md`
- `volresearch/data/derived/{TA,MA,FG,SH}/RESULTS_SC_METHOD.md`

> 说明：当前 4 个品种复跑使用同一套因子集合（`flow, iv_dev_ema5_ratio, iv_mom3, iv_willr10, resid_z, flow_ema10, shockF`）。
> 因此本清单先给出“可直接用于下一轮回测/上线候选”的保留建议；最终精简（逐品种 drop 列表）需在逐因子消融后固化。

---

## 1) 分品种推荐 Horizon（主推 + 备选）

### TA
- **主推：1m**
  - all_hit=0.6275, triggered_hit=0.7293, coverage=0.247
  - 在 1m 命中与覆盖的平衡最好；3m/5m 虽 avg_vol_points 更高，但 all_hit 明显回落。
- 备选：5m（若更看重单笔 vol 点数）

### MA
- **主推：1m**
  - all_hit=0.6325, triggered_hit=0.7451, coverage=0.247
  - 3m/5m 覆盖率明显下降（0.142/0.124），整体稳健性不如 1m。
- 备选：3m（作为中短周期补充）

### FG
- **主推：5m**
  - all_hit=0.6518, triggered_hit=0.7650, coverage=0.350
  - 三个 horizon 里 5m 的命中率与触发后命中都最优，且 coverage 高。
- 备选：3m（all_hit=0.6448, coverage=0.376，覆盖更高）

### SH（SA 对应 SH）
- **主推：1m**
  - all_hit=0.6696, triggered_hit=0.7519, coverage=0.209
  - 3m/5m 命中和覆盖同步走弱，尤其 5m coverage 仅 0.094。
- 备选：3m（若策略希望降低换手）

---

## 2) 分品种 v2 因子保留清单（当前可执行版）

统一候选因子：
- `flow`
- `iv_dev_ema5_ratio`
- `iv_mom3`
- `iv_willr10`
- `resid_z`
- `flow_ema10`
- `shockF`

### TA（主推 1m）
- **v2_keep（生产候选）**：
  - `flow`
  - `iv_dev_ema5_ratio`
  - `iv_mom3`
  - `resid_z`
  - `flow_ema10`
- **v2_reserve（观察）**：
  - `iv_willr10`
  - `shockF`

### MA（主推 1m）
- **v2_keep（生产候选）**：
  - `flow`
  - `iv_dev_ema5_ratio`
  - `iv_mom3`
  - `resid_z`
  - `flow_ema10`
- **v2_reserve（观察）**：
  - `iv_willr10`
  - `shockF`

### FG（主推 5m）
- **v2_keep（生产候选）**：
  - `flow`
  - `iv_dev_ema5_ratio`
  - `iv_mom3`
  - `resid_z`
  - `flow_ema10`
  - `shockF`
- **v2_reserve（观察）**：
  - `iv_willr10`

### SH（主推 1m）
- **v2_keep（生产候选）**：
  - `flow`
  - `iv_dev_ema5_ratio`
  - `iv_mom3`
  - `resid_z`
  - `flow_ema10`
- **v2_reserve（观察）**：
  - `iv_willr10`
  - `shockF`

---

## 3) 统一上线建议（先做一版可比回测）

- TA/MA/SH：先上 1m 主策略，3m 仅做旁路监控
- FG：先上 5m 主策略，3m 做增强/对照
- 统一触发阈值先用 `conf >= 0.15`，后续分品种细调

建议下一轮回测输出固定四项：
1. all_hit
2. triggered_hit
3. coverage
4. avg_vol_points

并额外补：
- turnover / 交易成本敏感性
- 分时段稳定性（开盘/午盘/尾盘）
- 滚动窗口退化监控（近 3/5/10 日）

---

## 4) 待完成项（用于最终“精简版 v2”）

为避免“全因子保留导致冗余”，下一步建议做逐品种逐因子消融（ablation）：
- 每个品种在其主推 horizon 下，执行 leave-one-out + greedy add-back
- 产出最终 `keep/drop` 白名单
- 再固化到 `spec` 与回测配置

在当前信息集下，本报告可直接作为**下一轮回测/上线候选配置**。