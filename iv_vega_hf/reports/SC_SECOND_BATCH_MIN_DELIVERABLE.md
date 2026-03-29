# SC 第二批因子收口（最小可用交付）

更新时间：2026-03-22 10:30 (Asia/Shanghai)
范围：SC（sc2604），分钟级主线，逻辑回归口径沿用既有结论。

> 说明：这是“可先用于回测配置”的最小可用版本。完整终版（逐因子定量边际全表）仍需补一轮统一重算。

---

## 1) 第二批因子排名（当前可确认）

基于现有已完成评估与状态文件，当前可确认的第二批优先级：

1. **flow_ema10**（优先级 S）
   - 结论：在 `H=5m, conf>=0.15` 下，对触发后命中与 avg_vol 均有正贡献。
2. **shockF**（优先级 A）
   - 结论：在 `H=5m, conf>=0.15` 下，对触发后命中有提升；avg_vol 存在轻微权衡。
3. 其余第二批候选（优先级 B/C）
   - 当前无稳定超越证据，先不进入 v2 生产组合。

---

## 2) keep/drop 清单（SC 主线）

### v2_keep（用于下一轮回测）
- flow
- iv_dev_ema5_ratio
- iv_mom3
- iv_willr10
- resid_z
- flow_ema10
- shockF

### v2_drop（本轮先不纳入）
- 第二批其余候选因子（待终版边际表补齐后再复核）

### 去留理由（最小版）
- `flow_ema10`：提升稳定、兼顾 hit 与收益口径。
- `shockF`：对方向预测增益明确，虽有收益口径权衡但仍值得保留。
- 其余候选：当前证据不足或稳定性不够，先降级观察。

---

## 3) 指标表（当前已锁定可用口径）

### Baseline（core + resid_z）
来源：`reports/SC_V2_RESULTS_STATUS.md`

- Horizon: 5m
- conf threshold: 0.15
- all_hit: **0.6115**
- triggered_hit: **0.7278**
- coverage: **0.1744**
- avg_vol_decimal: **0.02953**
- avg_vol_points: **2.953**

### Second-batch 增量结论（最小版）
- `+flow_ema10`：triggered_hit ↑，avg_vol_points ↑（方向同向）
- `+shockF`：triggered_hit ↑，avg_vol_points 存在轻微 tradeoff

> 注：终版会补“每个候选因子的定量边际（Δall_hit/Δtriggered_hit/Δcoverage/Δavg_vol_points）”。

---

## 4) 可复现运行（最小可跑命令）

### 4.1 数据构建（真实数据 -> 1min 主线）
```bash
python3 iv_vega_hf/src/load_volresearch_data.py \
  --input-dir /Users/shiyu/.openclaw/workspace/volresearch/data/derived \
  --output-dir /Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real \
  --topn 3 --atm-n 20
```

### 4.2 主线评估（1/5/15m）
```bash
python3 iv_vega_hf/src/eval_real_iv.py \
  --mainpool /Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/mainpool_1m.parquet \
  --top3 /Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/top3_contract_1m.parquet \
  --horizons 1,5,15
```

### 4.3 因子组消融（最小版）
```bash
python3 iv_vega_hf/src/eval_factor_ablation.py \
  --input /Users/shiyu/.openclaw/workspace/iv_vega_hf/data/real/mainpool_1m.parquet \
  --horizon 5
```

---

## 5) 下一步（终版补齐项）

1. 输出第二批**逐因子定量边际表**（统一口径）。
2. 固化 SC 最终 keep/drop 白名单（含定量阈值）。
3. 将同样流程复制到 TA/MA/FG/SH（SA=SH）。
