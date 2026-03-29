# SC 报告（ATM n=6）：Baseline + 两个二批因子

更新时间：2026-03-22 10:45 (Asia/Shanghai)

## 1. 本次运行配置

- 标的：SC（`sc2604`）
- IV 口径：ATM 邻域池化 IV（`atm-n=6`）
- 频率：分钟级（1min）
- Horizon：`h=5m`
- 触发阈值：`conf >= 0.15`
- 模型：walk-forward logistic regression（逐时点仅用历史训练）

运行命令：
```bash
python3 /Users/shiyu/.openclaw/workspace/iv_vega_hf/src/run_sc_pipeline.py \
  --atm-n 6 \
  --horizon 5 \
  --conf-thr 0.15 \
  --report /Users/shiyu/.openclaw/workspace/iv_vega_hf/reports/SC_SECOND_BATCH_PIPELINE_RESULT_ATM6.json
```

结果文件：
- `iv_vega_hf/reports/SC_SECOND_BATCH_PIPELINE_RESULT_ATM6.json`

---

## 2. 因子组合定义

### 2.1 Baseline（第一批核心骨架）
- `flow`
- `iv_dev_ema5_ratio`
- `iv_mom3`
- `iv_willr10`
- `resid_z`

### 2.2 两个第二批因子
- `flow_ema10`
- `shockF`

### 2.3 三种对照
1) Baseline
2) Baseline + `flow_ema10`
3) Baseline + `shockF`

（补充：报告末尾也给了“两个都加”的结果）

---

## 3. 指标计算方式（详细）

### 3.1 标签与方向
- 连续标签：`y_5m = IV(t+5m) - IV(t)`
- 真实方向：
  - `y_5m > 0` 记为上涨
  - `y_5m <= 0` 记为下跌

### 3.2 模型输出与触发
- 模型输出上涨概率 `p`
- 预测方向：
  - `p >= 0.5` -> 预测上涨（+1）
  - `p < 0.5` -> 预测下跌（-1）
- 置信度：`conf = |p - 0.5|`
- 触发条件：`conf >= 0.15`

### 3.3 四个核心指标
记：
- 总 OOS 样本数为 `N`
- 触发样本数为 `N_sig`
- 第 i 个样本预测方向 `pred_i in {+1,-1}`
- 真实方向 `true_i in {+1,-1}`
- 连续标签 `y_i = y_5m(i)`

1) `all_hit`：
\[
all\_hit = \frac{1}{N} \sum_{i=1}^{N} 1(pred_i = true_i)
\]

2) `triggered_hit`：
\[
triggered\_hit = \frac{1}{N_{sig}} \sum_{i\in Triggered} 1(pred_i = true_i)
\]

3) `coverage`：
\[
coverage = \frac{N_{sig}}{N}
\]

4) `avg_vol_points`：
先算
\[
avg\_vol\_decimal = \frac{1}{N_{sig}}\sum_{i\in Triggered}(pred_i \cdot y_i)
\]
再换算
\[
avg\_vol\_points = avg\_vol\_decimal \times 100
\]

---

## 4. 结果（ATM n=6）

| 组合 | all_hit | triggered_hit | coverage | avg_vol_points |
|---|---:|---:|---:|---:|
| Baseline | 0.5722 | 0.6452 | 0.2487 | 0.9684 |
| Baseline + flow_ema10 | 0.6043 | 0.6667 | 0.2487 | 0.7781 |
| Baseline + shockF | 0.5909 | 0.6509 | 0.2834 | 0.7873 |

补充（两个都加）：
- all_hit = 0.6043
- triggered_hit = 0.6698
- coverage = 0.2834
- avg_vol_points = 0.8091

---

## 5. 解读（仅基于本次 ATM n=6 结果）

- `flow_ema10`：提升了命中率（all_hit / triggered_hit），但单笔 vol points 下降。
- `shockF`：提高了 coverage，并小幅提升命中；单笔 vol points 也较 baseline 低。
- 两者一起：命中率和覆盖率更好，收益口径仍低于 baseline。

结论：
- 如果优先“信号稳定触发 + 命中”，可保留二批因子；
- 如果优先“单笔 vol points”，baseline 反而更强。

建议下一步：按你的交易偏好加权（命中优先 or 单笔收益优先）来定最终 keep/drop。