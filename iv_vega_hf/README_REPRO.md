# SC 可复现流程（本地）

下面这套是你要的“因子代码 + 全流程代码”落地版。

## 代码位置

- 因子构建：`iv_vega_hf/src/sc_factor_lib.py`
- 逻辑回归 walk-forward 评估：`iv_vega_hf/src/sc_train_eval.py`
- 一键流水线（输入→因子→模型→第二批收口指标）：`iv_vega_hf/src/run_sc_pipeline.py`

## 一键运行

```bash
python3 /Users/shiyu/.openclaw/workspace/iv_vega_hf/src/run_sc_pipeline.py
# 默认 atm-n=6
```

## 关键输入

- 原始数据：`/Users/shiyu/.openclaw/workspace/volresearch/data/derived/*.parquet`
- SC 原始文件（默认）：`/Users/shiyu/.openclaw/workspace/volresearch/data/derived/sc2604_iv_20260313.parquet`

## 关键输出

- 中间因子与标签：
  - `iv_vega_hf/data/sc_pipeline/sc_factors_targets.parquet`
- 第二批收口结果（JSON）：
  - `iv_vega_hf/reports/SC_SECOND_BATCH_PIPELINE_RESULT.json`

## 模块职责

1. `load_volresearch_data.py`
   - 从 `volresearch/data/derived` 读 parquet
   - 生成 1min 主线数据 `mainpool_1m.parquet`

2. `sc_factor_lib.py`
   - 生成核心因子：`flow, iv_dev_ema5_ratio, iv_mom3, iv_willr10, resid_z`
   - 生成第二批：`flow_ema10, shockF`
   - 生成标签：`y_1m/y_3m/y_5m`

3. `sc_train_eval.py`
   - walk-forward 逻辑回归（手写 GD，无 sklearn 依赖）
   - 输出：all_hit / triggered_hit / coverage / avg_vol_points

4. `run_sc_pipeline.py`
   - 串起完整流程
   - 输出 baseline、单因子 add-one、最终 keep 组合结果

## 可调参数

```bash
python3 iv_vega_hf/src/run_sc_pipeline.py \
  --horizon 5 \
  --conf-thr 0.15 \
  --report /Users/shiyu/.openclaw/workspace/iv_vega_hf/reports/SC_SECOND_BATCH_PIPELINE_RESULT.json
```
