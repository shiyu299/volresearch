# Real Data Onboarding Checklist (v1)

## 1) 数据到位
- [ ] 原始分钟级数据路径确认（或外部挂载路径）
- [ ] 覆盖区间确认（起止日期、交易日数量）
- [ ] 文件格式确认（csv/parquet）

## 2) 字段映射
- [ ] 按 `spec/field_mapping_v1.csv` 对齐字段名
- [ ] 校验必需字段：`timestamp`, `iv_atm`
- [ ] 可选字段缺失策略确认（置零/前填/不计算该特征）

## 3) 质量检查
- [ ] 时间戳单调递增、无重复
- [ ] 交易时段外样本处理（剔除或单独标记）
- [ ] 异常值规则（winsorize / MAD）

## 4) 首版跑通
- [ ] `build_dataset.py` 产出真实切片 parquet
- [ ] `build_features.py` 产出特征 parquet
- [ ] `build_labels.py` 产出 `y_1m/y_5m/y_15m`
- [ ] `eval_walkforward.py` 跑通并生成结果表

## 5) 输出物
- [ ] 更新 `reports/model_eval_v1.md`（新增真实数据结果）
- [ ] 更新 `PROGRESS_LOG.md`
- [ ] 提交 commit
