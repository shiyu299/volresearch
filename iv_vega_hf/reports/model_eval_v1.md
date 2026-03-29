# Model Eval v1 (Prototype, sample data)

## Scope
- Data: `data/sample_raw.csv` (10 rows synthetic/minimal sample)
- Pipeline: dataset -> features -> labels
- Split: chronological 80/20 proxy split

## Baseline Results

### Target: y_1m
- rows_train=7, rows_test=2
- naive_zero: MAE=0.001500, RMSE=0.001581
- linear_lstsq: MAE=0.001202, RMSE=0.001591

### Target: y_5m
- rows_train=4, rows_test=1
- naive_zero: MAE=0.004000, RMSE=0.004000
- linear_lstsq: MAE=0.004208, RMSE=0.004208

### Target: y_15m
- Not evaluable on current sample (insufficient non-null rows).

## Walk-forward (long synthetic sample, 240 bars)
- Dataset: `data/*_long.parquet`
- Evaluator: `src/eval_walkforward.py` (per-fold standardization + z-clip + ridge closed-form)

### y_1m
- folds=234
- naive_zero: MAE=0.000462, RMSE=0.000585
- ridge_cf: MAE=0.000490, RMSE=0.000630

### y_5m
- folds=230
- naive_zero: MAE=0.000921, RMSE=0.001205
- ridge_cf: MAE=0.000996, RMSE=0.001289

### y_15m
- folds=220
- naive_zero: MAE=0.001726, RMSE=0.002179
- ridge_cf: MAE=0.001677, RMSE=0.002208

## Ridge alpha quick grid (long synthetic sample)
Grid: `alpha ∈ {0.01, 0.1, 1, 5, 10}`

- `y_1m` best RMSE in grid: alpha=10, ridge RMSE=0.000608 (naive RMSE=0.000585)
- `y_5m` best RMSE in grid: alpha=10, ridge RMSE=0.001247 (naive RMSE=0.001205)
- `y_15m` best RMSE in grid: alpha=10, ridge RMSE=0.002181 (naive RMSE=0.002179)

结论（仅合成样本）：
- ridge 随 alpha 增大更稳定，但多数 horizon 仍未稳定优于 naive。
- 说明当前特征在该合成数据上增益有限，下一步应在真实数据上重新评估。

## Notes
- Current metrics are only for pipeline sanity-check, not strategy validation.
- Need real historical intraday data for meaningful OOS evaluation.
- Next step: run same protocol on production-like data slice and move to walk-forward folds.
