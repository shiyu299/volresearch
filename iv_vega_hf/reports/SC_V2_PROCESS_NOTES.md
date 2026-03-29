# SC V2 Process Notes (local)

This file records methodology/process decisions for reproducibility.

## Data pipeline
- Source root: volresearch/data/derived
- Current mainline subset: symbols containing sc2604
- Aggregation style: ATM-pool style aligned with appclaude.py concept

## Modeling decisions
- Primary model: walk-forward logistic regression
- Confidence trigger: |p - 0.5| >= threshold
- Thresholds commonly compared: 0.10, 0.15

## Factor policy
- Core factors are fixed unless explicitly revised
- F-IV relation policy:
  - resid_z used as trading factor
  - betaF / R2 percentile / stability are filter diagnostics

## Metric definitions
- all_hit: accuracy on all OOS predictions
- triggered_hit: accuracy on confidence-triggered subset
- coverage: triggered_count / total_predictions
- avg_vol_decimal: mean(sign(pred) * delta_iv)
- avg_vol_points: avg_vol_decimal * 100

## Output policy
- Intermediate checkpoints persisted under reports/
- Final package to include:
  1) factor ranking
  2) keep/drop rationale
  3) metrics table and recommended v2 combo
