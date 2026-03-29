# SC V2 Results Status (local persistence)

Last updated: 2026-03-21 22:05 Asia/Shanghai

## Scope
- Instrument scope: SC2604 chain (ATM-pool mainline)
- Modeling scope: minute-level logistic regression
- Core skeleton: flow, iv_dev_ema5_ratio, iv_mom3, iv_willr10, resid_z

## Confirmed intermediate results
- Baseline (core + resid_z, H=5m, conf threshold=0.15):
  - all_hit: 0.6115
  - triggered_hit: 0.7278
  - coverage: 0.1744
  - avg_vol_decimal: 0.02953
  - avg_vol_points: 2.953

## First-batch add-on screening (H=5m, conf threshold=0.15)
- Strong candidates:
  - flow_ema10 (improves triggered_hit and avg_vol)
  - shockF (improves triggered_hit, slight avg_vol tradeoff)

## Pending deliverables
1. Second-batch factor ranking table
2. Final keep/drop list with reasons
3. Final SC v2 full metrics table
   - all_hit
   - triggered_hit
   - coverage
   - avg_vol_decimal / avg_vol_points

## Notes
- Detailed hourly progress is logged in: iv_vega_hf/PROGRESS_LOG.md
- This file is a persistent local checkpoint for final packaging.
