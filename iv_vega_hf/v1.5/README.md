# v1.5

TA multi-file online analysis based on the v1.1 second-level 5-minute-blend feature design.

Key additions:
- process all `data/derived/TA/*.parquet` files in order
- first run warms up on labeled history, later trade dates continue directly
- exclude the first 3 minutes after session open from training samples
- start prediction only after 5 minutes from each session open
- save model parameter snapshots under `output/ta/models/`
