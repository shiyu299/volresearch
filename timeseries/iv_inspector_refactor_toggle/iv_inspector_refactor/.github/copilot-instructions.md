## 简要目标

为 AI 编码代理提供可立刻上手的仓库知识：架构要点、关键入口、常见约定、运行/调试命令以及修改敏感点。

## 关键文件与职责
- `app.py`: Streamlit UI 入口。负责参数 sidebar、缓存签名 `signature()`、按需触发计算 `compute_if_needed()` 并渲染图表/钻取。修改 UI/交互优先从此文件入手。
- `iv_inspector/data.py`: 数据加载与文件发现（`list_data_files()` 返回绝对路径；`load_data()` 负责类型转换与基础列校验）。修改数据格式处理请在此处实现向后兼容。
- `iv_inspector/aggregation.py`: 核心序列合成函数 `make_iv_and_bar_series()`（IV 指数、bar、debug/details 表）。包含 `iv_fill_mode`（`state_adjust`/`ffill`/`quote_only`）、候选池刷新逻辑、vega 加权等业务规则。
- `iv_inspector/selection.py`: “ATM±n” 选取规则（基于 `F_used`，ATM 为最接近的 K；非 ATM 可选 OTM 过滤）。重要：选取本身不填补缺失 IV，下游聚合会处理。
- `iv_inspector/viz.py`: Plotly/Matplotlib 绘图函数，推荐使用 `build_fut_iv_vega_stack_figure()` 来生成主视图。
- `iv_inspector/drilldown.py`: 区间钻取和导出（`render_drilldown_tabs()`、`tables_to_excel_bytes()`）。导出依赖 `openpyxl`。 

## 运行与调试
- 本地启动（在仓库根目录）：
```powershell
streamlit run app.py
```
- 数据放到 `data/` 目录（支持 `parquet`, `pq`, `csv`, `feather`）。UI 中会列出绝对路径。
- 注意 `load_data()` 会要求 `dt_exch` 存在并会把它解析为时间戳；缺失会抛错或返回空 DataFrame。

## 项目约定与易错点（必须遵守）
- 时间桶规则使用 pandas 的 resample 规则字符串（例如 `"250ms"`, `"1S"`, `"5min"`）。在聚合中 `base_rule` 决定原始序列桶，`kline_rule` 决定 K 线重采样。
- 候选池选取：基于 `F_used` 的当前值固定池（受 `pool_refresh_seconds` 控制），池不会为补 IV 而动态扩展。
- IV 填充模式：
  - `state_adjust`：使用上次状态按 -Delta/Vega * dF 修正；小幅期货变动（<= fut_move_threshold）会退化为 ffill。
  - `ffill`：直接向前填充最近 IV。
  - `quote_only`：仅使用当秒报价，不使用历史状态填充。
- `make_iv_and_bar_series()` 返回 `(ser, bar_ser, debug_df, details_df)`。当 `ser` 为空或全 NaN 时，上层会停止渲染。

## 依赖与集成点
- 主要依赖：`pandas`, `streamlit`, `plotly`, `numpy`, `openpyxl`（用于 Excel 导出）。`requirements.txt` 列出依赖。
- I/O：读取 parquet/feather 需要对应 engine（系统需有 pyarrow 或 fastparquet 支持）。

**关于 data 列表显示与路径解析**：`list_data_files()` 会在 UI 中以相对路径显示仓库内的文件（例如 `data/foo.parquet`），但在读取时 `app.py` 会把相对路径解析回绝对路径并传给 `load_data()`，因此既保持可移植性又保证可读性。

**固定的依赖版本（来自 requirements.txt）**：
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `plotly`
- `openpyxl`
- `pyarrow`
- `duckdb`

如果需要，我可以把上面改成 `package==version` 的精确格式（把 `requirements.txt` 中写好的版本号直接列出来）。

## 修改指南与示例
- 增加新填充策略：修改 `iv_inspector/aggregation.py::make_iv_and_bar_series`，同时在 `app.py` sidebar 增加对应 label 并映射到 `iv_fill_mode`。
- 改变 ATM 选取规则：编辑 `iv_inspector/selection.py::_infer_atm_strike` 或 `pick_atm_n_options`，并在 `drilldown` 中同步测试结果导出。
- 性能注意：`render_drilldown_tabs` 会构造 DataFrame 列表并导出到 Excel，若数据量大请使用 `is_ultra` 或分片策略来避免内存峰值（`make_iv_and_bar_series` 支持 `is_ultra` 参数）。

## 快速检查点（调试 checklist）
- 若图表为空：检查 `data/` 文件、`dt_exch` 列、以及 `make_iv_and_bar_series` 是否返回非空 `ser`。
- 缺少 signed vega：确认 `traded_vega_signed` 是否存在，否则系统会尝试基于 `trade_price` 与 `mid` 构造 `traded_vega_signed`。

## 我需要你确认或补充的点
- 是否需要把 `data/` 文件名显示为相对路径而不是绝对路径？
- 是否希望将 `requirements.txt` 中的版本固定进一步写进本文件？

请审阅以上内容，告诉我哪些部分不准确或需要更详细的示例，我会迭代更新。
