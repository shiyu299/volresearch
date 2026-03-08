# volresearch

这个目录用于波动率研究（IV）。

## 快速启动应用

桌面已放置启动图标：`打开IV研究.command`

双击后会启动 Streamlit：
- 默认地址：`http://localhost:8501`
- 主程序：`timeseries/iv_inspector_refactor_toggle/iv_inspector_refactor/app.py`

## 性能优化（已做）

已优化 `iv_inspector/data.py`：
- 文件列表优先显示 parquet / feather / csv.gz（减少误选大csv）
- CSV读取优先使用 `pyarrow` 引擎（可回退）
- 仅加载应用核心列（减少IO和内存）
- 支持 `timestamp -> dt_exch` 自动回退
- 低基数列转 `category` 降内存

## 建议的使用习惯

1. 原始CSV放在 `timeseries/`。
2. 优先使用 parquet/csv.gz 进行研究展示。
3. 大CSV首次处理后，尽量转为 parquet 再开 app。
