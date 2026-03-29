# Simple IV Pipeline

`simple/` 是给同事用的精简版流程，覆盖：

- `simple/data/raw/*.csv` 或 `simple/data/raw/<day_dir>/*.csv`
- 通过 `modified v4` 生成 `simple/data/derived/*.parquet`
- 用精简版 Streamlit app 直接查看 ATM 聚合 IV 图、单合约图、debug 和 drilldown

不包含：

- factor 预计算
- factor UI
- factor 评估逻辑

## 目录

- `app.py`: 精简版可视化 app
- `process_raw_to_derived.py`: raw CSV -> derived parquet
- `run_simple_app.bat`: 直接启动 app
- `run_modified_then_app.bat`: 先跑一份 raw，再直接启动 app
- `setup_st_env.bat`: 创建本地 `simple/st_env`
- `st_env.yml`: 运行依赖
- `data/raw`: 默认原始数据目录
- `data/derived`: 默认预处理结果目录
- `st_env`: 打包时一起带走的本地环境目录

## 用法

### 1. 安装环境

```bat
simple\setup_st_env.bat
```

### 2. 放 raw 数据

把 csv 放到：

```text
simple\data\raw\
```

或按交易日子目录放到：

```text
simple\data\raw\605320\
```

### 3. 先处理 raw 数据

单文件：

```bat
simple\st_env\python.exe simple\process_raw_to_derived.py --csv-path simple\data\raw\605320\TA260605.csv --expiry-date 2026-04-13 --spread-limit 15
```

批量：

```bat
simple\st_env\python.exe simple\process_raw_to_derived.py --day-dirs 605320 --products TA,MA,FG,SH --expiry-date 2026-04-13 --spread-limit 15 --skip-existing
```

### 4. 打开 app

```bat
simple\run_simple_app.bat
```

或者一键：

```bat
simple\run_modified_then_app.bat
```

一键脚本里 raw 输入默认只需要填文件名，例如 `TA605320.csv`。
脚本会自动去 `simple/data/raw` 下面递归查找；如果有重名文件，再改成输入相对路径更稳。

## 说明

- 默认数据目录都在 `simple/data/`
- app 默认读取 `simple/data/derived`
- 环境默认放在 `simple/st_env`
- 仍然沿用 `iv_inspector` 的 ATM 聚合和 drilldown 逻辑
- `simple` 只去掉 factor 相关内容，不改原有聚合口径
## Standalone note

- `simple` now vendors the app-side helper modules it needs.
- You can copy the whole `simple` folder to another location and run it without depending on other folders under `volresearch`.
- A copied `st_env` may still fail on another machine. If that happens, run `setup_st_env.bat` on the target machine or use a local Python environment with the packages from `st_env.yml`.
