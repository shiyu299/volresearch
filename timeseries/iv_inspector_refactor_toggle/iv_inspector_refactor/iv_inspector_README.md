
# iv_inspector / 波动率时序分析工具

## 项目目标
本项目用于从期权明细数据（parquet / csv）中，构造一个代表“近 ATM 波动率水平”的时间序列指数，
并以 K 线 + traded_vega 成交柱 的形式进行可视化。

适用于：
- 商品期权隐含波动率盘中演化观察
- 不同 ATM 范围 / 过滤规则下的 vol 行为对比
- 为 vol signal、gamma / vega 策略、期权组合回测提供输入

## 项目结构
app.py                Streamlit UI 入口
iv_inspector/         核心逻辑模块
data/raw/             原始输入数据
data/derived/         modified v4 输出，app 主数据源
data/factors/         factor 预计算中间层
out/                  输出结果
README.md             项目说明

## 核心流程
1. 读取数据
2. ATM 附近 n 个合约选择
3. Vega 加权 IV 计算
4. Resample 构造 OHLC
5. 可视化与导出

## 使用方式
cd iv_inspector_refactor
streamlit run app.py

将可视化使用的 parquet / csv 文件放入 `data/derived/`，原始输入放入 `data/raw/`。
如需单合约 factor 加速，先运行 `precompute_factor_materials.py`，产物会写入 `data/factors/`。
