# iv_vega_hf

`iv_vega_hf` 是一个围绕期权 ATM 隐含波动率短周期方向预测的研究项目。

这套项目的主线可以概括成 4 步：

1. 从 `volresearch/data/derived/*.parquet` 读取高频原始数据
2. 聚合出分钟级 ATM IV 池化序列
3. 在分钟级序列上构造 IV / vega / 标的冲击因子
4. 用滚动逻辑回归预测未来若干分钟 IV 是涨还是跌

如果只看最核心、最完整的一条链，建议把 `v1` 理解为项目主线：

- 数据读取与分钟聚合：`src/load_volresearch_data.py`
- 分钟级因子：`src/sc_factor_lib.py`
- 滚动逻辑回归：`src/sc_train_eval.py`
- 一键串联：`src/run_sc_pipeline.py`

后面的 `v1.1` 和 `v1.5` 则是在这条主线之上，进一步往秒级触发或在线更新方向做的实验分支。

## 目录结构

- `src/`
  - 主线版本代码
- `data/`
  - 中间产物和样例数据
- `reports/`
  - 实验结果和研究总结
- `v1/`
  - 第一版整理过的分钟级主线副本
- `v1.1/`
  - 秒级触发实验版
- `v1.5/`
  - 在线更新、跨日续用参数的实验版

## 版本关系

### `src/` / `v1/`

这是分钟级主线。

目标是：

- 从高频原始期权 / 期货 parquet 里提取 ATM 附近期权池
- 每分钟形成一个池化 IV 指数
- 用这个分钟级 IV 序列构造因子
- 预测未来 `1m / 3m / 5m` 的 IV 变化方向

### `v1.1/`

这是秒级实验版。

目标是：

- 仍然沿用 “ATM 池化 IV + flow + shockF + residual” 的研究框架
- 但把采样频率改成秒级
- 每秒都输出一个上涨概率 `p`
- 预测目标改成未来 `3m~5m` 平均 IV 相对当前 IV 的变化方向

### `v1.5/`

这是在 `v1.1` 思路上的在线更新实验版。

它的特点是：

- 首日需要 warmup 样本
- 后续交易日继续沿用前一日参数
- 保存模型参数快照
- 适合更接近实时滚动部署的场景

## 数据输入

主线分钟版默认读取 `volresearch/data/derived/*.parquet`，依赖的核心字段包括：

- `dt_exch`
- `symbol`
- `is_option`
- `is_future`
- `F_used`
- `iv`
- `vega`
- `traded_vega_signed`
- `d_volume`
- `volume`
- `spread`
- `cp`
- `K`
- `bidprice1`
- `askprice1`

这些字段和 `volresearch` 主项目中的 derived parquet 口径一致。

## 主线流程总览

分钟级主线的完整逻辑如下：

1. 读取原始 parquet，拼接成统一面板
2. 过滤没有双边盘口的期货分钟
3. 在每个时间点从期权中选出 ATM 附近合约
4. 用 vega 加权方式构造 ATM 池化 IV 指数
5. 每分钟聚合出 `iv_pool / flow / F_used / spread_pool`
6. 在分钟级序列上构造 IV 自身、vega flow、标的冲击残差三类因子
7. 生成未来若干分钟的 IV 变化标签
8. 用 walk-forward 方式滚动训练逻辑回归
9. 输出概率 `p`，再根据 `|p - 0.5|` 形成触发信号

下面把每一层展开说明。

## 第一步：分钟级 ATM 池化序列怎么构造

代码位置：

- [load_volresearch_data.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/load_volresearch_data.py)

### 1. 先做期货单边市过滤

如果某一分钟内，期货没有出现任何有效双边盘口：

- `bidprice1 > 0`
- `askprice1 > 0`

那么这一分钟整段数据都会被过滤掉。

这样做的目的，是避免在标的价格本身不可交易、不可观测的时候，继续用期权 quote 去构造 IV 池化指标。

### 2. 选出 ATM 附近期权池

在保留下来的原始数据中：

- 只取 `is_option == True`
- 去掉 `iv / F_used / K` 缺失的记录
- 计算 `dist = |K - F_used|`
- 对每个时间戳按 `dist` 从小到大排序
- 取最近的 `atm_n` 个期权

这一步得到的是一个“当前时刻最接近 ATM 的期权池”。

### 3. 用 vega 加权构造 `iv_pool`

对每个时间点选出来的期权池，聚合成一个代表性 IV：

- 权重：`vega`
- 指标：`iv`

如果 `vega` 总和足够大，就用加权平均：

`iv_pool = average(iv, weights=vega)`

如果 `vega` 全部缺失或几乎为 0，就退化成简单平均。

### 4. 同时生成分钟级成交流与标的变量

分钟级主线上，会同时聚合出：

- `vega_signed_1m`
  - 该分钟内 `traded_vega_signed` 的和
- `vega_abs_1m`
  - 该分钟内绝对值成交 vega 的和
- `spread_pool_1m`
  - 该分钟期权池平均 spread
- `F_used`
  - 该分钟最后一个可用的标的期货价格
- `f_ret_1m`
  - `log(F_t / F_{t-1})`

最终输出主线数据集：

- `data/real/mainpool_1m.parquet`

另一路还会生成高成交量前 `topn` 个单合约的分钟面板：

- `data/real/top3_contract_1m.parquet`

## 第二步：分钟级因子怎么构造

代码位置：

- [sc_factor_lib.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/sc_factor_lib.py)

这个项目的因子并不算很多，但逻辑很集中，基本围绕三类信息：

1. IV 自身位置和动量
2. vega 成交流
3. 标的期货冲击与 IV 的残差关系

### A. IV 自身位置 / 动量类

#### `iv_dev_ema5_ratio`

定义：

`(iv_pool - ema(iv_pool, 5)) / |ema(iv_pool, 5)|`

解释：

- 当前 IV 偏离短期均值多少
- 正值大：当前 IV 相对短期均值偏高
- 负值大：当前 IV 相对短期均值偏低

#### `iv_mom3`

定义：

`iv_pool[t] - iv_pool[t-3]`

解释：

- 过去 3 分钟的 IV 动量
- 正值大：最近 3 分钟 IV 在上行
- 负值大：最近 3 分钟 IV 在下行

#### `iv_willr10`

定义：

在过去 10 个分钟 bar 内，把 `iv_pool` 看作一个价格序列，计算 Williams %R：

`-100 * (HH - IV) / (HH - LL)`

解释：

- 反映当前 IV 在近 10 分钟区间高低位中的相对位置

### B. vega flow 类

#### `flow`

定义：

`flow = vega_signed_1m`

解释：

- 这一分钟净买入还是净卖出 vega
- 正值越大，说明净买 vol 压力越强
- 负值越大，说明净卖 vol 压力越强

#### `flow_ema10`

定义：

`EMA(flow, 10)`

解释：

- 不只看当前一分钟的 flow
- 而是看过去一小段时间的持续买 / 卖 vol 状态

### C. 标的冲击 / 残差类

#### `shockF`

先定义期货一分钟对数收益：

`dF_t = log(F_t / F_{t-1})`

再构造衰减冲击：

`shockF_t = dF_t + 0.5 * dF_{t-1} + 0.25 * dF_{t-2}`

解释：

- 当前和前几分钟标的价格冲击的加权和
- 用来表示近几分钟期货方向和强度

#### `resid_z`

先计算 IV 变化：

`dIV_t = iv_pool[t] - iv_pool[t-1]`

再用滚动窗口估计：

`betaF = Cov(dIV, shockF) / Var(shockF)`

然后取残差：

`resid = dIV - betaF * shockF`

最后再做 rolling zscore：

`resid_z = zscore_roll(resid, 60)`

解释：

- 这表示“扣掉标的冲击可以解释的那部分后，IV 还剩下多少异常变化”
- 是项目里比较核心的一个研究因子

## 第三步：标签怎么定义

代码位置：

- [sc_factor_lib.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/sc_factor_lib.py)

标签是直接从分钟级 `iv_pool` 生成的：

- `y_1m = iv_pool[t+1] - iv_pool[t]`
- `y_3m = iv_pool[t+3] - iv_pool[t]`
- `y_5m = iv_pool[t+5] - iv_pool[t]`

这是一个连续变量标签，但在训练逻辑回归时会进一步二值化：

- 如果 `y_h > 0`，记作上涨类 `1`
- 否则记作下跌类 `0`

也就是说，这套逻辑回归本质上预测的是：

- “未来 h 分钟 IV 方向是否为正”

而不是直接预测未来 IV 变化的精确数值。

## 第四步：逻辑回归怎么做

代码位置：

- [sc_train_eval.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/sc_train_eval.py)

### 1. 模型形式

这是一个标准的二分类逻辑回归：

- 输入：分钟级因子向量 `X_t`
- 输出：未来 IV 上涨的概率 `p_t`

数学形式：

`p_t = sigmoid(w_0 + w^T X_t)`

其中：

- `sigmoid(z) = 1 / (1 + exp(-z))`

### 2. 损失函数

虽然代码里没有直接写出公式，但 `fit_logit_gd` 做的就是：

- 最小化二分类交叉熵
- 外加一个 `L2` 正则项

同时偏置项 `w_0` 不参与正则。

### 3. 求解方式

没有依赖 `sklearn`，而是自己写了一个简单的梯度下降：

- 初始参数 `w = 0`
- 每一步：
  - 计算 `p = sigmoid(X @ w)`
  - 计算梯度
  - 做一次参数更新

分钟主线默认：

- `lr = 0.05`
- `steps = 200`
- `l2 = 1e-3`

### 4. 为什么每次都重新标准化

在每一个滚动训练时点，都会只用历史训练样本去计算：

- 均值 `mu`
- 标准差 `sd`

然后做标准化：

`X_std = (X - mu) / sd`

这样做有两个原因：

1. 防止不同因子量纲差别太大
2. 保证不会使用未来样本的信息做标准化

### 5. walk-forward 是怎么滚动的

这套训练不是：

- 先切一段训练集 fit 一次
- 再固定参数去预测后面整段测试集

它做的是 expanding window 的 walk-forward：

1. 到时点 `t` 为止，只用 `[:t]` 的历史样本训练
2. 只预测 `t` 这个点
3. 再走到 `t+1`
4. 用更长的历史样本重新训练
5. 再预测下一个点

所以更准确地说：

- 每个 OOS 时点都会重新拟合一个新的逻辑回归
- 模型参数会随着时间持续更新

这是一种研究中很常见的“严格时间顺序、无未来信息”的评估方式。

### 6. 触发概率 `p` 和 `conf` 是什么

逻辑回归输出：

- `p = P(y > 0 | X)`

然后进一步定义：

- `pred_sign = +1`，如果 `p >= 0.5`
- `pred_sign = -1`，如果 `p < 0.5`
- `conf = |p - 0.5|`

解释：

- `p` 越远离 `0.5`，模型越“偏向某一边”
- `conf` 越大，表示方向判断越强

如果设置阈值 `conf_thr`，则：

- 只有 `conf >= conf_thr` 才算真正触发信号

例如：

- `conf_thr = 0.15`
- 等价于只接受 `p >= 0.65` 或 `p <= 0.35`

## 第五步：模型怎么评估

分钟主线最后输出几类核心指标：

### `all_hit`

所有滚动预测点上，方向预测是否正确的命中率。

### `triggered_hit`

只在高置信度触发点上，方向预测是否正确的命中率。

### `coverage`

触发样本占全部 OOS 样本的比例。

### `avg_vol_decimal`

在触发点上，按模型方向对齐后的平均 IV 变化：

`mean(pred_sign * y)`

### `avg_vol_points`

把上面的值乘以 100，转换成常说的 vol points。

例如：

- 如果 `avg_vol_decimal = 0.012`
- 那么 `avg_vol_points = 1.2`

## 一键流水线怎么跑

代码位置：

- [run_sc_pipeline.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/run_sc_pipeline.py)

它做的事情是：

1. 读取 SC 原始 parquet
2. 调用分钟级数据构建脚本，生成 `mainpool_1m`
3. 基于 `mainpool_1m` 构造因子和标签
4. 用默认因子组合跑 baseline
5. 逐个测试第二批候选因子的 add-one 效果
6. 再跑一个最终保留组合
7. 输出 JSON 报告

默认 baseline 特征：

- `flow`
- `iv_dev_ema5_ratio`
- `iv_mom3`
- `iv_willr10`
- `resid_z`

第二批候选：

- `flow_ema10`
- `shockF`

最终保留组合：

- `flow`
- `iv_dev_ema5_ratio`
- `iv_mom3`
- `iv_willr10`
- `resid_z`
- `flow_ema10`
- `shockF`

## 模块化脚本与主线脚本的关系

项目里同时存在一组更早期、更通用的 skeleton 脚本：

- `src/build_dataset.py`
- `src/build_features.py`
- `src/build_labels.py`

它们的定位更像是：

- 原型阶段的通用模板
- 用于说明 “对齐数据 -> 构建特征 -> 构建标签” 这三个步骤的基本分层

真正和 SC / volresearch 数据接得最完整、最接近当前研究结论的，还是：

- `load_volresearch_data.py`
- `sc_factor_lib.py`
- `sc_train_eval.py`
- `run_sc_pipeline.py`

## 秒级版本和分钟主线的关系

### `v1.1` 秒级版本

核心脚本：

- [run_sc_v2_second_p_5mblend.py](e:/openclaw_workspace/volresearch/iv_vega_hf/v1.1/src/run_sc_v2_second_p_5mblend.py)

它延续了分钟主线的核心思想，但做了三个变化：

1. ATM 池化改成秒级
2. 特征用过去 5 分钟的秒级上下文构造
3. 标签改成未来 `3m~5m` 平均 IV 相对当前 IV 的变化

具体标签是：

`y_t = mean(IV[t+3m : t+5m]) - IV[t]`

然后同样转成分类目标：

- `y_t > 0` 记作 1
- 否则记作 0

逻辑回归仍然是 walk-forward expanding refit，只不过：

- 每个样本点变成秒级
- `min_train=3600` 表示至少需要 3600 个秒级样本

### `v1.5`

`v1.5` 是为了更接近在线场景做的增量实验：

- 允许跨交易日续用参数
- 保存 `.npz` 模型快照
- 支持首日 warmup、后续日直接续跑

它不是对 `v1.1` 的完全等价重写，而是更偏部署风格的实验分支。

## 这套研究框架的核心思想

把项目抽象一下，这套逻辑真正想抓的是三种驱动：

1. IV 自身是否处在高位 / 低位，是否有短期延续或均值回复
2. 市场是否在持续净买 / 净卖 vega
3. 标的价格冲击能解释多少 IV 变化，解释不了的“异常部分”有没有信息

所以从结构上看，它不是一个海量因子库项目，而是一个很聚焦的框架：

- 一个 ATM 池化 IV 指数
- 一组少量但有经济含义的因子
- 一个严格时间顺序的滚动逻辑回归

## 当前理解下的优点

- 数据链路清楚，能从原始 parquet 一直走到分钟级模型评估
- 因子数量少，解释性强
- 训练方式严格按时间滚动，避免未来信息泄露
- 输出的 `p / conf / coverage / triggered_hit` 很适合做信号筛选

## 当前理解下的限制

- 分钟版 ATM 池化仍然是简化版，不是完整状态延拓式指数
- 逻辑回归是线性分类器，表达能力有限
- 概率 `p` 没有额外做 calibration
- `shockF` 的权重和若干滚动窗口属于研究型经验参数

## 建议从哪里开始读代码

如果你第一次接这个项目，推荐顺序是：

1. `src/load_volresearch_data.py`
   - 先理解分钟级 `mainpool_1m` 是怎么来的
2. `src/sc_factor_lib.py`
   - 再理解因子和标签的定义
3. `src/sc_train_eval.py`
   - 看滚动逻辑回归怎么训练和评估
4. `src/run_sc_pipeline.py`
   - 最后看完整流程怎么串起来
5. `v1.1/src/run_sc_v2_second_p_5mblend.py`
   - 如果还要看秒级扩展，再读这个

## 相关文件

- [src/load_volresearch_data.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/load_volresearch_data.py)
- [src/sc_factor_lib.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/sc_factor_lib.py)
- [src/sc_train_eval.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/sc_train_eval.py)
- [src/run_sc_pipeline.py](e:/openclaw_workspace/volresearch/iv_vega_hf/src/run_sc_pipeline.py)
- [v1.1/src/run_sc_v2_second_p_5mblend.py](e:/openclaw_workspace/volresearch/iv_vega_hf/v1.1/src/run_sc_v2_second_p_5mblend.py)

