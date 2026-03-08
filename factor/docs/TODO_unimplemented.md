# TODO / 暂未实现

1. 多档盘口（L2-L10）相关指标
- 当前原始表只稳定包含 bidprice1/askprice1/bidvol1/askvol1。
- 深度斜率、凸性等高级盘口指标需要更多档位。

2. 严格事件驱动对齐
- 当前以时间重采样 + rolling 实现，后续可改成 event-time 桶。

3. 因子筛选与模型训练
- 本次先落地因子工程与目标字段，未加入完整训练脚本（可后续接 sklearn/lightgbm）。
