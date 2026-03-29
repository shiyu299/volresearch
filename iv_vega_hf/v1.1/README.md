# v2: 秒级触发 + 5分钟因子

目标：
- 用秒级数据做触发（trigger timeline）
- 用5分钟聚合数据构建因子与标签

脚本：
- `src/run_sc_v2_second_trigger_5m_factor.py`

运行：
```bash
python3 /Users/shiyu/.openclaw/workspace/iv_vega_hf/v2/src/run_sc_v2_second_trigger_5m_factor.py
```

输出：
- `reports/SC_V2_SECOND_TRIGGER_5M_FACTOR.json`

当前规则：
- 期货单边市秒级过滤（bid/ask 任一侧缺失/<=0 即剔除）
- 因子频率：5min
- 触发频率：1s（将5min信号扩展到对应5分钟秒级区间）
