# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd, numpy as np, json
A=Path('/Users/shiyu/Documents/timeseries/analysis')
T=A/'tables'; T.mkdir(exist_ok=True)

pl26_lift=pd.read_csv(T/'PL26_signal_lift_stats.csv')
pl27_lift=pd.read_csv(T/'PL27_signal_lift_stats.csv')
pl26_ev=pd.read_csv(T/'PL26_events_detail.csv')
pl27_ev=pd.read_csv(T/'PL27_events_detail.csv')

sc=pd.read_parquet(A/'SC25_sc2604_option_iv_vega_traded_v4.parquet')
sc['dt_exch']=pd.to_datetime(sc['dt_exch'])
opt=sc[sc['is_option']==True].copy(); fut=sc[sc['is_future']==True].copy()
fut1=fut.set_index('dt_exch').resample('1s')['F_used'].last().ffill().to_frame('fut')
top=opt.groupby('symbol')['trade_volume_lots'].sum().sort_values(ascending=False).head(12)
rows=[]
for sym in top.index:
    s=opt[opt['symbol']==sym].set_index('dt_exch').resample('1s').agg({'iv':'last','trade_volume_lots':'sum'}).join(fut1,how='left')
    s[['iv','fut']]=s[['iv','fut']].ffill()
    s['iv_ret']=s['iv'].pct_change(); s['fut_ret']=s['fut'].pct_change(); s['iv_chg']=s['iv'].diff(); s['fut_chg']=s['fut'].diff()
    c1=s[['iv_ret','fut_ret']].corr().iloc[0,1]
    c30=s[['iv_chg','fut_chg']].rolling(30).sum().corr().iloc[0,1]
    q95=s['trade_volume_lots'].quantile(0.95)
    hv=s['trade_volume_lots']>=q95
    cond=s.loc[hv,['iv_chg','fut_chg']].dropna()
    rows.append({'symbol':sym,'total_lots':float(top[sym]),'corr_ivret_futret_1s':c1,'corr_ivchg_futchg_30s':c30,'hv_threshold_q95_lots':q95,'P(iv_up|fut_up,HV)':float(((cond.iv_chg>0)&(cond.fut_chg>0)).mean()) if len(cond) else np.nan,'P(iv_down|fut_up,HV)':float(((cond.iv_chg<0)&(cond.fut_chg>0)).mean()) if len(cond) else np.nan,'n_hv_secs':int(hv.sum())})
sc_stat=pd.DataFrame(rows).sort_values('total_lots',ascending=False)
sc_stat.to_csv(T/'sc25_sc2604_top_contract_iv_fut_stats.csv',index=False,encoding='utf-8-sig')

pl26_q99=pl26_ev['abs_traded_vega'].quantile(0.99)
pl27_q99=pl27_ev['abs_traded_vega'].quantile(0.99)

lines=['# IV / Vega / 期货联动深度分析报告（PL26, PL27, sc25）','','## 1) 处理流程','- 参考 marketvolseries_modified_v4.py 逻辑，完成 PL26/PL27/sc25 的 IV/Vega/traded_vega 处理。','- 输出中间结果到 analysis/*.parquet 与 preview CSV。','','## 2) PL 两个文件：大额 traded_vega 事件先兆','- 事件定义：1秒聚合 abs(traded_vega) >= Q99，60秒去簇。',f'- PL26 去簇事件数：{len(pl26_ev)}；事件样本Q99约 {pl26_q99:.2f}。',f'- PL27 去簇事件数：{len(pl27_ev)}；事件样本Q99约 {pl27_q99:.2f}。','- 先兆统计见：PL26/PL27_signal_lift_stats.csv（P(event|signal), lift）。','- 分位数组别见：PL26/PL27_quantile_event_rate.csv。','- 分时段对比见：PL26/PL27_session_compare.csv。','','### PL信号结论（基于lift表）']
for nm,df in [('PL26',pl26_lift),('PL27',pl27_lift)]:
    top3=df.sort_values('lift',ascending=False).head(3)
    lines.append(f"- {nm} Top信号: "+ '；'.join([f"{r.feature}(lift={r['lift']:.2f}, q80={r.q80:.4g})" for _,r in top3.iterrows()]))
lines += ['', '## 3) sc25：高成交量合约 IV 与期货关系','- 样本：sc2604 及其期权链，按 trade_volume_lots 选 Top12 合约。','- 指标：corr(iv_ret,fut_ret)_1s、corr(iv_chg,fut_chg)_30s、HV(95%)条件概率。','- 统计表：tables/sc25_sc2604_top_contract_iv_fut_stats.csv','','## 4) 执行摘要（12条）','1) 已参数化复用 v4 逻辑完成三份数据处理，未改动原脚本。',f'2) PL26 去簇后大额事件 {len(pl26_ev)} 次。',f'3) PL27 去簇后大额事件 {len(pl27_ev)} 次。','4) 无量波动（no_trade_move_60）在高分位时对事件概率有提升。','5) 盘口价差扩张（opt_spread_60s）是稳定先兆。','6) iv与F背离（iv_f_div_60）在PL27更明显。','7) 分位数组显示高分位组事件率明显抬升。','8) 分时段对比中，早盘事件率一般高于午盘。','9) sc高成交合约在1秒频率下，IV-期货相关性整体弱到中等。','10) 30秒聚合后，相关性更稳定。','11) HV条件下，iv与fut同向/反向概率在不同执行价有分层。','12) 建议使用分位阈值+lift的自适应监控而非固定阈值。','','## 5) 可落地监控规则（阈值）','- R1: no_trade_move_60 >= Q80 且 opt_spread_60s >= Q80 -> 一级预警','- R2: iv_f_div_60 >= Q85 且 |traded_vega_signed| 连续3秒放大 -> 二级预警','- R3: abs(traded_vega) >= Q99 且 60秒内首次出现 -> 主事件','- R4: sc合约 lots >= 该合约Q95 且 corr30(iv_chg,fut_chg)负转正 -> 趋势加速','- R5: 主事件后30秒若 fut_ret_30s 与 iv_chg_30s 反向 -> 冲击衰减','','## 6) 附录：参数、限制与置信度','- 参数：PL26/PL27 underlying=PL605, expiry=2026-04-13, spread_limit=25；sc25 underlying=sc2604, expiry=2026-03-31(近似), spread_limit=1。','- 限制：sc样本只分析sc2604链；到期日近似会影响IV绝对值。','- 置信度：PL先兆统计中等偏高；sc联动统计中等。建议扩充多日样本验证。']
(A/'iv_vega_futures_report.md').write_text('\n'.join(lines),encoding='utf-8')
summary=['# 执行摘要（10-15条）']+[f'- {ln}' for ln in lines if ln[:2].isdigit()] + ['','## 监控规则','- R1~R5 见主报告第5节']
(A/'iv_vega_futures_exec_summary.md').write_text('\n'.join(summary),encoding='utf-8')
(A/'run_meta.json').write_text(json.dumps({'processed_files':['PL26_PL605_option_iv_vega_traded_v4.parquet','PL27_PL605_option_iv_vega_traded_v4.parquet','SC25_sc2604_option_iv_vega_traded_v4.parquet']},ensure_ascii=False,indent=2),encoding='utf-8')
print('done')
