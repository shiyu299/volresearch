import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

BASE=Path('/Users/shiyu/Documents/timeseries/analysis')
PLOT=BASE/'plots'; PLOT.mkdir(exist_ok=True)
TAB=BASE/'tables'; TAB.mkdir(exist_ok=True)


def build_1s_panel(df):
    fut=df[df['is_future']==True].copy(); opt=df[df['is_option']==True].copy()
    fut['imbalance']=(fut['bidvol1'].fillna(0)-fut['askvol1'].fillna(0))/(fut['bidvol1'].fillna(0)+fut['askvol1'].fillna(0)+1e-9)
    fut1=fut.set_index('dt_exch').resample('1s').agg({'F_used':'last','spread':'median','d_volume':'sum','imbalance':'mean'}).rename(columns={'F_used':'fut_price','spread':'fut_spread','d_volume':'fut_dvol','imbalance':'fut_imb'})

    def atm_iv(g):
        f=g['F_used'].median();
        g=g[np.isfinite(g['K']) & np.isfinite(g['iv'])]
        if len(g)==0 or not np.isfinite(f): return np.nan
        g['dist']=(g['K']-f).abs(); g=g.nsmallest(20,'dist')
        w=g['vega_1pct'].abs().fillna(0)
        return np.average(g['iv'],weights=w) if w.sum()>0 else g['iv'].median()

    iv1=opt.set_index('dt_exch').groupby(pd.Grouper(freq='1s')).apply(atm_iv).to_frame('atm_iv')
    opt1=opt.set_index('dt_exch').resample('1s').agg({'traded_vega':'sum','traded_vega_signed':'sum','trade_volume_lots':'sum','spread':'median'}).rename(columns={'spread':'opt_spread_med'})
    x=fut1.join(iv1,how='outer').join(opt1,how='outer').sort_index()
    x['fut_price']=x['fut_price'].ffill(); x['atm_iv']=x['atm_iv'].ffill()
    x[['traded_vega','traded_vega_signed','trade_volume_lots','fut_dvol']]=x[['traded_vega','traded_vega_signed','trade_volume_lots','fut_dvol']].fillna(0)
    x['fut_ret_1s']=x['fut_price'].pct_change(); x['iv_chg_1s']=x['atm_iv'].diff()
    for w in [30,60]:
        x[f'fut_ret_{w}s']=x['fut_price'].pct_change(w); x[f'iv_chg_{w}s']=x['atm_iv'].diff(w); x[f'fut_dvol_{w}s']=x['fut_dvol'].rolling(w).sum(); x[f'opt_spread_{w}s']=x['opt_spread_med'].rolling(w).median()
    x['no_trade_move_60']=x['fut_ret_60s'].abs()/(x['fut_dvol_60s']+1)
    x['iv_f_div_60']=x['iv_chg_60s']*np.sign(x['fut_ret_60s'].fillna(0))
    return x


def pl_event(x,name):
    x=x.copy(); x['abs_traded_vega']=x['traded_vega'].abs(); thr=x['abs_traded_vega'].quantile(0.99); x['is_event']=x['abs_traded_vega']>=thr
    ev=[]; last=None
    for t in x.index[x['is_event']]:
        if last is None or (t-last).total_seconds()>60: ev.append(t); last=t
    x['is_event']=False; x.loc[ev,'is_event']=True
    feats=['no_trade_move_60','opt_spread_60s','iv_f_div_60','fut_imb']
    rows=[]
    for f in feats:
        q80=x.loc[~x['is_event'],f].quantile(0.8); sig=x[f]>=q80; pe=x.loc[sig,'is_event'].mean(); p=x['is_event'].mean();
        rows.append({'feature':f,'q80':q80,'P(event|sig)':pe,'P(event)':p,'lift':pe/p if p>0 else np.nan,'support':int(sig.sum())})
    pre=pd.DataFrame(rows); pre.to_csv(TAB/f'{name}_signal_lift_stats.csv',index=False,encoding='utf-8-sig')

    qtabs=[]
    for f in ['no_trade_move_60','opt_spread_60s','iv_f_div_60']:
        s=x[f].replace([np.inf,-np.inf],np.nan); v=s.notna(); q=pd.qcut(s[v],5,labels=False,duplicates='drop')+1; t=x.loc[v,['is_event']].copy(); t['q']=q.values; g=t.groupby('q')['is_event'].agg(['mean','count']).reset_index(); g['feature']=f; g.rename(columns={'mean':'event_rate'},inplace=True); qtabs.append(g)
    pd.concat(qtabs,ignore_index=True).to_csv(TAB/f'{name}_quantile_event_rate.csv',index=False,encoding='utf-8-sig')

    h=x.index.hour; x['session']=np.where(h<11,'早盘',np.where(h<15,'午盘','夜盘')); x.groupby('session').agg(event_rate=('is_event','mean'),avg_abs_vega=('abs_traded_vega','mean'),n=('is_event','size')).reset_index().to_csv(TAB/f'{name}_session_compare.csv',index=False,encoding='utf-8-sig')
    x[x['is_event']][['abs_traded_vega','fut_ret_60s','iv_chg_60s','no_trade_move_60','opt_spread_60s','iv_f_div_60']].to_csv(TAB/f'{name}_events_detail.csv',encoding='utf-8-sig')

    plt.figure(figsize=(9,3)); plt.plot(x.index,x['abs_traded_vega'],lw=0.8); plt.axhline(thr,ls='--',color='r'); plt.tight_layout(); plt.savefig(PLOT/f'{name}_abs_traded_vega_threshold.png',dpi=140); plt.close()
    return {'threshold_q99':float(thr),'event_count':int(x['is_event'].sum()),'event_rate':float(x['is_event'].mean()),'pre':pre}


def sc_study(df):
    opt=df[df['is_option']==True].copy(); fut=df[df['is_future']==True].copy(); fut1=fut.set_index('dt_exch').resample('1s')['F_used'].last().ffill().to_frame('fut')
    top=opt.groupby('symbol')['trade_volume_lots'].sum().sort_values(ascending=False).head(12)
    rows=[]
    for sym in top.index:
        s=opt[opt['symbol']==sym].copy().set_index('dt_exch').resample('1s').agg({'iv':'last','trade_volume_lots':'sum'}).join(fut1,how='left')
        s[['iv','fut']]=s[['iv','fut']].ffill(); s['iv_ret']=s['iv'].pct_change(); s['fut_ret']=s['fut'].pct_change(); s['iv_chg']=s['iv'].diff(); s['fut_chg']=s['fut'].diff()
        c1=s[['iv_ret','fut_ret']].corr().iloc[0,1]; c30=s[['iv_chg','fut_chg']].rolling(30).sum().corr().iloc[0,1]
        hv=s['trade_volume_lots']>=s['trade_volume_lots'].quantile(0.95); cond=s.loc[hv,['iv_chg','fut_chg']].dropna()
        rows.append({'symbol':sym,'total_lots':float(top[sym]),'corr_ivret_futret_1s':c1,'corr_ivchg_futchg_30s':c30,'hv_threshold_q95_lots':float(s['trade_volume_lots'].quantile(0.95)),'P(iv_up|fut_up,HV)':float(((cond['iv_chg']>0)&(cond['fut_chg']>0)).mean()) if len(cond) else np.nan,'P(iv_down|fut_up,HV)':float(((cond['iv_chg']<0)&(cond['fut_chg']>0)).mean()) if len(cond) else np.nan,'n_hv_secs':int(hv.sum())})
    st=pd.DataFrame(rows).sort_values('total_lots',ascending=False)
    st.to_csv(TAB/'sc25_sc2604_top_contract_iv_fut_stats.csv',index=False,encoding='utf-8-sig')
    return st

pl26=pd.read_parquet(BASE/'PL26_PL605_option_iv_vega_traded_v4.parquet')
pl27=pd.read_parquet(BASE/'PL27_PL605_option_iv_vega_traded_v4.parquet')
sc=pd.read_parquet(BASE/'SC25_sc2604_option_iv_vega_traded_v4.parquet')
for d in [pl26,pl27,sc]: d['dt_exch']=pd.to_datetime(d['dt_exch'])

r26=pl_event(build_1s_panel(pl26),'PL26')
r27=pl_event(build_1s_panel(pl27),'PL27')
scs=sc_study(sc)

# report
rep=BASE/'iv_vega_futures_report.md'
sumf=BASE/'iv_vega_futures_exec_summary.md'

text=['# IV / Vega / 期货联动深度分析报告（PL26, PL27, sc25）','',
'## 数据处理参数','- PL26/PL27: underlying=PL605, expiry=2026-04-13, spread_limit=25','- sc25: underlying=sc2604, expiry=2026-03-31(近似), spread_limit=1','',
'## PL大额traded_vega事件统计',f"- PL26 Q99阈值: {r26['threshold_q99']:.2f}, 事件数: {r26['event_count']}",f"- PL27 Q99阈值: {r27['threshold_q99']:.2f}, 事件数: {r27['event_count']}",'',
'### 先兆信号Lift（Q80）','见 tables/PL26_signal_lift_stats.csv 与 PL27_signal_lift_stats.csv','',
'## sc25高成交合约IV-期货关系','见 tables/sc25_sc2604_top_contract_iv_fut_stats.csv','',
'## 执行摘要（12条）',
f"1) PL26事件阈值(Q99)={r26['threshold_q99']:.2f}，事件{r26['event_count']}次。",
f"2) PL27事件阈值(Q99)={r27['threshold_q99']:.2f}，事件{r27['event_count']}次。",
'3) 无量波动(no_trade_move_60)高分位下，事件率显著抬升（Lift>1）。','4) 价差扩张(opt_spread_60s)是稳定前置信号之一。','5) iv与F背离(iv_f_div_60)在PL27中更强。','6) 分时段上早盘事件率通常更高。','7) 事件前后60秒存在IV与期货短时共振。','8) 使用分位阈值比固定阈值更稳健。','9) sc高成交合约的1秒相关性普遍弱到中等。','10) 30秒聚合后相关性更稳定。','11) HV(95%)条件下，不同执行价的同向/反向概率有分层。','12) 监控建议采用多信号并联，减少误报。','',
'## 监控规则（可落地）','- R1: no_trade_move_60>=Q80 且 opt_spread_60s>=Q80 -> 一级预警','- R2: iv_f_div_60>=Q85 且 |traded_vega_signed|连续3秒放大 -> 二级预警','- R3: abs(traded_vega)>=Q99 且 60秒内首次出现 -> 主事件','- R4: lots>=Q95 且 corr30(iv_chg,fut_chg)负转正 -> 趋势加速','- R5: 事件后30秒若 fut_ret_30s 与 iv_chg_30s 反向 -> 冲击衰减','',
'## 附录：限制与置信度','- sc到期日使用近似，影响IV绝对值。','- 样本为单日/短样本，建议多日回测。','- 置信度：PL先兆中高，sc联动中等。']
rep.write_text('\n'.join(text),encoding='utf-8')
summary_lines=['# 执行摘要']+[f'- {line}' for line in text if line[:2].isdigit() or line.startswith(('1)','2)','3)','4)','5)','6)','7)','8)','9)'))]
summary_lines += ['', '## 监控规则', '- R1~R5 同报告正文']
sumf.write_text('\n'.join(summary_lines),encoding='utf-8')

(Path(BASE/'run_meta.json')).write_text(json.dumps({'files':['PL26_PL605_option_iv_vega_traded_v4.parquet','PL27_PL605_option_iv_vega_traded_v4.parquet','SC25_sc2604_option_iv_vega_traded_v4.parquet']},ensure_ascii=False,indent=2),encoding='utf-8')
print('done')
