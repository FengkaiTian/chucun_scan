import pandas as pd, numpy as np
df=pd.read_csv('signals_dataset.csv',encoding='utf-8-sig',parse_dates=['signal_day'],keep_default_na=False,na_values=[''])
df['grade']=df['grade'].fillna('')
print(df.groupby([df.grade!='',df.live_like]).size()); df=df[df.r5.notna()&((df.live_like==1)|(df.grade==''))].copy()
df['ex5']=df.r5-df.b5
df['win']=((df.r5>=6)|(df.r10>=10)).astype(int)
df['lose']=((df.r5<=-5)|(df.r10<=-8)).astype(int)
df['g']=df.grade!=''
def boot(s):
    g=s.groupby('signal_day').ex5.agg(['sum','count']);r=np.random.default_rng(1)
    if len(g)<5:return None
    res=[g['sum'].values[i].sum()/g['count'].values[i].sum() for i in (r.integers(0,len(g),len(g)) for _ in range(1500))]
    return tuple(np.percentile(res,[2.5,97.5]).round(2))
def row(nm,m):
  for gg in (True,False):
    for o in (0,1):
        s=df[(df.g==gg)&(df.oos2026==o)&m]
        if len(s)<5: continue
        print(f"{nm} {'G' if gg else 'U'} o{o} n={len(s)} up5={(s.r5>0).mean():.2f} r5mean={s.r5.mean():.2f} med={s.r5.median():.2f} ex={s.ex5.mean():.2f} win={s.win.mean():.2f} lose={s.lose.mean():.2f} sd={s.r5.std():.1f} ci={boot(s)}")
row('all',df.r5.notna())
row('atr>=4',df.atr_pct>=4)
row('atr<4',df.atr_pct<4)
row('spx<200',df.spx_vs_ma200<0)
row('ngr==1',df.n_base_day==1)
row('nbase>=8',df.n_base_day>=8)
# control for market: regress ex5 within graded on solo + spx_ret5 + vix, day-clustered
import statsmodels.formula.api as smf
for gg in (True,False):
    s=df[df.g==gg].copy(); s['solo']=(s.n_graded_day==1 if gg else s.n_base_day==1).astype(int)
    s['many']=(s.n_graded_day>=5 if gg else s.n_base_day>=8).astype(int)
    for o in (0,1):
        t=s[s.oos2026==o].dropna(subset=['spx_ret5','vix','spx_vs_ma200','atr_pct'])
        m=smf.ols('ex5~solo+spx_ret5+vix+spx_vs_ma200+atr_pct',t).fit(cov_type='cluster',cov_kwds={'groups':t.signal_day.astype('int64')})
        print('G' if gg else 'U',o,len(t),m.params.round(2).to_dict(),'p_solo',round(m.pvalues['solo'],3),'p_atr',round(m.pvalues['atr_pct'],3))
