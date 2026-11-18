import pandas as pd, numpy as np
df=pd.read_csv('signals_dataset.csv',encoding='utf-8-sig',parse_dates=['signal_day'],keep_default_na=False,na_values=[''])
df['grade']=df['grade'].fillna('')
d=df[(df.grade!='')&(df.live_like==1)&df.r5.notna()].copy()
d['ex5']=d.r5-d.b5
d['win']=((d.r5>=6)|(d.r10>=10)).astype(int)
d['beta']=((d.win==1)&(d.ex5<3)).astype(int); d['alpha']=((d.win==1)&(d.ex5>=3)).astype(int)
print(len(d),d.oos2026.value_counts().to_dict())
for o in (0,1):
    s=d[d.oos2026==o]; print('oos',o,'n',len(s),'win',s.win.mean().round(3),'beta',s.beta.sum(),'alpha',s.alpha.sum(),'up5',(s.r5>0).mean().round(3),'ex',s.ex5.mean().round(2),'b5 win',s[s.win==1].b5.mean().round(2),'b5 rest',s[s.win==0].b5.mean().round(2))
ins=d[d.oos2026==0]
feats=['rsi','cci','cmf','wr','hist','dist_52w','deviation','vol_ratio5','ret5d','ret20d','ret60d','dist_high252','dist_ma200','atr_pct','gap_open_pct','spx_vs_ma200','spx_ret5','spx_ret20','vix','vix_chg5','n_base_day','n_graded_day','sector_n_day','days_to_earn','earn_in_hold']
from scipy import stats
rows=[]
for f in feats:
    a=ins[ins.win==1][f].dropna();b=ins[ins.win==0][f].dropna()
    # day-clustered: mean per day then compare
    rows.append((f,a.median(),b.median(),stats.mannwhitneyu(a,b).pvalue, ins[[f,'ex5']].corr(method='spearman').iloc[0,1]))
print(pd.DataFrame(rows,columns=['f','win_med','rest_med','p','spear_ex5']).round(3).to_string())
# quintile / threshold
def rep(name,mask):
    out=[name]
    for o in (0,1):
        s=d[(d.oos2026==o)&mask.reindex(d.index,fill_value=False)]
        out.append(f"o{o}: n={len(s)} days={s.signal_day.nunique()} up5={(s.r5>0).mean():.2f} ex={s.ex5.mean():.2f} win={s.win.mean():.2f}")
    print(' | '.join(out))
rep('base',d.r5.notna())
for nm,m in {
 'vix>=25':d.vix>=25,'vix_chg5<0':d.vix_chg5<0,'vix_chg5<-5':d.vix_chg5<-5,'spx_ret5>0':d.spx_ret5>0,'spx<ma200':d.spx_vs_ma200<0,
 'earn_in_hold':d.earn_in_hold==1,'no earn':d.earn_in_hold==0,
 'atr>=4':d.atr_pct>=4,'atr>=3':d.atr_pct>=3,'ret60<-20':d.ret60d<-20,'ret20<-15':d.ret20d<-15,'dist252<-35':d.dist_high252<-35,'dist200<-15':d.dist_ma200<-15,
 'vol>=1.5':d.vol_ratio5>=1.5,'vol>=2':d.vol_ratio5>=2,'gap<-2':d.gap_open_pct<-2,'gap>0':d.gap_open_pct>0,
 'sector_n>=3':d.sector_n_day>=3,'sector_n==1':d.sector_n_day==1,'n_graded>=5':d.n_graded_day>=5,'n_graded==1':d.n_graded_day==1,
 'cmf<-.1':d.cmf<-0.1,'cmf>0':d.cmf>0,'rsi<25':d.rsi<25,'cci<-150':d.cci<-150}.items(): rep(nm,m)
print(d.groupby('grade').apply(lambda s:pd.Series({'n':len(s),'win':s.win.mean(),'ex':s.ex5.mean()})).round(2))
ins2=ins.copy()
for c in ['sector']:
    print(ins2.groupby(c).apply(lambda s:pd.Series({'n':len(s),'win':s.win.mean(),'ex':s.ex5.mean()})).round(2).sort_values('ex'))
# conds
from collections import defaultdict
cs=set(c for x in d.conds.dropna() for c in str(x).replace('|',',').replace('+',',').split(','))
print(list(cs)[:30])
for c in sorted(cs):
    c=c.strip()
    if c: rep('cond '+c,d.conds.fillna('').str.contains(c,regex=False))
# bootstrap by day for key
def boot(mask,nb=2000):
    s=d[(d.oos2026==0)&mask]; g=s.groupby('signal_day').ex5.agg(['sum','count']); 
    days=g.index.values; r=np.random.default_rng(0); res=[]
    for _ in range(nb):
        i=r.integers(0,len(g),len(g)); res.append(g['sum'].values[i].sum()/g['count'].values[i].sum())
    return np.percentile(res,[2.5,97.5]).round(2)
print('boot base',boot(d.r5.notna()))
