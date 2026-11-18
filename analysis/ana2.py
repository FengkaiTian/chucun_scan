exec(open('ana_winners.py').read().split("print(len(d)")[0])
import numpy as np
w=d[d.win==1]
for o in (0,1):
    s=w[w.oos2026==o]; print(o,len(s),'b5>=2:',(s.b5>=2).sum(),'ex<2:',(s.ex5<2).sum(),'med ex',s.ex5.median(),'med b5',s.b5.median())
def boot(m,o):
    s=d[(d.oos2026==o)&m]; g=s.groupby('signal_day').ex5.agg(['sum','count']);r=np.random.default_rng(1)
    res=[g['sum'].values[i].sum()/g['count'].values[i].sum() for i in (r.integers(0,len(g),len(g)) for _ in range(2000))]
    return np.percentile(res,[2.5,97.5]).round(2)
for nm,m in {'atr>=4':d.atr_pct>=4,'spx<ma200':d.spx_vs_ma200<0,'ngr>=5':d.n_graded_day>=5,'atr4|spx<200':(d.atr_pct>=4)|(d.spx_vs_ma200<0),'atr4&ret20<-15':(d.atr_pct>=4)&(d.ret20d<-15)}.items():
    for o in (0,1):
        s=d[(d.oos2026==o)&m]; print(nm,o,len(s),s.signal_day.nunique(),round((s.r5>0).mean(),2),round(s.ex5.mean(),2),round(s.win.mean(),2),boot(m,o))
