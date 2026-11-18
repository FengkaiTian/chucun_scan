import pandas as pd,numpy as np
d=pd.read_csv('signals_dataset.csv',encoding='utf-8-sig',parse_dates=['signal_day'],keep_default_na=False,na_values=[''])
d['grade']=d['grade'].fillna('')
d=d[(d.live_like==1)|(d.grade=='')].dropna(subset=['r5','r10']).copy()
d['ex']=d.r5-d.b5;d['bad']=((d.r5<=-5)|(d.r10<=-8)).astype(int);d['up']=(d.r5>0).astype(int);d['big']=(d.r5>=5).astype(int)
d['G']=d.grade!=''
rules={'atr>=4':d.atr_pct>=4,'earn_hold':d.earn_in_hold==1,'hi252<=-35&atr>=3.5':(d.dist_high252<=-35)&(d.atr_pct>=3.5)}
def row(s):return f"n={len(s)} r5mean={s.r5.mean():.2f} med={s.r5.median():.2f} up={s.up.mean():.2f} bad={s.bad.mean():.3f} big={s.big.mean():.3f} ex={s.ex.mean():.2f}"
for G in [True,False]:
  for o in [0,1]:
    b=d[(d.G==G)&(d.oos2026==o)];print('\n== graded' if G else '\n== ungraded','oos',o,'base',row(b))
    for k,m in rules.items():
        mm=m[b.index];print(' ',k,'HIT',row(b[mm]),'\n   ',' '*len(k),'REST',row(b[~mm]))
