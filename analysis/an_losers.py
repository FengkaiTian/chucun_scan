import pandas as pd,numpy as np
d=pd.read_csv('signals_dataset.csv',encoding='utf-8-sig',parse_dates=['signal_day'],keep_default_na=False,na_values=[''])
d['grade']=d['grade'].fillna('')
g=d[(d.grade!='')&(d.live_like==1)].dropna(subset=['r5']).copy()
g['ex']=g.r5-g.b5
g['bad']=((g.r5<=-5)|(g.r10<=-8)).astype(int)
g['up']=(g.r5>0).astype(int)
tr=g[g.oos2026==0];te=g[g.oos2026==1]
print(len(tr),len(te),tr.bad.mean(),te.bad.mean(),tr.up.mean(),te.up.mean())
feats=['rsi','cmf','hist','dist_52w','deviation','vol_ratio5','ret5d','ret20d','ret60d','dist_high252','dist_ma200','atr_pct','gap_open_pct','spx_vs_ma200','spx_ret5','spx_ret20','vix','vix_chg5','n_base_day','n_graded_day','sector_n_day','days_to_earn','days_since_earn','earn_in_hold','combo_wr5']
for nm,s in [('IS',tr),('OOS',te)]:
    print(nm);print(s.groupby('bad')[feats].median().T.round(2).to_string())
print(g.groupby('sector').agg(n=('bad','size'),bad=('bad','mean'),up=('up','mean')).round(2))
def rule(name,f):
    out=[]
    for nm,s in [('IS',tr),('OOS',te)]:
        m=f(s);k=s[m];r=s[~m]
        out.append(f"{nm} hit n={len(k)} up={k.up.mean():.2f} bad={k.bad.mean():.2f} ex={k.ex.mean():.2f} | rest n={len(r)} up={r.up.mean():.2f} bad={r.bad.mean():.2f} ex={r.ex.mean():.2f} | killed winners={k.up.mean()*100:.0f}% ")
    # day-cluster bootstrap on all
    ud,inv=np.unique(g.signal_day.values,return_inverse=True);K=len(ud);rng=np.random.default_rng(0)
    m=f(g).values.astype(float);ex=g.ex.values
    A=np.bincount(inv,m,K);B=np.bincount(inv,m*ex,K);C=np.bincount(inv,1-m,K);D=np.bincount(inv,(1-m)*ex,K)
    diffs=[]
    for _ in range(2000):
        i=rng.integers(0,K,K);a_=A[i].sum();c_=C[i].sum()
        if a_>0 and c_>0: diffs.append(B[i].sum()/a_-D[i].sum()/c_)
    print(name,'\n  '+'\n  '.join(out),f'\n  boot ex diff CI {np.percentile(diffs,[2.5,97.5]).round(2)} P(<0)={np.mean(np.array(diffs)<0):.3f}')
rule('earn_in_hold',lambda s:s.earn_in_hold==1)
rule('earn<=7',lambda s:s.days_to_earn<=7)
rule('post-earn fall',lambda s:(s.days_since_earn<=15)&(s.ret20d<=-10))
rule('ret20<-15',lambda s:s.ret20d<=-15)
rule('ma200<-15',lambda s:s.dist_ma200<=-15)
rule('atr>=5',lambda s:s.atr_pct>=5)
rule('atr>=4',lambda s:s.atr_pct>=4)
rule('spx<0',lambda s:s.spx_vs_ma200<0)
rule('vix>=25',lambda s:s.vix>=25)
rule('ngr>=3',lambda s:s.n_graded_day>=3)
rule('sec>=3',lambda s:s.sector_n_day>=3)
rule('gap<=-2',lambda s:s.gap_open_pct<=-2)
rule('earn_hold&ma200<-10',lambda s:(s.earn_in_hold==1)&(s.dist_ma200<=-10))
rule('earnhold&ret20<-10',lambda s:(s.earn_in_hold==1)&(s.ret20d<=-10))
rule('ma200<-15|atr>=5',lambda s:(s.dist_ma200<=-15)|(s.atr_pct>=5))
rule('ret60<-25',lambda s:s.ret60d<=-25)
rule('hi252<-40',lambda s:s.dist_high252<=-40)
