import pandas as pd,numpy as np,warnings,sys;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
rng=np.random.default_rng(1)
d=pd.read_csv('signals_dataset.csv',encoding='utf-8-sig',parse_dates=['signal_day'],keep_default_na=False,na_values=[''])
d['grade']=d.grade.fillna('');d=d.dropna(subset=['r5','b5']).copy()
d['ex']=d.r5-d.b5;d['up']=(d.r5>0)*1.0;d['g']=d.grade!=''
d['bad']=((d.r5<=-5)|(d.r10<=-8)).astype(float)
print(d.earn_in_hold.value_counts(dropna=False).head(4).to_dict(), d.dist_high252.describe()[['min','50%','max']].to_dict(), d.n_base_day.describe()[['50%','max']].to_dict(),d.spx_vs_ma200.describe()[['min','50%','max']].to_dict())
feats={
'earn_in_hold':d.earn_in_hold.astype(float)==1,
'deep252&atr>=3.5':(d.dist_high252<=-35)&(d.atr_pct>=3.5),
'ret20d<=-15':d.ret20d<=-15,
'atr>=4':d.atr_pct>=4,
'spx<ma200':d.spx_vs_ma200<0,
'n_base_day>=5':d.n_base_day>=5,
'isolated(n_base_day==1)':d.n_base_day==1}
def cb(x,m,col,B=3000):
  days=np.sort(x.signal_day.unique());di=pd.Series(range(len(days)),index=days)[x.signal_day].values;nd=len(days)
  v=x[col].values;mm=m.values
  a=np.bincount(di[mm],v[mm],nd);ac=np.bincount(di[mm],minlength=nd);b=np.bincount(di[~mm],v[~mm],nd);bc=np.bincount(di[~mm],minlength=nd)
  est=a.sum()/ac.sum()-b.sum()/bc.sum();r=[]
  for _ in range(B):
    k=rng.integers(0,nd,nd)
    if ac[k].sum() and bc[k].sum(): r.append(a[k].sum()/ac[k].sum()-b[k].sum()/bc[k].sum())
  lo,hi=np.percentile(r,[2.5,97.5]);return est,lo,hi,'*' if lo>0 or hi<0 else ''
for scope,x in [('ALL',d),('graded',d[d.g]),('ungraded',d[~d.g])]:
  for per,xx in [('IS',x[x.oos2026==0]),('OOS',x[x.oos2026==1])]:
    for f,m in feats.items():
      mm=m.loc[xx.index].fillna(False)
      if mm.sum()<10 or (~mm).sum()<10: continue
      e=cb(xx,mm,'ex');b=cb(xx,mm,'bad')
      print(scope,per,f,'n=%d'%mm.sum(),'ex %.2f[%.2f,%.2f]%s'%e,'bad %.3f[%.3f,%.3f]%s'%b)
G=d[d.g]
print(G.groupby('grade').agg(n=('r5','size'),wr=('combo_wr5','mean'),up=('up','mean'),ex=('ex','mean'),bad=('bad','mean')).round(3).to_string())
for o in [0,1]:
  x=G[G.oos2026==o];print('oos',o,'spearman wr5~ex',round(x.combo_wr5.corr(x.ex,method='spearman'),3),'wr5~up',round(x.combo_wr5.corr(x.up,method='spearman'),3))
  x=x.assign(q=pd.qcut(x.combo_wr5,4,duplicates='drop'));print(x.groupby('q').agg(n=('r5','size'),up=('up','mean'),ex=('ex','mean')).round(3).to_string())
x=G.copy();x['z']=x.combo_wr5-x.combo_wr5.mean();days=np.sort(x.signal_day.unique())
grp=x.groupby('signal_day');sxy=grp.apply(lambda t:(t.z*t.ex).sum()).reindex(days).values;sxx=grp.apply(lambda t:(t.z**2).sum()).reindex(days).values
sl=[sxy[k].sum()/sxx[k].sum() for k in (rng.integers(0,len(days),len(days)) for _ in range(3000))]
print('slope ex per 1pt wr5 %.3f'%(sxy.sum()/sxx.sum()),np.percentile(sl,[2.5,97.5]).round(3))
