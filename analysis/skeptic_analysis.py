import pandas as pd,numpy as np,warnings,sys;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
rng=np.random.default_rng(0)
d=pd.read_csv('signals_dataset.csv',encoding='utf-8-sig',parse_dates=['signal_day'],keep_default_na=False,na_values=[''])
d['grade']=d.grade.fillna('');d=d.dropna(subset=['r5','b5']).copy()
d['ex']=d.r5-d.b5;d['up']=(d.r5>0)*1.0
L=d[d.live_like==1]; G=L[L.grade!='']; U=d[d.grade=='']
def s(x):return f"n={len(x)} days={x.signal_day.nunique()} claimed={x.combo_wr5.mean():.1f} up={x.up.mean()*100:.1f} r5={x.r5.mean():.2f} b5={x.b5.mean():.2f} ex={x.ex.mean():.2f} upvsb={((x.r5>x.b5).mean()*100):.1f}"
print('ALL graded',s(G)); print('ungraded',s(U))
for o in [0,1]:
  print('oos',o,'G',s(G[G.oos2026==o]));print('oos',o,'U',s(U[U.oos2026==o]))
  for g in ['SSS','S','A','B']:
    x=G[(G.oos2026==o)&(G.grade==g)]
    if len(x):print('  ',g,s(x))
print('year');print(G.groupby(G.signal_day.dt.year).apply(lambda x:pd.Series(dict(n=len(x),claimed=x.combo_wr5.mean(),up=x.up.mean()*100,ex=x.ex.mean(),b5=x.b5.mean()))))
print(U.groupby(U.signal_day.dt.year).apply(lambda x:pd.Series(dict(n=len(x),up=x.up.mean()*100,ex=x.ex.mean(),b5=x.b5.mean()))))
def boot(x,col,B=10000):
  g=x.groupby('signal_day')[col].agg(['sum','count']);S=g['sum'].values;C=g['count'].values;n=len(g)
  idx=rng.integers(0,n,(B,n));m=S[idx].sum(1)/C[idx].sum(1)
  return S.sum()/C.sum(),np.percentile(m,[2.5,97.5]),n
for nm,x in [('G all',G),('G is',G[G.oos2026==0]),('G oos',G[G.oos2026==1]),('U all',U),('U oos',U[U.oos2026==1]),('U is',U[U.oos2026==0])]:
  print(nm,'ex',boot(x,'ex'),'up',boot(x,'up'))
for o in [0,1]:
  x=d[(d.oos2026==o)&((d.live_like==1)|(d.grade==''))].copy();x['g']=(x.grade!='')
  days=np.sort(x.signal_day.unique());ix={k:i for i,k in enumerate(days)}
  di=x.signal_day.map(ix).values
  def agg(m):
    return np.bincount(di[m],weights=x.ex.values[m],minlength=len(days)),np.bincount(di[m],minlength=len(days))
  gs,gc=agg(x.g.values);us,uc=agg(~x.g.values);ds=[]
  for _ in range(5000):
    k=rng.integers(0,len(days),len(days))
    if gc[k].sum()==0 or uc[k].sum()==0: continue
    ds.append(gs[k].sum()/gc[k].sum()-us[k].sum()/uc[k].sum())
  print('diff G-U ex oos',o,gs.sum()/gc.sum()-us.sum()/uc.sum(),np.percentile(ds,[2.5,97.5]))
# regression
import numpy.linalg as la
for nm,x in [('G',G),('G is',G[G.oos2026==0]),('G oos',G[G.oos2026==1]),('U',U),('all',L)]:
  X=np.c_[np.ones(len(x)),x.b5];b=la.lstsq(X,x.r5,rcond=None)[0]
  print(nm,'alpha',b[0],'beta',b[1],'R2',1-((x.r5-X@b)**2).sum()/((x.r5-x.r5.mean())**2).sum(), 'mean b5',x.b5.mean(),'mean r5',x.r5.mean())
# sector
print(G.groupby('sector').agg(n=('r5','size'),up=('up','mean'),ex=('ex','mean')).sort_values('n',ascending=False).head(12))
# grade-level sig oos
for g in ['SSS','A','B']:
  x=G[(G.oos2026==1)&(G.grade==g)];print(g,'oos',boot(x,'ex'))
# null: random-label combo selection. Use conds sets: count unique combos
print('unique combos in G',G.combo.nunique(), 'wr5 vs actual per combo corr');
c=G[G.oos2026==0].groupby('combo').agg(n=('r5','size'),cl=('combo_wr5','mean'),up=('up','mean'));c=c[c.n>=8];print(len(c),c.corr().round(2).to_string())
# null selection: 1626 random combos each hitting random 12% of base-signal pool, p(up)=base rate
IS=d[d.oos2026==0];p=IS.up.mean();print('base up rate IS all',p,'n',len(IS))
N=1626;n=40;mx=[];sel=[]
for _ in range(20):
  w=rng.binomial(n,p,N)/n; top=w[w>=np.quantile(w,0.9)];mx.append(w.max());sel.append(top.mean())
print('null n=40/combo: max wr',np.mean(mx),'top10% mean',np.mean(sel))
for n in [20,40,80]:
  w=rng.binomial(n,p,N)/n;print(n,'top1%',np.quantile(w,.99),'max',w.max())
