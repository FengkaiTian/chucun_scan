import os, numpy as np, pandas as pd
pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 70)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out')
d = pd.read_parquet(os.path.join(OUT, 'agentB.parquet'))
print('总数', len(d), d.family.value_counts().to_dict())
ok = (d.train_n >= 100) & (d.valid_n >= 30)
d['score'] = np.where(ok, d[['train_t', 'valid_t']].min(1), np.nan)
d['score_ex'] = np.where(ok, d[['train_t_excess', 'valid_t_excess']].min(1), np.nan)
r = d[ok].sort_values('score', ascending=False).reset_index(drop=True)
r['rank'] = np.arange(1, len(r) + 1)
print('合格', len(r))
c = ['rank', 'name', 'train_n', 'train_win', 'train_avg', 'train_t', 'valid_n', 'valid_win', 'valid_avg', 'valid_t', 'train_excess', 'valid_excess', 'score_ex']
print(r[c].head(30).round(2).to_string())
rnd = r[r.family == '随机对照']
print('随机对照最好排名', rnd['rank'].min(), '共', len(rnd), '最好score', rnd.score.max().round(2), '随机score 95分位', rnd.score.quantile(.95).round(2))
print(rnd[c].head(5).round(2).to_string())
print('score 超过随机最好的策略数', (r.score > rnd.score.max()).sum())
# 市场状态贡献：以 (base, hold) 为单位，和不加状态的基准比
b0 = d[d.n_states == 0].set_index(['base', 'hold'])
for k in (1, 2):
    x = d[(d.n_states == k) & ok].copy()
    j = x.join(b0[['train_t', 'valid_t', 'train_avg', 'valid_avg', 'score']], on=['base', 'hold'], rsuffix='_0')
    j['d_tavg'] = j.train_avg - j.train_avg_0; j['d_vavg'] = j.valid_avg - j.valid_avg_0
    both = (j.d_tavg > 0) & (j.d_vavg > 0)
    print(f'\n{k}个状态: n={len(j)}  训练平均收益提高比例 {np.mean(j.d_tavg>0):.2f}  验证提高 {np.mean(j.d_vavg>0):.2f}  两期都提高 {both.mean():.2f}')
    print('  训练提高者中验证也提高:', round(((j.d_vavg > 0) & (j.d_tavg > 0)).sum() / max((j.d_tavg > 0).sum(), 1), 3))
    print('  score 比基准高的比例', np.mean(j.score > j.score_0).round(3), ' 中位 Δscore', (j.score - j.score_0).median().round(2))
    # 训练里挑最好的状态，看验证表现（诚实的选择检验）
    best = j.sort_values('train_t', ascending=False).groupby(['base', 'hold']).head(1)
    print('  每个(base,hold)按训练t挑最好状态 → 验证平均 Δavg', best.d_vavg.median().round(3), '验证改善比例', np.mean(best.d_vavg > 0).round(2))
# 状态平均效果（跨所有 base/hold，验证期 Δavg 中位数）
x = d[(d.n_states == 1) & ok].join(b0[['train_avg', 'valid_avg']], on=['base', 'hold'], rsuffix='_0')
x['dt'] = x.train_avg - x.train_avg_0; x['dv'] = x.valid_avg - x.valid_avg_0
g = x.groupby('state1')[['dt', 'dv']].median().sort_values('dt', ascending=False)
g['同号'] = np.sign(g.dt) == np.sign(g.dv)
print('\n状态效果(中位 Δavg) 前12/后6:'); print(g.head(12).round(3).to_string()); print(g.tail(6).round(3).to_string())
print('状态训练/验证 Δavg 相关', np.corrcoef(g.dt, g.dv)[0, 1].round(2), ' 同号比例', g['同号'].mean().round(2))
print('\n基础信号本身(无状态)按 score:')
print(r[r.n_states == 0][c].head(15).round(2).to_string())
u = r[r.base.str.contains('用户基础')]
print('\n用户基础信号最好:'); print(u[c].head(8).round(2).to_string())
