import os, pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out', 'agentA.parquet')
d = pd.read_parquet(p)
print('rows', len(d), 'random', (d.family == '随机对照').sum(), '样本不足', d['样本不足'].sum())
ok = d[~d['样本不足']].copy()
ok['score'] = np.minimum(ok.train_t, ok.valid_t)
ok = ok.sort_values('score', ascending=False).reset_index(drop=True)
ok['rank'] = ok.index + 1
cols = ['rank', 'name', 'score', 'train_n', 'train_win', 'train_avg', 'train_per_year', 'valid_n', 'valid_win', 'valid_avg', 'valid_per_year', 'train_excess', 'valid_excess']
print(ok[cols].head(30).round(2).to_string(index=False))
r = ok[ok.family == '随机对照']
print('随机对照最好:'); print(r[cols].head(5).round(2).to_string(index=False))
print('随机对照 score 分布', r.score.describe().round(2).to_dict())
print('有效策略数', len(ok), '超过最好随机的数量', (ok.score > r.score.max()).sum())
print('score>2', (ok.score > 2).sum(), 'score>3', (ok.score > 3).sum())
print(ok[ok.family != '随机对照'].groupby('hold').score.describe().round(2))
print('按条件数', ok[ok.family != '随机对照'].groupby('n_conds').score.describe().round(2))
# 超额t
ok['score_ex'] = np.minimum(ok.train_t_excess, ok.valid_t_excess)
print(ok.sort_values('score_ex', ascending=False)[['name', 'score_ex', 'train_excess', 'valid_excess', 'train_n', 'valid_n']].head(10).round(2).to_string(index=False))
print('随机 score_ex 最大', ok[ok.family == '随机对照'].score_ex.max().round(2))
