import os, sys, json
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); pd.set_option('display.max_rows', 200)
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, '..', 'out', 'agentC.parquet'))
print('rows', len(df)); print(df.family.value_counts())
ok = (df.train_n >= 200) & (df.valid_n >= 50)
df['score'] = np.where(ok, np.minimum(df.train_t, df.valid_t), np.nan)
df = df.sort_values('score', ascending=False).reset_index(drop=True)
df['rank'] = np.arange(1, len(df) + 1)
cols = ['rank', 'name', 'score', 'train_n', 'train_avg', 'train_win', 'train_hold', 'train_per_day', 'train_t', 'valid_n', 'valid_avg', 'valid_win', 'valid_hold', 'valid_per_day', 'valid_t']
print(df[cols].head(30).round(3).to_string())
rnd = df[df.family == '随机对照']
print('随机对照最好名次', rnd['rank'].min(), 'of', ok.sum(), '有效'); print(rnd[cols].head(5).round(3).to_string())
e1 = df[df.name.str.startswith('E01 ')]
print('E01 top'); print(e1[cols].head(15).round(3).to_string())
print(e1[e1.name.str.endswith('| 固定5天')][cols].round(3).to_string())
print('E01 按 per_day(train) top'); print(e1.sort_values('train_per_day', ascending=False)[cols].head(8).round(3).to_string())
# 每类出场的平均名次
df['cat'] = df.params.map(lambda s: json.loads(s).get('cat', 'grid') if s.startswith('{') else '')
print(df[ok.reindex(df.index, fill_value=False) if False else df.score.notna()].groupby('family').score.agg(['count', 'median', 'max']).round(2))
