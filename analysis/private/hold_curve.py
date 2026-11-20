"""主推：T1 收盘买，持有 d=1..20 个交易日收盘卖，每个 d 的胜率（涨超 0.1%）和每笔净收益。训练 vs 2026，全部 vs A+B。"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
C = pd.read_parquet(os.path.join(HERE, '..', 'rebuild', 'data2000', 'gh_johnbrick', 'panel', 'C.parquet'))
s = pd.read_parquet(os.path.join(HERE, 'main_rule_signals.parquet'))
s['AB'] = ((s.gap <= -1) | s.open_lt_pl) | ((s.vix >= 25) | (s.nday >= 5))
D = 20; R = np.full((len(s), D), np.nan)
for t, g in s.groupby('ticker'):
    c = C[t].dropna().values; pos = C[t].dropna().index.get_indexer(g.signal_day) + 1
    for k, p in zip(g.index, pos):
        if p <= 0 or p >= len(c): continue
        for d in range(1, D + 1):
            if p + d < len(c): R[k, d - 1] = (c[p + d] / c[p] - 1) * 100
tr = (s.signal_day < '2025-11-01').values; te = (s.year == 2026).values
rows = []
for d in range(1, D + 1):
    row = {'持有天': d}
    for lab, m in (('训练全部', tr), ('训练A+B', tr & s.AB.values), ('2026全部', te), ('2026A+B', te & s.AB.values)):
        v = R[m, d - 1]; v = v[~np.isnan(v)]
        row[f'{lab}胜率%'] = round((v > 0.1).mean() * 100, 1); row[f'{lab}净%'] = round(v.mean() - 0.1, 2)
    rows.append(row)
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, 'hold_curve.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); print(T.to_string(index=False))
