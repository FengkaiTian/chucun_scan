"""买入 = T+1 开盘价；卖出 = T+n 收盘价（n=1,3,5,10 个交易日，节假日不算）。涨超 1% 算赢。
给每个信号加 q1/q3/q5/q10 列（%），存 jb_signals_open.parquet；再算 2026 年全部组合的胜率。
"""
import os, sys, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
COND = list(ds.COND_LABEL); H = [1, 3, 5, 10]
P = os.path.join(ROOT, 'analysis', 'rebuild', 'data2000', 'gh_johnbrick', 'panel')
O = pd.read_parquet(os.path.join(P, 'O.parquet')).loc[:'2026-11-17']
C = pd.read_parquet(os.path.join(P, 'C.parquet')).loc[:'2026-11-17']

s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[(s.vix < 30) & s.has_open].sort_values(['ticker', 'signal_day'])
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    ok = t not in last or (d - last[t]).days > 14; keep.append(ok)
    if ok: last[t] = d
s = s[np.array(keep)].reset_index(drop=True)
for n in H: s[f'q{n}'] = np.nan
s['entry_open'] = np.nan
for t, g in s.groupby('ticker'):
    c = C[t].dropna(); o = O[t].reindex(c.index)
    pos = c.index.get_indexer(g.signal_day)
    for n in H:
        ok = pos + n < len(c)
        e = np.full(len(g), np.nan); x = np.full(len(g), np.nan)
        e[ok] = o.values[pos[ok] + 1]; x[ok] = c.values[pos[ok] + n]
        s.loc[g.index, f'q{n}'] = (x / e - 1) * 100
    s.loc[g.index, 'entry_open'] = np.where(pos + 1 < len(c), o.values[np.minimum(pos + 1, len(c) - 1)], np.nan)
s.to_parquet(os.path.join(HERE, 'jb_signals_open.parquet'), index=False)

for lab, m in [('2000–2025', s.year <= 2025), ('2026', s.year == 2026)]:
    q = s[m]
    print(f'全部基础信号 {lab}（{len(q)} 笔）涨>1% 比例：' + '，'.join(f'T+{n} {(q[f"q{n}"] > 1).mean()*100:.0f}%' for n in H)
          + '｜平均涨幅：' + '，'.join(f'T+{n} {q[f"q{n}"].mean():+.2f}%' for n in H))

s26 = s[s.year == 2026].reset_index(drop=True)
X = s26[COND].values.astype(float)
A = np.vstack([(~s26[f'q{n}'].isna()).values.astype(float) for n in H] + [(s26[f'q{n}'] > 1).values.astype(float) for n in H])
keys = [c for k in (2, 3, 4) for c in itertools.combinations(range(37), k)]
idx = np.array([list(c) + [-1] * (4 - len(c)) for c in keys]); XX = np.hstack([X, np.ones((len(X), 1))])
st = np.hstack([A @ (XX[:, b[:, 0]] * XX[:, b[:, 1]] * XX[:, b[:, 2]] * XX[:, b[:, 3]]) for b in np.array_split(idx, 15)])
r = pd.DataFrame({'组合': ['＋'.join(ds.COND_LABEL[COND[j]] for j in c) for c in keys]})
for j, n in enumerate(H):
    r[f'T+{n}笔数'] = st[j].astype(int)
    r[f'T+{n}胜率%'] = np.where(st[j] > 0, st[4 + j] / np.maximum(st[j], 1) * 100, np.nan).round(0)
r.to_csv(os.path.join(HERE, 'combos_2026_open_win1pct.csv'), index=False, encoding='utf-8-sig')
q = r[r['T+10笔数'] >= 10]
print(f'\n2026 年 ≥10 笔的组合 {len(q):,} 个（共 {len(r):,} 个组合）')
for th in (50, 60, 70, 80):
    allw = (q[[f'T+{n}胜率%' for n in H]] >= th).all(axis=1)
    print(f'胜率≥{th}%：' + ' '.join(f'T+{n} {int((q[f"T+{n}胜率%"] >= th).sum()):,}' for n in H) + f'｜四个都≥{th}% {int(allw.sum()):,}')
t = pd.read_csv(os.path.join(HERE, 'top100_2000_2025.csv')); lab = {v: k for k, v in ds.COND_LABEL.items()}
m = np.zeros(len(s26), bool)
for c in t.组合: m |= s26[[lab[z] for z in c.split('＋')]].all(axis=1).values
print(f'训练集 top100 任一命中（2026 年 {m.sum()} 笔）涨>1%：' + '，'.join(f'T+{n} {(s26.loc[m, f"q{n}"] > 1).mean()*100:.0f}%' for n in H))
best = q[(q[[f'T+{n}胜率%' for n in H]] >= 60).all(axis=1)].sort_values('T+5胜率%', ascending=False)
best.to_csv(os.path.join(HERE, 'combos_2026_open_all4_60.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(f'\n≥10 笔、四个周期都≥60% 的 {len(best)} 个，前 20：'); print(best.head(20).to_string(index=False))
