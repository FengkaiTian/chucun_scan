"""全部 2/3/4 条件组合，2026 年（到 11-17）在 T+1/3/5/10 跑赢 SPY（含分红）的有多少。
入场 = 信号次日收盘；持有 n 个交易日（节假日休市不算交易日）；扣 0.10%；同进同出比 SPY。
"""
import os, sys, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
COND = list(ds.COND_LABEL); H = [1, 3, 5, 10]

s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[(s.vix < 30) & s.has_open].sort_values(['ticker', 'signal_day'])
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    ok = t not in last or (d - last[t]).days > 14; keep.append(ok)
    if ok: last[t] = d
s = s[np.array(keep)]
s = s[s.year == 2026].reset_index(drop=True)
spy = pd.read_parquet(os.path.join(ROOT, 'analysis', 'rebuild', 'data2000', 'index_spy_tr.parquet')).iloc[:, 0].loc[:'2026-11-17']
for n in H:
    s[f'xx{n}'] = s[f'r{n}'] - 0.10 - ((spy.shift(-n) / spy - 1) * 100).reindex(s.entry_day).values
C = s[COND].values.astype(float)
X = {n: np.nan_to_num(s[f'xx{n}'].values) for n in H}
V = {n: (~np.isnan(s[f'xx{n}'].values)).astype(float) for n in H}
A = np.vstack([V[n] for n in H] + [X[n] for n in H])           # 8 × n

keys, st = [], []
for k in (2, 3, 4):
    for comb in itertools.combinations(range(37), k):
        keys.append(comb)
idx = np.array([list(c) + [-1] * (4 - len(c)) for c in keys])
CC = np.hstack([C, np.ones((len(C), 1))])                      # -1 → 全 1 列
for i in range(0, len(keys), 5000):
    b = idx[i:i + 5000]
    M = CC[:, b[:, 0]] * CC[:, b[:, 1]] * CC[:, b[:, 2]] * CC[:, b[:, 3]]
    st.append(A @ M)
st = np.hstack(st)
r = pd.DataFrame({'组合': ['＋'.join(ds.COND_LABEL[COND[j]] for j in c) for c in keys]})
for j, n in enumerate(H):
    r[f'T+{n}笔数'] = st[j].astype(int)
    r[f'T+{n}超额%'] = np.where(st[j] > 0, st[4 + j] / np.maximum(st[j], 1), np.nan).round(2)
r.to_csv(os.path.join(HERE, 'combos_2026_all.csv'), index=False, encoding='utf-8-sig')

print(f'组合总数：{len(r):,}（2 条件 {sum(len(c)==2 for c in keys):,}，3 条件 {sum(len(c)==3 for c in keys):,}，4 条件 {sum(len(c)==4 for c in keys):,}）')
has = r['T+1笔数'] > 0
print(f'2026 年至少出过 1 次信号的组合：{has.sum():,}；2026 年基础信号 {len(s)} 笔')
for mn in (1, 5, 10):
    q = r[r['T+10笔数'] >= mn]
    line = [f'2026 年 ≥{mn} 笔的组合 {len(q):,} 个，跑赢 SPY 的：']
    for n in H: line.append(f'T+{n} {int((q[f"T+{n}超额%"] > 0).sum()):,}（{(q[f"T+{n}超额%"] > 0).mean()*100:.0f}%）')
    allw = (q[[f'T+{n}超额%' for n in H]] > 0).all(axis=1)
    line.append(f'｜四个都赢 {int(allw.sum()):,}（{allw.mean()*100:.0f}%）')
    print(' '.join(line))
base = {n: np.nanmean(s[f'xx{n}']) for n in H}
print('参照：2026 全部基础信号超额 ' + '，'.join(f'T+{n} {base[n]:+.2f}%' for n in H))
q = r[(r['T+10笔数'] >= 10) & (r[[f'T+{n}超额%' for n in H]] > 0).all(axis=1)].sort_values('T+5超额%', ascending=False)
q.to_csv(os.path.join(HERE, 'combos_2026_win_all4_n10.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(f'\n≥10 笔且 T+1/3/5/10 全部跑赢的组合 {len(q)} 个，按 T+5 超额排前 40：')
print(q.head(40).to_string(index=False))
