"""2000–2025 年"稳定赚钱"的组合有多少：
在 T+1、T+3、T+5、T+10 四个持有期，平均收益和中位收益都 > 0.1%（从信号日 T 当天收盘价算，不比 SPY）。
信号池 = 基础信号（不看 VIX，同票 5 天去重，有开盘价），和主推规则同一口径。
组合 = 条件库（big_conditions.py 的全部条件 + 「低开」）里任取 1/2/3 个；训练期至少 30 笔。
训练 = 信号日 < 2025-12-01；另外看这些组合在 2026 年的表现。
"""
import os, itertools, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'big_conditions.py'), encoding='utf-8').read().split('\n')
cut = next(i for i, l in enumerate(src) if l.startswith('# ---------- 收益'))
code = '\n'.join(src[:cut]).replace(
    "s = pd.read_parquet(os.path.join(HERE, 'jb_signals_grid.parquet'))",
    "s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))\n"
    "s = s[s.has_open].sort_values(['ticker', 'signal_day']).reset_index(drop=True)\n"
    "_k, _l = [], {}\n"
    "for _t, _d in zip(s.ticker, s.signal_day):\n"
    "    _ok = _t not in _l or (_d - _l[_t]).days > 5; _k.append(_ok)\n"
    "    if _ok: _l[_t] = _d\n"
    "s = s[np.array(_k)].reset_index(drop=True)")
exec(code)
add('低开(开<昨收)', O < pc)

# 收益：从 T 当天收盘算，持有 n 个交易日（该股自己的交易日）
H = [1, 3, 5, 10]
ret = {n: np.full(len(s), np.nan) for n in H}
for t, g in s.groupby('ticker'):
    c = C[t].dropna(); pos = c.index.get_indexer(g.signal_day); cv = c.values
    for n in H:
        ok = (pos >= 0) & (pos + n < len(c))
        v = np.full(len(g), np.nan); v[ok] = (cv[pos[ok] + n] / cv[pos[ok]] - 1) * 100
        ret[n][g.index.values] = v
Rm = np.column_stack([ret[n] for n in H])                     # n × 4

names = list(F); X = np.column_stack([F[k] for k in names]).astype(np.float32)
tr = (s.signal_day < pd.Timestamp('2025-12-01')).values & ~np.isnan(Rm).any(1)
te = (s.year == 2026).values & ~np.isnan(Rm).any(1)
prev = X[tr].mean(0); keepc = (prev > 0.003) & (prev < 0.99) & (X[tr].sum(0) >= 30)
names = [k for k, ok in zip(names, keepc) if ok]; X = X[:, keepc]; NC = len(names)
Xt, Xe, Rt, Re = X[tr], X[te], Rm[tr], Rm[te]
print(f'条件 {NC} 个；训练信号 {tr.sum()}，2026 信号 {te.sum()}', flush=True)
for lab, R_ in (('训练全部基础信号', Rt), ('2026 全部基础信号', Re)):
    print(lab + '：' + '，'.join(f'T+{n} 均{R_[:, j].mean():+.2f}% 中{np.median(R_[:, j]):+.2f}%' for j, n in enumerate(H)))

# 第一步：矩阵乘法算笔数和平均，筛"四个持有期平均都 >0.1%"
A = np.vstack([np.ones(len(Rt))] + [Rt[:, j] for j in range(4)]).astype(np.float32)
keys, st = [], []
for a in range(NC): keys.append((a,)); st.append(A @ Xt[:, a:a + 1])
for a in range(NC - 1):
    keys += [(a, b) for b in range(a + 1, NC)]; st.append(A @ (Xt[:, a:a + 1] * Xt[:, a + 1:]))
for a, b in itertools.combinations(range(NC), 2):
    m2 = Xt[:, a] * Xt[:, b]
    if m2.sum() < 30 or b + 1 >= NC: continue
    keys += [(a, b, c) for c in range(b + 1, NC)]; st.append(A @ (Xt[:, b + 1:] * m2[:, None]))
st = np.hstack(st).astype(np.float64); n = st[0]; mean = st[1:] / np.maximum(n, 1)
ok_n = n >= 30
ok_mean = ok_n & (mean > 0.1).all(0)
print(f'组合总数 {len(keys):,}；≥30 笔 {int(ok_n.sum()):,}；四个持有期平均都 >0.1%：{int(ok_mean.sum()):,}', flush=True)

# 第二步：对通过的组合算中位数（分块）
cand = np.where(ok_mean)[0]; med = np.full((len(cand), 4), np.nan); t0 = time.time()
def masks(idx, Xm):
    M = np.ones((Xm.shape[0], len(idx)), bool)
    for j, i in enumerate(idx):
        M[:, j] = np.all(Xm[:, list(keys[i])] > 0, axis=1)
    return M
for b0 in range(0, len(cand), 3000):
    idx = cand[b0:b0 + 3000]; M = masks(idx, Xt)
    for j in range(4):
        V = np.where(M, Rt[:, j:j + 1], np.nan); med[b0:b0 + len(idx), j] = np.nanmedian(V, axis=0)
    if b0 % 30000 == 0: print(f'  中位数 {b0:,}/{len(cand):,} {time.time()-t0:.0f}s', flush=True)
ok_med = (med > 0.1).all(1)
win = cand[ok_med]
print(f'\n✅ 2000–2025 年四个持有期 平均和中位数都 >0.1% 的组合：{len(win):,} 个（占 ≥30 笔组合的 {len(win)/ok_n.sum()*100:.1f}%）')
for k in (1, 2, 3):
    print(f'   其中 {k} 个条件：{sum(len(keys[i]) == k for i in win):,}')

# 第三步：这些组合在 2026 年
M26 = masks(win, Xe); n26 = M26.sum(0)
res = []
for j, n_ in enumerate(H):
    V = np.where(M26, Re[:, j:j + 1], np.nan)
    res.append((np.nanmean(V, 0), np.nanmedian(V, 0)))
mean26 = np.vstack([r[0] for r in res]); med26 = np.vstack([r[1] for r in res])
for mn in (5, 10, 20):
    h = n26 >= mn; both = h & (mean26 > 0.1).all(0) & (med26 > 0.1).all(0)
    print(f'2026 年 ≥{mn} 笔的 {int(h.sum()):,} 个 → 2026 年四个持有期平均和中位数也都 >0.1%：{int(both.sum()):,}（{both.sum()/max(h.sum(),1)*100:.0f}%）')
# 对照：训练没通过的组合，2026 年同样标准的通过率
rest = np.where(ok_n & ~np.isin(np.arange(len(keys)), win))[0]
rng = np.random.default_rng(0); smp = rng.choice(rest, min(len(rest), 30000), replace=False)
Mr = masks(smp, Xe); nr = Mr.sum(0)
mr = np.vstack([np.nanmean(np.where(Mr, Re[:, j:j + 1], np.nan), 0) for j in range(4)])
dr = np.vstack([np.nanmedian(np.where(Mr, Re[:, j:j + 1], np.nan), 0) for j in range(4)])
h = nr >= 10
print(f'对照：训练没通过的组合（抽 {len(smp):,} 个，2026 ≥10 笔 {int(h.sum()):,} 个）→ 2026 年同样通过：{((mr > 0.1).all(0) & (dr > 0.1).all(0))[h].mean()*100:.0f}%')

out = pd.DataFrame({'组合': ['＋'.join(names[k] for k in keys[i]) for i in win], '条件数': [len(keys[i]) for i in win],
                    '训练笔数': n[win].astype(int)})
for j, n_ in enumerate(H):
    out[f'训练T+{n_}均%'] = mean[j, win].round(2); out[f'训练T+{n_}中%'] = med[ok_med][:, j].round(2)
out['2026笔数'] = n26
for j, n_ in enumerate(H):
    out[f'2026 T+{n_}均%'] = mean26[j].round(2); out[f'2026 T+{n_}中%'] = med26[j].round(2)
out.sort_values('训练笔数', ascending=False).to_csv(os.path.join(HERE, 'stable_combos.csv'), index=False, encoding='utf-8-sig')
print('已存 stable_combos.csv')
