"""2000–2025 年：37 个条件的全部 2/3/4 组合（约 7.4 万个）逐个算成绩，排 top100；附 2026 年测试。
信号 = 基础信号（VIX<30，14 天去重，有开盘价），次日收盘入场持有 5 天，扣 0.10%，超额相对同期 SPY 含分红。
排序 = 时间加权（半衰期 10 年）超额的 t 值；至少 30 笔；信号集合完全相同的组合只留条件最少的。
"""
import os, sys, itertools, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
HL, MINN = 10.0, 30
COND = list(ds.COND_LABEL)

s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[(s.vix < 30) & s.x5.notna() & s.has_open].sort_values(['ticker', 'signal_day'])
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    ok = t not in last or (d - last[t]).days > 14; keep.append(ok)
    if ok: last[t] = d
s = s[np.array(keep)].reset_index(drop=True)
tr = (s.signal_day < pd.Timestamp('2025-12-20')).values
te = (s.year == 2026).values
S = s[tr].reset_index(drop=True)
C = S[COND].values.astype(np.float64)                          # n × 37
x = S.x5.values; win = ((S.r5 - 0.10) > 0).values.astype(float); net = (S.r5 - 0.10).values
ty = (S.signal_day.dt.year + S.signal_day.dt.dayofyear / 366).values
w = 0.5 ** ((2026 - ty) / HL)
A = np.vstack([np.ones_like(w), w, w * x, w * x * x, w * w, w * win, net])   # 7 × n
print('训练信号', len(S), '测试信号', te.sum())

res = []; t0 = time.time()
def add(keys, m):                         # m: n × k 的 0/1 矩阵，每列一个组合
    st = A @ m                            # 7 × k
    for j in range(m.shape[1]):
        res.append((keys[j], *st[:, j]))

for a, b in itertools.combinations(range(37), 2):              # 2 条件
    add([(a, b)], (C[:, a] * C[:, b])[:, None])
for tri in itertools.combinations(range(37), 3):               # 3 条件 + 4 条件
    a, b, c = tri
    m3 = C[:, a] * C[:, b] * C[:, c]
    if m3.sum() < MINN: continue
    add([tri], m3[:, None])
    ds_ = [d for d in range(c + 1, 37)]
    if ds_:
        add([tri + (d,) for d in ds_], C[:, ds_] * m3[:, None])
print('组合数', len(res), f'{time.time()-t0:.0f}s')

r = pd.DataFrame(res, columns=['k', 'n', 'W', 'Wx', 'Wxx', 'Ww2', 'Wwin', 'net'])
r = r[r.n >= MINN].copy()
r['mu'] = r.Wx / r.W; var = (r.Wxx / r.W - r.mu ** 2).clip(lower=1e-9); neff = r.W ** 2 / r.Ww2
r['t'] = r.mu / np.sqrt(var / neff); r['wr'] = r.Wwin / r.W * 100; r['net'] = r.net / r.n
r = r.sort_values('t', ascending=False)
print('≥30 笔的组合', len(r))

# 去重：信号集合完全一样的只留条件最少的；取 top 100
def mask_tr(k): return np.all(C[:, list(k)] > 0, axis=1)
seen, top = set(), []
for row in r.itertuples():
    m = mask_tr(row.k); h = hash(m.tobytes())
    if h in seen: continue
    seen.add(h); top.append(row)
    if len(top) == 100: break
T = pd.DataFrame(top)

# 正确的按信号日聚类 t + 2026 测试
G = S.signal_day.values
def ctw(m):
    ww = w[m]; xx = x[m]; W = ww.sum(); mu = (ww * xx).sum() / W
    e = pd.Series(ww * (xx - mu)).groupby(G[m]).sum().values
    return mu / (np.sqrt((e ** 2).sum()) / W)
Ste = s[te]
out = []
for i, row in enumerate(T.itertuples(), 1):
    m = mask_tr(row.k); mt = Ste[[COND[j] for j in row.k]].all(axis=1).values
    out.append({'排名': i, '组合': '＋'.join(ds.COND_LABEL[COND[j]] for j in row.k), '笔数': int(row.n),
                '加权胜率%': round(row.wr), '每笔净收益%': round(row.net, 2), '加权超额%': round(row.mu, 2),
                't': round(ctw(m), 1),
                '2026笔数': int(mt.sum()),
                '2026胜率%': round(((Ste.r5[mt] - 0.10) > 0).mean() * 100) if mt.sum() else None,
                '2026超额%': round(Ste.x5[mt].mean(), 2) if mt.sum() else None})
O = pd.DataFrame(out)
O.to_csv(os.path.join(HERE, 'top100_2000_2025.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 200)
print(O.to_string(index=False))
# 纯靠运气的参照：把超额在训练集里随机打乱后，最好的组合 t 能到多少
rng = np.random.default_rng(0)

# 1) 纯靠运气的参照：同一天内打乱超额（保留市场整体涨跌），重跑全部组合，看第 1 名和第 100 名的 t
def run_t(xx):
    A2 = np.vstack([w, w * xx, w * xx * xx, w * w])
    ts = []
    for a, b in itertools.combinations(range(37), 2):
        ts.append(A2 @ (C[:, a] * C[:, b])[:, None])
    for a, b, c in itertools.combinations(range(37), 3):
        m3 = C[:, a] * C[:, b] * C[:, c]
        if m3.sum() < MINN: continue
        ts.append(A2 @ m3[:, None])
        if c + 1 < 37: ts.append(A2 @ (C[:, c + 1:] * m3[:, None]))
    st = np.hstack(ts); n_ = None
    return st
cnt = []
for a, b in itertools.combinations(range(37), 2): cnt.append(np.array([(C[:, a] * C[:, b]).sum()]))
for a, b, c in itertools.combinations(range(37), 3):
    m3 = C[:, a] * C[:, b] * C[:, c]
    if m3.sum() < MINN: continue
    cnt.append(np.array([m3.sum()]))
    if c + 1 < 37: cnt.append((C[:, c + 1:] * m3[:, None]).sum(0))
cnt = np.concatenate(cnt); ok = cnt >= MINN
null = []
for k in range(10):
    xx = x.copy()
    for idx in pd.Series(np.arange(len(x))).groupby(S.signal_day.values).indices.values():
        xx[idx] = xx[rng.permutation(idx)]
    xx = xx[rng.permutation(len(xx))] if False else xx
    st = run_t(xx)[:, ok]
    mu = st[1] / st[0]; var = np.clip(st[2] / st[0] - mu ** 2, 1e-9, None); tt = mu / np.sqrt(var / (st[0] ** 2 / st[3]))
    tt = np.sort(tt)[::-1]; null.append((tt[0], tt[99]))
null = np.array(null)
real = r.t.values
print(f'\n真实数据：第1名 t={real[0]:.1f}，第100名 t={real[99]:.1f}')
print(f'同日打乱（纯运气，10 次）：第1名 t 平均 {null[:,0].mean():.1f}（最高 {null[:,0].max():.1f}），第100名 t 平均 {null[:,1].mean():.1f}')

# 2) top100 合起来在 2026 年
m26 = np.zeros(len(Ste), bool)
for row in T.itertuples(): m26 |= Ste[[COND[j] for j in row.k]].all(axis=1).values
q = Ste[m26]
print(f'top100 任一命中 → 2026 年 {len(q)} 笔，胜率 {((q.r5-0.1)>0).mean()*100:.0f}%，每笔超额 {q.x5.mean():+.2f}%｜2026 全部基础信号 {Ste.x5.mean():+.2f}%')
