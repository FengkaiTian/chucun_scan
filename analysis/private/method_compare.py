"""几种选股方法在 2026 年的对比（全部只用 2000–2025 年训练）。
信号池 = 基础信号（不看 VIX，同票 5 天去重，有开盘价），和主推同一口径。
评估 = T1 收盘买、持有 10 个交易日：每笔净收益（扣 0.10%）、胜率（涨超 0.1% 算赢）；另报 5 天。
方法：
  0 全部基础信号
  1 主推（低开）；1a 主推 A+B
  2 单个最好组合（按训练期 10 天净收益 t 值，笔数够多）
  3 简单投票：命中多少个"训练期稳定"组合（stable_combos.csv：T+1/3/5/10 平均和中位数都 >0.1%）
  4 加权投票：同上，每个组合按训练期成绩加权
  5 模型：Ridge 回归、梯度提升树，用全部条件预测 10 天收益
2–5 都把分数线定在训练期每年约 120 笔 / 250 笔，然后原样用到 2026。
"""
import os, itertools, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'stable_combos.py'), encoding='utf-8').read()
exec(src.split("# 收益：从 T 当天收盘算")[0])            # 得到 s、F（含「低开」）

names = list(F); X = np.column_stack([F[k] for k in names]).astype(np.float32)
s['net10'] = s.r10 - 0.10; s['net5'] = s.r5 - 0.10
ok = s.net10.notna().values
tr = (s.signal_day < pd.Timestamp('2025-12-01')).values & ok
te = (s.year == 2026).values & ok
NY = (pd.Timestamp('2025-12-01') - pd.Timestamp('2000-01-03')).days / 365.25
w = (0.5 ** ((2026 - (s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366)) / 10)).values
idx = {k: i for i, k in enumerate(names)}
rows = []

def report(name, m):
    out = {'方法': name}
    for lab, per in (('训练', tr), ('2026', te)):
        q = m & per
        out[f'{lab}每年' if lab == '训练' else '2026笔数'] = round(q.sum() / NY) if lab == '训练' else int(q.sum())
        e = q.sum() == 0
        out[f'{lab}10天净%'] = None if e else round(float(s.net10[q].mean()), 2)
        out[f'{lab}胜率%'] = None if e else round(float((s.r10[q] > 0.1).mean() * 100))
        out[f'{lab}5天净%'] = None if e else round(float(s.net5[q].mean()), 2)
    rows.append(out); print(out, flush=True)

def top_by_score(score, per_year):
    """训练期分数最高、每年约 per_year 笔的分数线，用到全部信号"""
    k = int(per_year * NY); thr = np.sort(score[tr])[::-1][min(k, tr.sum() - 1)]
    return score > thr

report('0 全部基础信号', np.ones(len(s), bool))
gap = X[:, idx['低开(开<昨收)']] > 0
report('1 主推（低开）', gap)
o = s.index.values
deep = (s.gap_down_close.values.astype(bool)) | (X[:, idx['开盘<昨日最低']] > 0) | (X[:, idx['跳空低开>2%']] > 0)
nday = s.assign(g=gap).groupby('signal_day').g.transform('sum').values
fear = (s.vix.values >= 25) | (nday >= 5)
report('1a 主推 A+B', gap & (deep | fear))

# 2 单个最好组合：全部 1–3 条件组合，按训练期时间加权 10 天净收益的 t 值
Xt = X[tr]; wt = w[tr].astype(np.float32); yt = s.net10.values[tr].astype(np.float32); NC = X.shape[1]
A = np.vstack([np.ones_like(wt), wt, wt * yt, wt * yt * yt, wt * wt])
keys, st = [], []
for a in range(NC): keys.append((a,)); st.append(A @ Xt[:, a:a + 1])
for a in range(NC - 1): keys += [(a, b) for b in range(a + 1, NC)]; st.append(A @ (Xt[:, a:a + 1] * Xt[:, a + 1:]))
for a, b in itertools.combinations(range(NC), 2):
    m2 = Xt[:, a] * Xt[:, b]
    if m2.sum() < 30 or b + 1 >= NC: continue
    keys += [(a, b, c) for c in range(b + 1, NC)]; st.append(A @ (Xt[:, b + 1:] * m2[:, None]))
st = np.hstack(st).astype(np.float64); n, W, Wy, Wyy, W2 = st
mu = Wy / np.maximum(W, 1e-12); var = np.clip(Wyy / np.maximum(W, 1e-12) - mu ** 2, 1e-9, None)
tt = mu / np.sqrt(var / np.maximum(W ** 2 / np.maximum(W2, 1e-12), 1))
for py in (120, 250):
    lo, hi = py * NY * 0.7, py * NY * 1.5
    cand = np.where((n >= lo) & (n <= hi))[0]; best = cand[np.argmax(tt[cand])]
    m = np.all(X[:, list(keys[best])] > 0, axis=1)
    report(f'2 单个最好组合 ~{py}/年：' + '＋'.join(names[k] for k in keys[best]), m)

# 3/4 投票
sc = pd.read_csv(os.path.join(HERE, 'stable_combos.csv'))
parts = sc.组合.str.split('＋')
ci = [[idx[z] for z in p] for p in parts]
qual = sc[[c for c in sc.columns if c.startswith('训练T+')]].mean(1).clip(lower=0).values   # 8 个数（4 期平均+中位）的平均
votes = np.zeros(len(s)); wvotes = np.zeros(len(s))
for b0 in range(0, len(ci), 5000):
    blk = ci[b0:b0 + 5000]; pad = np.array([c + [c[-1]] * (3 - len(c)) for c in blk])
    M = X[:, pad[:, 0]] * X[:, pad[:, 1]] * X[:, pad[:, 2]]
    votes += M.sum(1); wvotes += M @ qual[b0:b0 + 5000]
for py in (120, 250):
    report(f'3 简单投票 ~{py}/年', top_by_score(votes, py))
    report(f'4 加权投票 ~{py}/年', top_by_score(wvotes, py))

# 5 模型
y = np.clip(s.net10.values, -30, 30)
rd = Ridge(alpha=50.0).fit(X[tr], y[tr], sample_weight=w[tr]); prd = rd.predict(X)
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=80,
                                   l2_regularization=1.0, random_state=0).fit(X[tr], y[tr], sample_weight=w[tr])
pgb = gb.predict(X)
for py in (120, 250):
    report(f'5 Ridge 模型 ~{py}/年', top_by_score(prd, py))
    report(f'5 梯度提升树 ~{py}/年', top_by_score(pgb, py))
coef = pd.Series(rd.coef_, index=names).sort_values()
print('\nRidge 权重最大的 12 个：', coef.tail(12).round(2).to_dict())
print('Ridge 权重最小的 8 个：', coef.head(8).round(2).to_dict())

R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'method_compare.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print('\n' + R.to_string(index=False))
