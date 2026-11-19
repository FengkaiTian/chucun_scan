"""334 个组合逐个检验：训练 2000–2025（时间加权，半衰期 10 年），测试 2026。
按训练成绩给每个组合重新定星级，一个信号取它命中组合里的最高星级，再看 2026 年各档表现。
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
from math import erf, sqrt
HL = 10.0

s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[(s.vix < 30) & s.x5.notna() & s.has_open].sort_values(['ticker', 'signal_day'])
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    ok = t not in last or (d - last[t]).days > 14; keep.append(ok)
    if ok: last[t] = d
s = s[np.array(keep)].reset_index(drop=True)
X, G, N = s.x5.values, s.signal_day.values, (s.r5 - 0.10).values
tr = (s.signal_day < pd.Timestamp('2025-12-20')).values; te = (s.year == 2026).values
ty = (s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366).values
w = 0.5 ** ((2026 - ty) / HL)


def ws(m):
    ww = w[m]; x = X[m]; W = ww.sum(); mu = (ww * x).sum() / W
    e = pd.Series(ww * (x - mu)).groupby(G[m]).sum().values
    se = np.sqrt((e ** 2).sum()) / W
    return mu, (mu / se if se > 0 else 0), (ww * (N[m] > 0)).sum() / W


combos = json.load(open(os.path.join(ROOT, 'validated_combos.json'), encoding='utf-8'))
rows, masks = [], []
for i, c in enumerate(combos):
    m = s[c['conditions']].all(axis=1).values
    masks.append(m)
    mt = m & tr
    mu, t, wr = ws(mt) if mt.sum() >= 5 else (np.nan, np.nan, np.nan)
    rows.append({'i': i, '原等级': c['grade'], '组合': '＋'.join(ds.COND_LABEL[k] for k in c['conditions']),
                 '训练笔数': int(mt.sum()), '训练胜率': wr, '训练超额': mu, '训练t': t, '测试笔数': int((m & te).sum())})
r = pd.DataFrame(rows)


def star(row):
    if row.训练笔数 < 30 or not (row.训练超额 > 0): return '参考'
    if row.训练t >= 3: return '★★★'
    if row.训练t >= 2: return '★★'
    if row.训练t >= 1: return '★'
    return '参考'


r['新星级'] = r.apply(star, axis=1)
print('334 个组合按训练成绩重新定级（笔数≥30 才定级）：')
print(pd.crosstab(r.原等级, r.新星级, margins=True).to_string())
# 纯靠运气大约会有多少个 t≥2 / t≥3（正态近似，单边）
ok = r[r.训练笔数 >= 30]
p2, p3 = 1 - 0.5 * (1 + erf(2 / sqrt(2))), 1 - 0.5 * (1 + erf(3 / sqrt(2)))
print(f'\n可定级的组合 {len(ok)} 个；纯靠运气预计 t≥2 约 {len(ok)*p2:.0f} 个、t≥3 约 {len(ok)*p3:.1f} 个；实际 t≥2 {int((ok.训练t>=2).sum())} 个、t≥3 {int((ok.训练t>=3).sum())} 个')
print(f'笔数不足 30、没法定级的组合：{int((r.训练笔数 < 30).sum())} 个')

# 信号定档：取命中组合里的最高新星级
order = {'★★★': 0, '★★': 1, '★': 2, '参考': 3}
best = np.full(len(s), 3)
for m, st in zip(masks, r.新星级): best = np.where(m, np.minimum(best, order[st]), best)
s['新档'] = pd.Series(best).map({v: k for k, v in order.items()}).values


def ct(m):
    x = X[m]; mu = x.mean(); e = pd.Series(x - mu).groupby(G[m]).sum().values
    return mu / (np.sqrt((e ** 2).sum()) / len(x))


print('\n每个信号取命中组合的最高新星级：')
for lab, msk in [('训练 2000–2025', tr), ('测试 2026', te)]:
    for k in order:
        m = msk & (s.新档 == k).values
        if m.sum() < 5: print(f'  {lab} {k}: {m.sum()} 笔'); continue
        yrs = 26 if lab.startswith('训练') else 320 / 365
        print(f'  {lab} {k:3s}: {m.sum():5d} 笔（每年约 {m.sum()/yrs:.0f}），胜率 {(N[m]>0).mean()*100:.0f}%，每笔净 {N[m].mean():+.2f}%，超额 {X[m].mean():+.2f}%（t {ct(m):.1f}）')
pd.set_option('display.width', 250)
print('\n训练集最好的 15 个组合：')
print(r.sort_values('训练t', ascending=False).head(15)[['原等级', '组合', '训练笔数', '训练胜率', '训练超额', '训练t', '测试笔数', '新星级']].round(2).to_string(index=False))
r.to_csv(os.path.join(HERE, 'combo_2026.csv'), index=False)
