"""训练 2000–2025（时间加权，半衰期 10 年）选规则，2026 年测试。
候选：A) 已有规则（SSS、★★★、★★、★）；B) 只用训练集重新挑的单条件和两两组合（干净）。
选中标准：加权每笔超额 > 0；单条件 t > 2，两两组合 t > 3（候选多，门槛更高）；笔数 ≥ 50。
"""
import os, sys, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
HL = 10.0; COND = list(ds.COND_LABEL)

s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[(s.vix < 30) & s.x5.notna() & s.has_open].sort_values(['ticker', 'signal_day'])
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    ok = t not in last or (d - last[t]).days > 14; keep.append(ok)
    if ok: last[t] = d
s = s[np.array(keep)].reset_index(drop=True)
s['net5'] = s.r5 - 0.10
tr = (s.signal_day < pd.Timestamp('2025-12-20')).values           # 去掉 2025 年最后 10 天，5 日收益不跨进 2026
te = (s.year == 2026).values
ty = (s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366).values
w = 0.5 ** ((2026 - ty) / HL)
X, G, N = s.x5.values, s.signal_day.values, s.net5.values


def ws(m):
    ww = w[m]; x = X[m]; W = ww.sum(); mu = (ww * x).sum() / W
    e = pd.Series(ww * (x - mu)).groupby(G[m]).sum().values
    return mu, mu / (np.sqrt((e ** 2).sum()) / W)


def ct(m):
    x = X[m]; mu = x.mean(); e = pd.Series(x - mu).groupby(G[m]).sum().values
    return mu / (np.sqrt((e ** 2).sum()) / len(x))


cands = {'SSS': s.sss.values, '★★★': (s.tier == '★★★').values, '★★': (s.tier == '★★').values, '★': (s.tier == '★').values}
for c in COND: cands[c] = s[c].values
for a, b in itertools.combinations(COND, 2): cands[f'{a}+{b}'] = s[a].values & s[b].values

rows = []
for k, m in cands.items():
    mt = m & tr
    if mt.sum() < 50: continue
    mu, t = ws(mt)
    need = 3 if '+' in k else 2
    mte = m & te
    rows.append({'规则': k, '训练笔数': int(mt.sum()), '训练加权超额': round(mu, 2), '训练加权t': round(t, 1),
                 '选中': mu > 0 and t > need, '测试笔数': int(mte.sum()),
                 '测试胜率%': round((N[mte] > 0).mean() * 100) if mte.sum() else np.nan,
                 '测试超额': round(X[mte].mean(), 2) if mte.sum() else np.nan})
r = pd.DataFrame(rows).sort_values('训练加权t', ascending=False)
sel = r[r.选中]
pd.set_option('display.width', 250)
print(f'候选 {len(r)} 条（笔数≥50），训练集选中 {len(sel)} 条：'); print(sel.to_string(index=False))

base = te
print(f'\n2026 测试：全部基础信号 {base.sum()} 笔，胜率 {(N[base]>0).mean()*100:.0f}%，每笔超额 {X[base].mean():+.2f}%（t {ct(base):.1f}）')
for lab, ks in [('已有规则里选中的', [k for k in sel.规则 if k in ('SSS', '★★★', '★★', '★')]),
                ('训练集重新挑的（单条件+组合）', [k for k in sel.规则 if k not in ('SSS', '★★★', '★★', '★')]),
                ('全部选中的合并', list(sel.规则))]:
    if not ks: print(f'{lab}：无'); continue
    m = np.zeros(len(s), bool)
    for k in ks: m |= cands[k]
    m &= te
    print(f'{lab}（{len(ks)} 条）→ 2026 年 {m.sum()} 笔，胜率 {(N[m]>0).mean()*100:.0f}%，每笔净收益 {N[m].mean():+.2f}%，每笔超额 {X[m].mean():+.2f}%（t {ct(m):.1f}）')
r.to_csv(os.path.join(HERE, 'split_2026_all.csv'), index=False)
