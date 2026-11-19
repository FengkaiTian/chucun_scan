"""滚动 train/test + 轻微时间加权（半衰期 HL 年）。
每个测试年 Y：只用 Y 之前的信号（去掉 Y 前 10 天，防止 5 日收益跨进测试年）选规则，
选中标准：时间加权的每笔超额 > 0 且 加权 t > 2；测试年里命中任一选中规则的信号 = 当年的买入名单。
候选规则：37 个单条件 + SSS + ★★★/★★/★ 星级。只用有开盘价的信号。
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
HL = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
COND = list(ds.COND_LABEL)

s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[(s.vix < 30) & s.x5.notna() & s.has_open].sort_values(['ticker', 'signal_day'])
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    ok = t not in last or (d - last[t]).days > 14; keep.append(ok)
    if ok: last[t] = d
s = s[np.array(keep)].reset_index(drop=True)
s['t_year'] = s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366
s['net5'] = s.r5 - 0.10
rules = {c: s[c].values for c in COND}
rules['SSS'] = s.sss.values
for t in ['★★★', '★★', '★']: rules[t] = (s.tier == t).values


def wstat(x, w, g):
    """加权均值和按信号日聚类的加权 t。"""
    W = w.sum(); m = (w * x).sum() / W
    e = pd.Series(w * (x - m)).groupby(g).sum().values
    se = np.sqrt((e ** 2).sum()) / W
    return m, (m / se if se > 0 else 0)


out, picks = [], {}
for Y in range(2008, 2027):
    tr = (s.signal_day < pd.Timestamp(f'{Y}-01-01') - pd.Timedelta(days=10)).values
    te = (s.year == Y).values
    w = 0.5 ** ((Y - s.t_year.values[tr]) / HL)
    x, g = s.x5.values[tr], s.signal_day.values[tr]
    sel = []
    for k, m in rules.items():
        mm = m[tr]
        if mm.sum() < 30: continue
        mu, t = wstat(x[mm], w[mm], g[mm])
        if mu > 0 and t > 2: sel.append(k)
    picks[Y] = sel
    hit = np.zeros(len(s), bool)
    for k in sel: hit |= rules[k]
    a, b = s[te & hit], s[te]
    out.append({'测试年': Y, '选中规则': ','.join(sel) or '（无）', '买入笔数': len(a),
                '胜率%': round(((a.net5 > 0).mean()) * 100, 0) if len(a) else np.nan,
                '每笔超额': round(a.x5.mean(), 2) if len(a) else np.nan,
                '全部基础信号超额': round(b.x5.mean(), 2)})
r = pd.DataFrame(out)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(f'半衰期 {HL} 年'); print(r.to_string(index=False))

# 汇总：所有测试年拼在一起
allhit = []
for Y in picks:
    hit = np.zeros(len(s), bool)
    for k in picks[Y]: hit |= rules[k]
    allhit.append(s[(s.year == Y).values & hit])
A = pd.concat(allhit)
def ct(x, g):
    m = x.mean(); e = pd.Series(x - m).groupby(g).sum().values; return m / (np.sqrt((e**2).sum()) / len(x))
for lab, a, b in [('2008–2026', 2008, 2026), ('2008–2019', 2008, 2019), ('2020–2026', 2020, 2026)]:
    q = A[(A.year >= a) & (A.year <= b)]; base = s[(s.year >= a) & (s.year <= b)]
    print(f'{lab} 样本外：选中 {len(q)} 笔，胜率 {(q.net5>0).mean()*100:.0f}%，每笔超额 {q.x5.mean():+.2f}%（t {ct(q.x5.values, q.signal_day.values):.1f}）'
          f'｜全部基础信号 {base.x5.mean():+.2f}%')

# 现在（用到 2026-11-17 的全部数据）会选什么，给 2027 用
w = 0.5 ** ((2026.88 - s.t_year.values) / HL); x, g = s.x5.values, s.signal_day.values
now = []
for k, m in rules.items():
    if m.sum() < 30: continue
    mu, t = wstat(x[m], w[m], g[m]); wr = (w[m] * (s.net5.values[m] > 0)).sum() / w[m].sum()
    now.append((k, int(m.sum()), round(mu, 2), round(t, 1), round(wr * 100, 0)))
now = pd.DataFrame(now, columns=['规则', '笔数', '加权超额', '加权t', '加权胜率%']).sort_values('加权t', ascending=False)
print('\n用全部数据（时间加权）排序，前 12：'); print(now.head(12).to_string(index=False))
r.to_csv(os.path.join(HERE, f'walk_HL{int(HL)}.csv'), index=False); now.to_csv(os.path.join(HERE, f'now_HL{int(HL)}.csv'), index=False)
