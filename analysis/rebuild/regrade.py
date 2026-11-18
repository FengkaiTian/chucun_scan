"""逐个检查 validated_combos.json 的 334 个组合，在挑选时没见过的年份里还管不管用，给出 删除/降级/保留/升级 建议。

样本外两段：2000–2019（组合是 2020–2025 挑的，这段没见过）、2025–2026（挑完之后；注意 2025 年部分可能在挑选期内）。
对比：命中该组合的信号 vs 同期没命中它的基础信号。指标：T+5 上涨比例差、当天内超额差（x5 减同一天全部信号平均）。
噪音对照：把收益按信号日整块打乱 500 次，看纯运气下有多少个组合会被判为"保留"。
输出：out2000/regrade.csv、out2000/regrade.txt、validated_combos_v2.json（只是建议，不接入扫描）。
"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
s = pd.read_parquet(os.path.join(HERE, 'data2000', 'signals_20y.parquet'))
d = s[s.in_index & (s.vix < 30) & s.r5.notna()].sort_values('signal_day').reset_index(drop=True)
d['xdm'] = d.x5 - d.groupby('signal_day').x5.transform('mean')
yr = d.signal_day.dt.year.values
P = {'A': yr <= 2019, 'IN': (yr >= 2020) & (yr <= 2024), 'B': yr >= 2025}
vc = json.load(open(os.path.join(ROOT, 'validated_combos.json'), encoding='utf-8'))
M = np.array([np.logical_and.reduce([d[k].fillna(False).astype(bool).values for k in c['conditions']]) for c in vc])
win = (d.r5.values > 0).astype(float)
xdm = d.xdm.values
day = pd.factorize(d.signal_day)[0]

lines = []
def say(*a):
    t = ' '.join(str(x) for x in a); print(t, flush=True); lines.append(t)


def stats(Mp, y):
    """每个组合：命中数、命中均值 − 没命中均值"""
    Mp = Mp.astype(np.float64)
    n = Mp.sum(1); tot = y.sum(); N = len(y)
    s1 = Mp @ y
    m1 = np.where(n > 0, s1 / np.maximum(n, 1), np.nan)
    m0 = np.where(N - n > 0, (tot - s1) / np.maximum(N - n, 1), np.nan)
    return n, m1 - m0


def classify(nA, wA, xA, nB, wB, xB):
    """
    升级：2000–2019 胜率差 ≥ +5 点 且 当天内超额 > 0 且 n≥40，2025–26 胜率差 > 0
    保留：2000–2019 胜率差 > 0 且 当天内超额 > 0，2025–26 胜率差 ≥ 0（或样本 <10）
    删除：两段胜率差都 < 0，或 2000–2019 胜率差 ≤ −5 点
    降级：其余
    """
    okB = (nB < 10) | (wB >= 0)
    up = (nA >= 40) & (wA >= 5) & (xA > 0) & (nB >= 10) & (wB > 0)
    keep = (nA >= 20) & (wA > 0) & (xA > 0) & okB
    dele = ((wA < 0) & ((nB >= 10) & (wB < 0))) | (wA <= -5)
    out = np.where(up, '升级', np.where(keep, '保留', np.where(dele, '删除', '降级')))
    return out


res = {}
for k, m in P.items():
    nw, dw = stats(M[:, m], win[m])
    _, dx = stats(M[:, m], xdm[m])
    res[k] = (nw, dw * 100, dx)
cls = classify(res['A'][0], res['A'][1], res['A'][2], res['B'][0], res['B'][1], res['B'][2])

T = pd.DataFrame({
    'grade': [c['grade'] for c in vc], 'conditions': [' + '.join(c['conditions']) for c in vc],
    'claimed_wr5': [c['wr5'] for c in vc],
    'n_00_19': res['A'][0], 'wr_diff_00_19': res['A'][1].round(1), 'xdm_diff_00_19': res['A'][2].round(2),
    'n_20_24': res['IN'][0], 'wr_diff_20_24': res['IN'][1].round(1),
    'n_25_26': res['B'][0], 'wr_diff_25_26': res['B'][1].round(1), 'xdm_diff_25_26': res['B'][2].round(2),
    'action': cls})
T.to_csv(os.path.join(HERE, 'out2000', 'regrade.csv'), index=False, encoding='utf-8-sig')

# 噪音对照：整块打乱信号日
rng = np.random.default_rng(1)
days_u = np.unique(day); by_day = [np.where(day == k)[0] for k in days_u]
cnt_null = {a: [] for a in ('升级', '保留', '删除', '降级')}
for p in range(500):
    perm = rng.permutation(len(days_u))
    idx = np.empty(len(day), np.int64)
    for a, b in enumerate(perm):
        dst, src = by_day[a], by_day[b]
        idx[dst] = src[np.arange(len(dst)) % len(src)]
    w2, x2 = win[idx], xdm[idx]
    rr = {}
    for k in ('A', 'B'):
        m = P[k]
        nw, dw = stats(M[:, m], w2[m]); _, dx = stats(M[:, m], x2[m])
        rr[k] = (nw, dw * 100, dx)
    c2 = classify(*rr['A'], *rr['B'])
    for a in cnt_null:
        cnt_null[a].append((c2 == a).sum())

say('# 334 个组合逐个复查')
say('样本外：2000–2019（挑选时没见过）和 2025–2026。比较的是命中 vs 同期没命中的基础信号。\n')
say('## 按等级')
for g in ('SSS', 'S', 'A', 'B'):
    gm = np.array([c['grade'] == g for c in vc])
    hit = M[gm].any(0)
    row = [f'{g}（{gm.sum()} 个）']
    for k, lab in (('A', '2000–19'), ('IN', '2020–24 挑选期'), ('B', '2025–26')):
        m = P[k]; h = hit[m]
        row.append(f'{lab}: n={h.sum()} 胜率 {win[m][h].mean()*100:.1f}% vs {win[m][~h].mean()*100:.1f}%（{(win[m][h].mean()-win[m][~h].mean())*100:+.1f}）'
                   f' 当天内超额 {xdm[m][h].mean()-xdm[m][~h].mean():+.2f}%')
    say('  ' + ' | '.join(row))

say('\n## 逐个组合判定（真实） vs 纯运气平均（500 次打乱）')
for a in ('升级', '保留', '降级', '删除'):
    real = (T.action == a).sum()
    nl = np.array(cnt_null[a])
    say(f'  {a}: 真实 {real} 个 | 纯运气平均 {nl.mean():.0f} 个（95% 区间 {np.percentile(nl,2.5):.0f}–{np.percentile(nl,97.5):.0f}）')
say('\n## 按原等级 × 判定')
say(pd.crosstab(T.grade, T.action).reindex(['SSS', 'S', 'A', 'B']).to_string())

say('\n## 建议升级/保留的组合')
for r in T[T.action.isin(['升级', '保留'])].sort_values(['action', 'wr_diff_00_19'], ascending=[False, False]).itertuples():
    say(f'  [{r.action}] {r.grade:<3} {r.conditions:<55} 00–19: n={r.n_00_19:>4} 胜率差 {r.wr_diff_00_19:+5.1f} 超额 {r.xdm_diff_00_19:+.2f} | '
        f'25–26: n={r.n_25_26:>3} 胜率差 {r.wr_diff_25_26:+5.1f}')

# 建议的新等级文件（不接入）
newg = {'升级': 'SSS', '保留': None, '降级': 'B', '删除': 'DEL'}
v2 = []
for c, a in zip(vc, T.action):
    g = newg[a] or c['grade']
    if a == '保留' and c['grade'] == 'B':
        g = 'A'
    if g == 'DEL':
        continue
    v2.append(dict(c, grade=g, regrade=a))
json.dump(v2, open(os.path.join(HERE, 'out2000', 'validated_combos_v2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
say(f'\n建议文件 out2000/validated_combos_v2.json：保留 {len(v2)} 个；' +
    ' '.join(f'{g}:{sum(1 for c in v2 if c["grade"]==g)}' for g in ('SSS', 'S', 'A', 'B')))
open(os.path.join(HERE, 'out2000', 'regrade.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
