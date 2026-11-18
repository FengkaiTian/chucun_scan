"""多个组合叠加：一只股票同时命中多少个"好组合"，命中越多是不是越好？不用机器学习，只数组合。

做法：
  1. 只在挖掘期挑"好组合"（几种标准，见 FAMILIES）。
  2. 每个信号数一下命中了几个好组合（票数）。
  3. 在挖掘期之后的年份，按票数分档，看 T+5 胜率、超额 x5、以及"当天内超额"（x5 减去同一天全部基础信号的平均，
     剔除大盘涨跌，只看选股本身）。t 值按信号日聚类。
  4. 贪心拼盘：在挖掘期逐个加组合，让并集的当天内超额最大，再看之后的年份。
  5. 现有 validated_combos.json 的 334 个组合也按命中个数分档。

用法：python combo_votes.py <挖掘截止日> <组合目录> <组合文件名>
  例：python combo_votes.py 2012-12-31 out2012 combos_d4_n40.parquet
"""
import os, sys, json
TRAIN_END = sys.argv[1] if len(sys.argv) > 1 else '2024-12-31'
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else 'out2000'
FNAME = sys.argv[3] if len(sys.argv) > 3 else 'combos_d4_n60.parquet'
os.environ['MS_TRAIN_END'] = TRAIN_END
sys.argv = sys.argv[:1]
import numpy as np, pandas as pd
from numba import njit
import mega_search as ms

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)


@njit(cache=True)
def add_votes(bits, combos, votes):
    W = bits.shape[1]
    tmp = np.empty(W, np.uint64)
    for j in range(combos.shape[0]):
        for w in range(W):
            tmp[w] = bits[combos[j, 0], w]
        for k in range(1, combos.shape[1]):
            c = combos[j, k]
            if c < 0:
                break
            for w in range(W):
                tmp[w] &= bits[c, w]
        for w in range(W):
            x = tmp[w]
            while x != np.uint64(0):
                low = x & (~x + np.uint64(1))
                r = w * 64 + int(ms.popcnt(low - np.uint64(1)))
                votes[r] += 1
                x ^= low


def cl_t(y, day):
    """均值和按信号日聚类的 t"""
    if len(y) < 5:
        return np.nan, np.nan
    df = pd.DataFrame({'y': y, 'd': day})
    mu = y.mean()
    g = df.groupby('d').y.apply(lambda v: (v - mu).sum())
    G = len(g)
    se = np.sqrt((g ** 2).sum() * G / max(G - 1, 1)) / len(y)
    return mu, mu / se if se > 0 else np.nan


def diff_t(hit, y, day):
    """命中 − 没命中 的差，按信号日聚类（回归 y ~ 1 + hit）"""
    import statsmodels.api as sm
    X = sm.add_constant(hit.astype(float))
    if hit.sum() < 5 or (~hit).sum() < 5:
        return np.nan, np.nan
    m = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(day)[0]})
    return m.params[1], m.tvalues[1]


BUCKETS = [(0, 0, '0 个'), (1, 9, '1–9'), (10, 99, '10–99'), (100, 999, '100–999'), (1000, 10**9, '≥1000')]


def table(lab, votes, mask, d):
    """mask 指定的年份里，按票数分档"""
    sub = d[mask]
    v = votes[mask]
    say(f'  {lab}（n={mask.sum()}，天数={sub.signal_day.nunique()}）')
    say(f'    {"票数":<9}{"n":>6}{"天数":>6}{"T+5胜率":>9}{"超额x5":>9}{"当天内超额":>11}{"t":>7}')
    for lo, hi, name in BUCKETS:
        m = (v >= lo) & (v <= hi)
        if m.sum() < 10:
            continue
        s = sub[m]
        mu, t = cl_t(s.xdm.values, s.signal_day.values)
        say(f'    {name:<9}{m.sum():>6}{s.signal_day.nunique():>6}{(s.r5 > 0).mean()*100:>8.1f}%{s.x5.mean():>+8.2f}%{mu:>+10.2f}%{t:>7.2f}')
    hit = v > 0
    dw, tw = diff_t(hit, (sub.r5.values > 0) * 100.0, sub.signal_day.values)
    dx, tx = diff_t(hit, sub.xdm.values, sub.signal_day.values)
    rho = pd.Series(v).corr(pd.Series(sub.xdm.values), method='spearman')
    say(f'    命中≥1 vs 0：胜率差 {dw:+.1f} 点（t={tw:.2f}），当天内超额差 {dx:+.2f}%（t={tx:.2f}）；票数与当天内超额秩相关 ρ={rho:+.3f}')
    top = v >= np.quantile(v[v > 0], 0.75) if (v > 0).sum() > 40 else None
    if top is not None:
        dw, tw = diff_t(top, (sub.r5.values > 0) * 100.0, sub.signal_day.values)
        dx, tx = diff_t(top, sub.xdm.values, sub.signal_day.values)
        say(f'    票数最多的 1/4（命中者里，≥{np.quantile(v[v > 0], 0.75):.0f} 票） vs 其余：胜率差 {dw:+.1f} 点（t={tw:.2f}），当天内超额差 {dx:+.2f}%（t={tx:.2f}）')


def main():
    d, tr, names, groups, C = ms.build()
    d = d.copy()
    d['xdm'] = d.x5 - d.groupby('signal_day').x5.transform('mean')   # 当天内超额
    yr = d.signal_day.dt.year.values
    periods = [('挖掘期内（样本内，仅供对照）', tr)]
    if pd.Timestamp(TRAIN_END).year < 2020:
        periods += [(f'{pd.Timestamp(TRAIN_END).year+1}–2024（样本外）', (~tr) & (yr <= 2024)),
                    ('2025–2026（样本外）', yr >= 2025)]
    else:
        periods += [('2025–2026（样本外）', ~tr)]
    say(f'挖掘截止 {TRAIN_END}；组合文件 {OUTDIR}/{FNAME}；条件 {len(names)} 个；信号 {len(d)} 条')

    R = pd.read_parquet(os.path.join(HERE, OUTDIR, FNAME))
    combos_all = R[['c0', 'c1', 'c2', 'c3', 'c4']].values.astype(np.int64)
    bits = ms.pack(C)
    FAMILIES = {
        '原挑法（四周期都跑赢 + T+5≥65%）': (R.orig == 1) & (R.wr5 >= 65),
        '原挑法里 T+5≥80%（S 级）': (R.orig == 1) & (R.wr5 >= 80),
        '训练 t≥3（天数≥30）': (R.t >= 3) & (R.days >= 30),
        '训练 t≥2（天数≥30）': (R.t >= 2) & (R.days >= 30),
    }
    for fam, sel in FAMILIES.items():
        sel = sel.values
        votes = np.zeros(len(d), np.int64)
        add_votes(bits, combos_all[sel], votes)
        say(f'\n## 好组合 = {fam}：{sel.sum()} 个')
        for lab, m in periods:
            table(lab, votes, m, d)

    # 贪心拼盘
    say('\n## 贪心拼盘：挖掘期逐个加组合，让并集的当天内超额 × √n 最大（最多 30 个）')
    cand = R[(R.t >= 2) & (R.days >= 30)].sort_values('t', ascending=False).head(3000)
    M = np.zeros((len(cand), len(d)), bool)
    for i, row in enumerate(cand[['c0', 'c1', 'c2', 'c3', 'c4']].values.astype(int)):
        mm = np.ones(len(d), bool)
        for c in row:
            if c >= 0: mm &= C[:, c]
        M[i] = mm
    y = d.xdm.values
    Mt = M[:, tr]; yt = y[tr]
    U = np.zeros(tr.sum(), bool); chosen = []
    for step in range(30):
        newU = Mt | U
        cnt = newU.sum(1); s = (newU * yt).sum(1)
        score = s / np.sqrt(np.maximum(cnt, 1))
        score[chosen] = -np.inf
        j = int(np.argmax(score))
        if chosen and score[j] <= (U * yt).sum() / np.sqrt(max(U.sum(), 1)) + 1e-9:
            break
        chosen.append(j); U = newU[j]
    say(f'  选了 {len(chosen)} 个组合，挖掘期并集 n={U.sum()}')
    for r in cand.iloc[chosen[:10]].itertuples():
        say(f'    {r.name[:90]}')
    hitall = M[chosen].any(0)
    for lab, m in periods:
        sub = d[m]; h = hitall[m]
        dw, tw = diff_t(h, (sub.r5.values > 0) * 100.0, sub.signal_day.values)
        dx, tx = diff_t(h, sub.xdm.values, sub.signal_day.values)
        say(f'  {lab}：命中 {h.sum()} 条，胜率 {(sub.r5[h] > 0).mean()*100:.1f}% vs {(sub.r5[~h] > 0).mean()*100:.1f}%（差 {dw:+.1f} t={tw:.2f}），'
            f'当天内超额差 {dx:+.2f}%（t={tx:.2f}）')

    # 现有 334 个组合：命中个数
    say('\n## 现有 validated_combos.json：每个信号命中几个组合（2020–2025 挑出，所以 2020–2024 是样本内）')
    vc = json.load(open(os.path.join(HERE, '..', '..', 'validated_combos.json'), encoding='utf-8'))
    for grade_set, lab in ((None, '全部 334 个'), ({'SSS'}, 'SSS 108 个'), ({'SSS', 'S', 'A'}, 'SSS+S+A')):
        cnt = np.zeros(len(d), np.int64)
        for c in vc:
            if grade_set and c['grade'] not in grade_set:
                continue
            m = np.ones(len(d), bool)
            for k in c['conditions']:
                m &= d[k].fillna(False).astype(bool).values
            cnt += m
        say(f'  {lab}')
        for plab, pm in (('2000–2019（挑选时没见过）', yr <= 2019), ('2025–2026', yr >= 2025)):
            sub = d[pm]; v = cnt[pm]
            parts = []
            for lo, hi, nm in ((0, 0, '0'), (1, 1, '1'), (2, 3, '2–3'), (4, 9, '4–9'), (10, 10**6, '≥10')):
                mm = (v >= lo) & (v <= hi)
                if mm.sum() >= 10:
                    mu, t = cl_t(sub.xdm.values[mm], sub.signal_day.values[mm])
                    parts.append(f'{nm}个:n={mm.sum()} 胜率{(sub.r5.values[mm] > 0).mean()*100:.0f}% 当天内{mu:+.2f}%(t={t:.1f})')
            say(f'    {plab}： ' + ' | '.join(parts))

    out = os.path.join(HERE, OUTDIR, f'votes_{pd.Timestamp(TRAIN_END).year}.txt')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines))
    say(f'\n完成：{out}')


if __name__ == '__main__':
    main()
