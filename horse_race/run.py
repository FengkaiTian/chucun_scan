"""策略擂台：16 个策略在同一引擎、同一数据、同一成本下比较。协议见 PREREGISTRATION_horse_race.md。
用法: python -m horse_race.run            （需要 data/hr_*.parquet，由 .github/workflows/horse_race.yml 下载）"""
import math, os, sys
import numpy as np, pandas as pd
from arch.bootstrap import SPA, StationaryBootstrap
from horse_race.engine import load, simulate, truncate
from horse_race import strategies as S
from harness import nw_test

OUT = 'results/horse_race'
SIM_START = '2010-01-01'
BLOCK, REPS, SEED = 10, 10000, 20260924
DEVIATIONS = [
    '首次运行时 RSI(2) 在 2026-09-22 仍持有一只当天没有收盘价的股票，触发了引擎的"信号日必须有收盘价"断言，程序中止，没有产出任何结果。'
    '修正：RSI(2) 和 Clenow 遇到没有收盘价的持仓就卖出（其余策略原本已经这样处理）。',
    'Yahoo 在 2026-09-22 缺了 88% 成分股的行情（NYSE 股票几乎全部缺失）。fetch.py 对中间缺口重新下载了 3 轮，仍然补不齐，属于数据源缺陷。'
    '处理：从交易日历中剔除"缺失比例比前后 11 日中位数高出 10 个百分点以上"的日子，只命中 2026-09-22。'
    '所有资产（包括 SPY）09-21→09-23 的收益都记在 09-23，主区间因此从 182 个交易日变为 181 个。'
    '第一次处理时用的是"缺失 >10%"的规则，它还误删了 2007–2009 年的若干日子；改成现在的规则后重跑，2026 年的全部数字没有变化。',
]


def windows(days):
    last = days[-1]
    return {'2026 年初至今（主检验）': ('2026-01-01', last), '近 12 个月': (days[-252], last), '2025 年': ('2025-01-01', '2025-12-31'),
            '2021–2025': ('2021-01-01', '2025-12-31'), '2011–2025（有幸存者偏差）': ('2011-01-01', '2025-12-31')}


def probe(P, dates=('2012-03-30', '2016-11-15', '2020-03-20', '2023-06-30', '2026-02-27')):
    """未来函数探针：用截至 d 的数据重算特征与目标权重，必须和全样本逐位一致"""
    days = P['C'].index
    for cls in S.CONTESTANTS:
        full = cls(); full.prepare(P)
        for d in dates:
            if pd.Timestamp(d) > days[-1]:
                continue
            i = days.searchsorted(pd.Timestamp(d))
            Q = truncate(P, i)
            tr = cls(); tr.prepare(Q)
            for k in full.F:
                a, b = full.F[k][i], tr.F[k][i]
                assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), f'{cls.name} 特征 {k} 在 {d} 有未来函数'
            w0 = np.zeros(P['C'].shape[1])
            fresh = cls(); fresh.prepare(P)
            t1, t2 = fresh.target(i, P['U'].values[i], w0.copy(), P), tr.target(i, Q['U'].values[i], w0.copy(), Q)
            assert (t1 is None and t2 is None) or np.allclose(t1, t2), f'{cls.name} 权重在 {d} 有未来函数'
    print('未来函数探针：全部通过')


def metrics(r, spy, rf, turn):
    x = (r - spy).values
    ex = r - rf
    eq = (1 + r).cumprod()
    t, p = nw_test(x) if x.std() > 0 else (np.nan, np.nan)
    X = np.column_stack([np.ones(len(r)), spy.values])
    b = np.linalg.lstsq(X, r.values, rcond=None)[0]
    return {'总收益': eq.iloc[-1] - 1, '年化收益': eq.iloc[-1] ** (252 / len(r)) - 1, '年化波动': r.std() * 252 ** .5,
            'Sharpe': ex.mean() / ex.std() * 252 ** .5, '最大回撤': (eq / eq.cummax() - 1).min(),
            '年化超额': x.mean() * 252, 'IR': x.mean() / x.std() * 252 ** .5 if x.std() > 0 else np.nan, 'NW_t': t, '单侧p': p,
            'β': b[1], '年化α': b[0] * 252, '年换手': turn.sum() * 252 / len(r)}


def holm(p):
    p = pd.Series(p)
    o = p.sort_values()
    adj = (o * np.arange(len(o), 0, -1)).cummax().clip(upper=1)
    return adj.reindex(p.index)


def best_prob(R, rf):
    """平稳自助法（Politis & Romano 1994）联合重抽日收益，统计每个策略 Sharpe 排第一的频率与 90% 区间"""
    ex = R.sub(rf, axis=0).values
    bs = StationaryBootstrap(BLOCK, ex, seed=SEED)
    win, srs = np.zeros(ex.shape[1]), []
    for (a,), _ in bs.bootstrap(2000):
        sr = a.mean(0) / a.std(0) * 252 ** .5
        win[np.argmax(sr)] += 1
        srs.append(sr)
    lo, hi = np.percentile(srs, [5, 95], axis=0)
    return pd.DataFrame({'P(Sharpe第一)': win / win.sum(), 'Sharpe 5%': lo, 'Sharpe 95%': hi}, index=R.columns)


def fmt(df, pct=('总收益', '年化收益', '年化波动', '最大回撤', '年化超额', '年化α', 'P(Sharpe第一)')):
    out = df.copy().astype(object)
    for c in df.columns:
        out[c] = [(f'{v:.1%}' if c in pct else f'{v:.2f}' if abs(v) < 100 else f'{v:.0f}') if pd.notna(v) else '' for v in df[c]]
    return '\n'.join(['| | ' + ' | '.join(out.columns) + ' |', '|---' * (len(out.columns) + 1) + '|']
                     + [f'| {i} | ' + ' | '.join(row) + ' |' for i, row in zip(out.index, out.values)])


def main():
    os.makedirs(OUT, exist_ok=True)
    P = load()
    days = P['C'].index
    probe(P)
    runs = {}
    for lag in (1, 0):
        for cls in S.CONTESTANTS + S.BENCHMARKS:
            s = cls()
            r, turn, hold, last = simulate(s, P, SIM_START, days[-1], lag=lag)
            runs[lag, s.name] = r, turn, hold, last
            print(f'lag={lag} {s.name}: 完成', flush=True)
    names = [c.name for c in S.CONTESTANTS]
    spy_name, rf = S.SPY.name, P['rf']
    R1 = pd.DataFrame({n: runs[1, n][0] for n in names + [spy_name, S.EqualWeight.name]})
    R1.to_csv(f'{OUT}/daily_returns_lag1.csv')
    pd.DataFrame({n: runs[0, n][0] for n in names + [spy_name]}).to_csv(f'{OUT}/daily_returns_lag0.csv')

    rep = [f'# 策略擂台结果\n\n数据截至 **{days[-1]:%Y-%m-%d}**；协议见 `PREREGISTRATION_horse_race.md`。'
           f'下单：信号日收盘计算、**次一交易日收盘成交**，单边成本 10bp，现金按 3 个月国债利率计息。\n']
    for wname, (a, b) in windows(days).items():
        sl = lambda s: s.loc[a:b]
        spy = sl(runs[1, spy_name][0]); rfw = sl(rf)
        tab = pd.DataFrame({n: metrics(sl(runs[1, n][0]), spy, rfw, sl(runs[1, n][1])) for n in names + [S.EqualWeight.name, spy_name]}).T
        tab = tab.sort_values('Sharpe', ascending=False)
        rep.append(f'\n## {wname}：{spy.index[0]:%Y-%m-%d} ~ {spy.index[-1]:%Y-%m-%d}（{len(spy)} 个交易日）\n')
        if wname.startswith('2026'):
            tab['Holm p'] = holm(tab.loc[names, '单侧p'])
            Rw = sl(R1[names])
            spa = SPA(-spy.values, -Rw.values, block_size=BLOCK, reps=REPS, bootstrap='stationary', seed=SEED)
            spa.compute()
            bp = best_prob(Rw, rfw)
            tab = tab.join(bp)
            rep.append(fmt(tab) + '\n')
            rep.append(f'\n**多重检验**：16 个策略对 SPY 的单侧 NW 检验，Holm 校正后 p < 0.05 的有 '
                       f'{int((tab["Holm p"] < 0.05).sum())} 个。\n\n'
                       f'**Hansen (2005) SPA 检验**（H0：没有任何策略的期望收益高于 SPY；平稳自助法，块长 {BLOCK}，{REPS} 次）：'
                       f'p = {spa.pvalues["consistent"]:.3f}（lower {spa.pvalues["lower"]:.3f} / upper {spa.pvalues["upper"]:.3f}）。\n')
            lag0 = pd.DataFrame({n: metrics(sl(runs[0, n][0]), spy, rfw, sl(runs[0, n][1])) for n in names}).T
            gross = {n: sl(runs[1, n][0] + 0.001 * runs[1, n][1]) for n in names}
            sens = pd.DataFrame({'Sharpe（主口径）': tab.loc[names, 'Sharpe'], 'Sharpe（当日收盘成交）': lag0['Sharpe'],
                                 'Sharpe（零成本）': {n: ((g - rfw).mean() / (g - rfw).std() * 252 ** .5) for n, g in gross.items()},
                                 '总收益（当日收盘成交）': lag0['总收益']}).loc[tab.index.intersection(names)]
            rep.append('\n### 敏感性（非主检验）\n\n' + fmt(sens, pct=('总收益（当日收盘成交）',)) + '\n')
        else:
            rep.append(fmt(tab) + '\n')

    rep.append('\n## 与预注册的偏差（数据下载后发生）\n\n' + '\n'.join(f'{k}. {d}' for k, d in enumerate(DEVIATIONS, 1)) + '\n')
    rep.append('\n## 各策略当前持仓（最后一个交易日收盘，权重前 10）\n')
    for n in names:
        h = runs[1, n][2]
        rep.append(f'- **{n}**（{len(h)} 只，现金 {max(0.0, 1 - h.sum()):.0%}）：' + ', '.join(f'{t} {v:.1%}' for t, v in h.head(10).items()))
    rep.append('\n## 策略来源与改动说明\n\n| # | 策略 | 调仓 | 来源 | 规则与改动 |\n|---|---|---|---|---|')
    for k, c in enumerate(S.CONTESTANTS, 1):
        rep.append(f'| {k} | {c.name} | {c.freq} | {c.src} | {c.note} |')
    open(f'{OUT}/report.md', 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    sys.exit(main())
