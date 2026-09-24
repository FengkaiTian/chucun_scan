"""标普 500 择时研究：`python3 -m timing.research` 复现 timing/RESEARCH.md 的全部数字。

流程（PREREGISTRATION_timing.md 与研究协议）
  1. 全部配置（CONFIGS，编号即尝试顺序，含失败的）在训练期 2000–2020 上评估 K1–K3；
  2. 训练期结束后、看验证期之前写下 ≤5 个候选（CANDIDATES）；
  3. 选择规则固定：验证期 2021–2025「策略夏普 − 基准夏普」最大者（并列看训练期）；
  4. 选定后事后披露全部配置的验证期数值（未参与选择），并与 timing/strategy_final.py 逐日对账。
数据只用 data/timing_train.parquet；回测一律 timing.harness.backtest，判据一律 timing.harness.criteria。
"""
import numpy as np, pandas as pd
from timing.harness import backtest, criteria, WMAX

TRAIN, VAL = ('2000-01-01', '2020-12-31'), ('2021-01-01', '2025-12-31')


# ---------------------------------------------------------------- 信号（只用引擎传入的 ≤600 行）
def sma(tr, n):
    """价格（总收益指数）在 n 日均线之上 → 1（Brock, Lakonishok & LeBaron 1992）；历史不足则 1"""
    return 1.0 if len(tr) < n else float(tr[-1] > tr[-n:].mean())


def sma_month(df, n=10):
    """已完成月份的月末值 > 近 n 个月末均值 → 1（Faber 2007）。当月未结束，只用此前各月月末。"""
    m = df['tr'].groupby(df.index.to_period('M')).last().iloc[:-1]
    return 1.0 if len(m) < n else float(m.iloc[-1] > m.iloc[-n:].mean())


def tsmom(tr, rf, n=252):
    """近 n 日总收益超过同期现金收益 → 1（Moskowitz, Ooi & Pedersen 2012）"""
    return 1.0 if len(tr) <= n else float(tr[-1] / tr[-1 - n] > np.prod(1 + rf[-n:]))


def vol_scale(ret, n=21, power=1, target=None):
    """波动率管理（Moreira & Muir 2017：1/方差；Barroso & Santa-Clara 2015：1/波动）。
    目标默认取窗口内（≤600 日）长期波动，使平均仓位约为 1 且不依赖年代的波动水平；上限 WMAX。"""
    s = ret[-n:].std()
    tgt = ret[1:].std() if target is None else target / 252 ** .5
    return min(WMAX, (tgt / s) ** power) if s > 0 else 1.0


# ---------------------------------------------------------------- 通用策略工厂
def make(trend=None, vol=None, low=0.0, cap=WMAX):
    """trend: None|'sma200'|'sma10m'|'tsmom'|'ens'（三者平均）；vol: None 或 vol_scale 参数字典；
    low: 趋势为负时保留的仓位比例（0 = 全部转现金）；cap: 仓位上限（1.0 = 不加杠杆）"""
    def strat(df):
        tr, ret, rf = df['tr'].to_numpy(), df['ret'].to_numpy(), df['rf'].to_numpy()
        sig = {None: lambda: 1.0, 'sma200': lambda: sma(tr, 200), 'sma10m': lambda: sma_month(df),
               'tsmom': lambda: tsmom(tr, rf),
               'ens': lambda: (sma(tr, 200) + sma_month(df) + tsmom(tr, rf)) / 3}[trend]()
        w = low + (1 - low) * sig
        if vol is not None:
            w *= vol_scale(ret, **vol)
        return min(cap, max(0.0, w))
    return strat


CONFIGS = [
    ('T01', '基准：买入持有（w=1）', dict()),
    ('T02', '200 日均线 0/1', dict(trend='sma200')),
    ('T03', '10 个月均线 0/1（月度）', dict(trend='sma10m')),
    ('T04', '12 个月时序动量 0/1', dict(trend='tsmom')),
    ('T05', '波动率目标 1/σ（21 日），上限 1.5', dict(vol=dict(n=21))),
    ('T06', '方差目标 1/σ²（21 日），上限 1.5', dict(vol=dict(n=21, power=2))),
    ('T07', '200 日均线 × 波动率目标（21 日）', dict(trend='sma200', vol=dict(n=21))),
    ('T08', '三趋势信号平均（200 日、10 月、12 月动量）', dict(trend='ens')),
    ('T09', '三趋势信号平均 × 波动率目标（21 日）', dict(trend='ens', vol=dict(n=21))),
    ('T10', '200 日均线，跌破时保留 50%', dict(trend='sma200', low=0.5)),
    ('T11', '三趋势信号平均 × 波动率目标（63 日）', dict(trend='ens', vol=dict(n=63))),
    ('T12', '三趋势信号平均 × 波动率目标（21 日），不加杠杆（上限 1）', dict(trend='ens', vol=dict(n=21), cap=1.0)),
    ('T13', '10 个月均线 × 波动率目标（21 日）', dict(trend='sma10m', vol=dict(n=21))),
    ('T14', '12 个月时序动量 × 波动率目标（21 日）', dict(trend='tsmom', vol=dict(n=21))),
    ('T15', '10 个月均线 × 波动率目标（21 日），不加杠杆（上限 1）', dict(trend='sma10m', vol=dict(n=21), cap=1.0)),
]
CANDIDATES = ['T02', 'T03', 'T08', 'T12', 'T15']     # 训练期结束后、看验证期之前确定
FINAL = 'T02'                                            # 候选中验证期夏普差最大（+0.11）
CFG = {c[0]: c[2] for c in CONFIGS}
NAME = {c[0]: c[1] for c in CONFIGS}


def load():
    return pd.read_parquet('data/timing_train.parquet')


DF = None


def evaluate(strategy, df, period):
    r, b, w = backtest(strategy, df, *period)
    k, n = criteria(r, b, df['rf'].loc[r.index])
    return {**n, '夏普差': n['夏普'] - n['基准夏普'], '平均仓位': w.mean(), '换手(年)': w.diff().abs().sum() / len(w) * 252,
            **{kk.split()[0]: v for kk, v in k.items()}}, r, b, w


def _job(a):
    cid, period = a
    return a, evaluate(make(**CFG[cid]), DF, period)


def md(t, fmt='{:.3f}', head='配置'):
    cell = lambda v: (('✅' if v else '—') if isinstance(v, (bool, np.bool_)) else v if isinstance(v, str)
                      else str(v) if isinstance(v, (int, np.integer)) else fmt.format(v))
    rows = ['| ' + ' | '.join([head] + [str(c) for c in t.columns]) + ' |', '|' + '---|' * (t.shape[1] + 1)]
    rows += ['| ' + ' | '.join([str(i)] + [cell(v) for v in row]) + ' |' for i, row in zip(t.index, t.astype(object).values.tolist())]
    return '\n'.join(rows)


COLS = ['年化收益', '基准年化', '夏普', '基准夏普', '夏普差', '夏普差 p', '年化超额(算术)', '超额 p', '收益更差 p',
        '最大回撤', '基准最大回撤', '回撤差 p', '平均仓位', '换手(年)', 'K1', 'K2', 'K3']


if __name__ == '__main__':
    import multiprocessing as mp, warnings
    import timing.strategy_final as S
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    DF = load()
    jobs = [(c, TRAIN) for c in CFG] + [(c, VAL) for c in CFG] + [(FINAL, ('2000-01-01', '2025-12-31'))]
    with mp.get_context('fork').Pool() as pool:
        res = {a: v for a, v in pool.map(_job, jobs)}
    tab = lambda period, ids: pd.DataFrame({c: res[c, period][0] for c in ids}).T[COLS].rename(index=lambda c: f'{c} {NAME[c]}')
    print('## 1. 全部配置·训练期 2000–2020（判据各自 p < 0.05/3；✅ = 成立）\n')
    print(md(tab(TRAIN, CFG)))
    print('\n## 2. 候选·验证期 2021–2025（用于选择：夏普差最大者当选）\n')
    print(md(tab(VAL, CANDIDATES)))
    pick = max(CANDIDATES, key=lambda c: (res[c, VAL][0]['夏普差'], res[c, TRAIN][0]['夏普差']))
    assert pick == FINAL, pick
    print(f'\n按规则选中：{pick}（验证期夏普差 {res[pick, VAL][0]["夏普差"]:+.3f}）')
    print('\n## 3. 非候选·验证期（选定后事后披露，未参与选择）\n')
    print(md(tab(VAL, [c for c in CFG if c not in CANDIDATES])))

    # ---- 最终策略：对账 + 各期完整数值
    print(f'\n## 4. 最终策略 timing/strategy_final.py（= {FINAL}）\n')
    full = {}
    for name, per in [('训练 2000–2020', TRAIN), ('训练前半 2000–2009', ('2000-01-01', '2009-12-31')),
                      ('训练后半 2010–2020', ('2010-01-01', '2020-12-31')), ('验证 2021–2025', VAL),
                      ('全期 2000–2025', ('2000-01-01', '2025-12-31'))]:
        n, r, b, w = evaluate(S.target_position, DF, per)
        full[name] = n
        if per in (TRAIN, VAL):
            gap = (r - res[FINAL, per][1]).abs().max()
            assert gap < 1e-12, gap
    print('与研究配置逐日对账（训练、验证）：最大日收益差 0\n')
    print(md(pd.DataFrame(full).T[COLS], head='区间'))
    n, r, b, w = evaluate(S.target_position, DF, ('2000-01-01', '2025-12-31'))
    g = lambda s: s.groupby(s.index.year)
    yr = pd.DataFrame({'策略': g(r).apply(lambda x: (1 + x).prod() - 1), '基准': g(b).apply(lambda x: (1 + x).prod() - 1)})
    yr['差'] = yr['策略'] - yr['基准']
    yr['在场比例'] = g(w).mean()
    yr['切换次数'] = g(w.diff().abs()).sum().round().astype(int)
    yr['策略回撤'] = g(r).apply(lambda x: (np.cumprod(1 + x) / np.maximum.accumulate(np.cumprod(1 + x)) - 1).min())
    yr['基准回撤'] = g(b).apply(lambda x: (np.cumprod(1 + x) / np.maximum.accumulate(np.cumprod(1 + x)) - 1).min())
    print('\n逐年（年内回撤按年内净值计算）：\n\n' + md(yr, head='年份'))
    print(f'\n2000–2025：在场 {w.mean():.1%} 的交易日，共 {int(w.diff().abs().sum())} 次切换（每年约 {w.diff().abs().sum() / len(w) * 252:.1f} 次），'
          f'平均每段持续 {len(w) / max(1, w.diff().abs().sum()):.0f} 个交易日')
    print(f'2025-12-31 收盘信号：仓位 {S.target_position(DF.iloc[-600:]):.0f}')
