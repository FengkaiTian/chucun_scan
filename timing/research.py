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
FINAL = None
CFG = {c[0]: c[2] for c in CONFIGS}
NAME = {c[0]: c[1] for c in CONFIGS}


def load():
    return pd.read_parquet('data/timing_train.parquet')


def evaluate(strategy, df, period):
    r, b, w = backtest(strategy, df, *period)
    k, n = criteria(r, b, df['rf'].loc[r.index])
    return {**n, '平均仓位': w.mean(), '换手(年)': w.diff().abs().sum() / len(w) * 252,
            **{kk.split()[0]: v for kk, v in k.items()}}, r, b, w


if __name__ == '__main__':
    import sys
    df = load()
    for cid in sys.argv[1:] or list(CFG):
        n = evaluate(make(**CFG[cid]), df, TRAIN)[0]
        print(cid, NAME[cid], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in n.items()}, flush=True)
