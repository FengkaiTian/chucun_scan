"""统一回测引擎：训练（2010–2025）和 2026 测试走同一套代码。

策略接口  target_weights(close, dvol, members) -> pd.Series(ticker -> weight)
  close/dvol : 截至调仓日（含）最近 LOOKBACK 个交易日的日线——引擎负责截断，结构上杜绝未来函数
  members    : 调仓日所在月的时点成分股集合
  权重 >= 0、合计 <= 1（剩余为现金，收益 0）；只能持有 members 内股票或 'SPY'；调仓日必须有收盘价
执行：每月最后一个交易日收盘调仓，月内权重随价格漂移，单边成本 COST。
"""
import math
import numpy as np, pandas as pd

COST, LOOKBACK = 0.001, 600


def load_members(path='data/membership.csv.gz'):
    m = pd.read_csv(path, parse_dates=['month_end'])
    return {d: set(g.ticker) for d, g in m.groupby('month_end')}


def backtest(strategy, close, dvol, members, start, end, cost=COST):
    """返回 [start, end] 内每日净收益（pd.Series）"""
    close = close.sort_index()
    rets = close.pct_change(fill_method=None)
    days = close.index
    ends = pd.Series(days, days).groupby(days.to_period('M')).last()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rebal = [ends[ends < start].iloc[-1]] + [d for d in ends if start <= d < end]
    out, prev = [], pd.Series(dtype=float)
    for k, d in enumerate(rebal):
        i = days.get_loc(d)
        mem = members[d + pd.offsets.MonthEnd(0)]
        w = strategy(close.iloc[max(0, i - LOOKBACK + 1):i + 1], dvol.iloc[max(0, i - LOOKBACK + 1):i + 1], mem)
        w = w[w != 0].astype(float)
        assert (w >= 0).all() and w.sum() <= 1 + 1e-6, f'{d:%F}: 权重必须非负且合计 <= 1'
        assert set(w.index) <= mem | {'SPY'}, f'{d:%F}: 持有了非成分股 {set(w.index) - mem - {"SPY"}}'
        assert close.loc[d, w.index].notna().all(), f'{d:%F}: 调仓日无收盘价'
        turnover = w.sub(prev, fill_value=0).abs().sum()
        nxt = rebal[k + 1] if k + 1 < len(rebal) else end
        idx = rets.index[(rets.index > d) & (rets.index <= nxt)]
        if not len(idx):
            continue
        r = rets.loc[idx, w.index].fillna(0)               # 空仓时 r 无列，组合收益为 0
        value = (1 + r).cumprod() * w                          # 各持仓市值（初始资金 1）
        port = value.sum(axis=1) + (1 - w.sum())
        daily = port.pct_change()
        daily.iloc[0] = port.iloc[0] - 1
        daily.iloc[0] -= cost * turnover
        out.append(daily)
        prev = value.iloc[-1] / port.iloc[-1]
    res = pd.concat(out)
    return res[(res.index >= start) & (res.index <= end)]


def nw_test(x):
    """日超额收益均值的 Newey-West t 统计量与单侧 p 值（H1: 均值 > 0）"""
    x = np.asarray(x, dtype=float)
    T = len(x)
    lags = int(4 * (T / 100) ** (2 / 9))
    u = x - x.mean()
    s = u @ u / T + sum(2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / T for L in range(1, lags + 1))
    t = x.mean() / math.sqrt(s / T)
    return t, 0.5 * math.erfc(t / math.sqrt(2))


def stats(r, bench):
    x = (r - bench).dropna()
    eq = (1 + r).cumprod()
    t, p = nw_test(x)
    return {'年化收益': eq.iloc[-1] ** (252 / len(r)) - 1, '基准年化': (1 + bench).prod() ** (252 / len(bench)) - 1,
            '年化波动': r.std() * 252 ** .5, '最大回撤': (eq / eq.cummax() - 1).min(),
            '跟踪误差': x.std() * 252 ** .5, 'IR': x.mean() / x.std() * 252 ** .5, 'NW_t': t, '单侧p': p}
