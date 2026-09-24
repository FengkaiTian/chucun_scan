"""日频组合回测引擎（harness.backtest 的日/周/月通用版）。

时序（每个交易日收盘时依次）：
  1. 持仓按当日收益漂移；现金按 3 个月国债利率计息
  2. 执行到期的调仓单（信号日 + lag 个交易日的收盘价成交），单边成本 cost × |Δw|
  3. 若为信号日：strategy.target(i, members, w, P) 只能读取第 i 行及以前的数据，返回目标权重或 None（不调仓）
约束（与 harness 一致）：权重 >= 0、合计 <= 1；只能持有信号日的时点成分股或 SPY；信号日必须有收盘价。
成交日没有收盘价的目标股票权重置 0（留作现金）。收益用 pct_change(fill_method=None).fillna(0)，同 harness。
"""
import numpy as np, pandas as pd

COST = 0.001


def load(prefix='data/hr_', snap='results/wiki_test_{}.html'):
    from horse_race.universe import membership
    P = {k: pd.read_parquet(f'{prefix}{f}.parquet').astype('float64') for k, f in
         [('C', 'close'), ('H', 'high'), ('L', 'low'), ('V', 'volume')]}
    days, tick = P['C'].index, P['C'].columns
    P['U'] = membership(days, snap).reindex(columns=tick, fill_value=False)
    try:
        irx = pd.read_parquet(f'{prefix}irx.parquet')['irx'].reindex(days).ffill()
    except FileNotFoundError:
        irx = pd.Series(0.0, days)
    P['rf'] = (irx.fillna(0) / 100 / 252)
    return finish(P)


def finish(P):
    """派生量：成交额、收益、SPY 列号"""
    P['DV'] = (P['C'] * P['V']).where(lambda x: x > 0)
    P['R'] = P['C'].pct_change(fill_method=None).fillna(0)
    P['spy'] = P['C'].columns.get_loc('SPY')
    P['me'] = schedule(P['C'].index, 'M')        # 交易日历（非价格信息），截断时保持完整
    return P


def truncate(P, i):
    """只保留前 i+1 行——用于未来函数探针"""
    Q = {k: (v.iloc[:i + 1] if isinstance(v, (pd.DataFrame, pd.Series)) else v) for k, v in P.items()}
    return Q


def schedule(days, freq):
    d = pd.Series(days, days)
    if freq == 'D':
        return np.ones(len(days), bool)
    if freq == 'M':
        last = d.groupby(days.to_period('M')).transform('max')
    elif freq == 'W':
        last = d.groupby(days.to_period('W-FRI')).transform('max')
    elif freq == 'WED':
        return np.asarray(days.weekday == 2)
    return np.asarray(last == d)


def simulate(strat, P, start, end, lag=1, cost=COST):
    days = P['C'].index
    C, R, U = P['C'].values, P['R'].values, P['U'].values
    rf = P['rf'].values
    s, e = days.searchsorted(pd.Timestamp(start)), days.searchsorted(pd.Timestamp(end), 'right') - 1
    sig = schedule(days, strat.freq)
    strat.prepare(P)
    N = C.shape[1]
    w, pending = np.zeros(N), {}
    out, turn, last_tgt = np.zeros(e - s + 1), np.zeros(e - s + 1), None

    def execute(i, tgt):
        nonlocal w
        tgt = np.where(np.isnan(C[i]), 0.0, tgt)
        t = np.abs(tgt - w).sum()
        w = tgt
        return t

    for k, i in enumerate(range(s, e + 1)):
        g = 0.0
        if k:
            r = R[i]
            g = w @ r + (1 - w.sum()) * rf[i]
            w = w * (1 + r) / (1 + g)
        if i in pending:
            t = execute(i, pending.pop(i))
            g -= cost * t
            turn[k] += t
        if sig[i]:
            tgt = strat.target(i, U[i], w.copy(), P)
            if tgt is not None:
                tgt = np.asarray(tgt, float)
                nz = tgt != 0
                assert (tgt >= 0).all() and tgt.sum() <= 1 + 1e-9, f'{strat.name} {days[i]:%F}: 权重非法'
                ok = U[i].copy(); ok[P['spy']] = True
                assert ok[nz].all(), f'{strat.name} {days[i]:%F}: 持有非成分股'
                assert not np.isnan(C[i][nz]).any(), f'{strat.name} {days[i]:%F}: 信号日无收盘价'
                last_tgt = (days[i], tgt)
                if lag == 0:
                    t = execute(i, tgt)
                    g -= cost * t
                    turn[k] += t
                else:
                    pending[i + lag] = tgt
        out[k] = g
    idx = days[s:e + 1]
    hold = pd.Series(w, P['C'].columns)
    return pd.Series(out, idx), pd.Series(turn, idx), hold[hold > 0].sort_values(ascending=False), last_tgt
