"""择时回测引擎：训练/验证/测试同一套代码。
策略接口  strategy(df) -> 标普仓位 w ∈ [0, WMAX]
  df 为截至当日收盘（含）最近 LOOKBACK 行的 [tr, ret, rf]——引擎截断，结构上杜绝未来函数
执行：当日收盘决定仓位，赚次日收益；现金赚 rf；杠杆部分付 rf + SPREAD；仓位变动付 COST。"""
import math
import numpy as np, pandas as pd

LOOKBACK, WMAX, COST, SPREAD = 600, 1.5, 0.0005, 0.005


def backtest(strategy, df, start, end):
    """返回 (策略日收益, 基准日收益, 仓位)，区间 [start, end]"""
    days = df.index
    s, e = days.searchsorted(pd.Timestamp(start)), days.searchsorted(pd.Timestamp(end), 'right')
    assert s >= 1, '起点之前至少要有 1 天数据'
    ret, rf = df['ret'].to_numpy(), df['rf'].to_numpy()
    out, pos, w_prev = [], [], 1.0
    for i in range(s - 1, e - 1):
        w = float(strategy(df.iloc[max(0, i - LOOKBACK + 1):i + 1]))
        assert 0 <= w <= WMAX + 1e-9, f'{days[i]:%F}: 仓位 {w} 越界'
        out.append(w * ret[i + 1] + (1 - w) * rf[i + 1] - max(w - 1, 0) * SPREAD / 252 - COST * abs(w - w_prev))
        pos.append(w)
        w_prev = w
    idx = days[s:e]
    return pd.Series(out, idx), df['ret'].iloc[s:e], pd.Series(pos, idx)


def nw_test(x):
    """均值的 Newey-West t 与单侧 p（H1: 均值 > 0）"""
    x = np.asarray(x, dtype=float)
    T = len(x)
    lags = int(4 * (T / 100) ** (2 / 9))
    u = x - x.mean()
    s = u @ u / T + sum(2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / T for L in range(1, lags + 1))
    t = x.mean() / math.sqrt(s / T)
    return t, 0.5 * math.erfc(t / math.sqrt(2))


def sharpe(r, rf):
    x = r - rf
    return x.mean() / x.std() * math.sqrt(252)


def mdd(r):
    eq = np.cumprod(1 + np.asarray(r))
    return (eq / np.maximum.accumulate(eq) - 1).min()


def boot_p(stat, cols, B=2000, block=20, seed=0):
    """平稳块自助法（Politis & Romano 1994），H1: stat > 0 的单侧 p；以观测值为中心近似零假设分布"""
    rng = np.random.default_rng(seed)
    X = np.column_stack(cols)
    T = len(X)
    obs = stat(*X.T)
    d = np.empty(B)
    ar = np.arange(T)
    for b in range(B):
        jump = rng.random(T) < 1 / block
        jump[0] = True
        blk = np.cumsum(jump) - 1                                   # 每天所属的块
        head = ar[jump]                                             # 各块起始位置
        idx = (rng.integers(T, size=len(head))[blk] + ar - head[blk]) % T
        d[b] = stat(*X[idx].T)
    return obs, float(np.mean(d - obs >= obs))


def criteria(r, bench, rf, alpha=0.05 / 3):
    """预注册的三条判据（Bonferroni：各自 p < 0.05/3）"""
    x = (r - bench).to_numpy()
    t, p_ret = nw_test(x)
    _, p_worse = nw_test(-x)
    d_sr, p_sr = boot_p(lambda a, b, f: sharpe(a, f) - sharpe(b, f), [r.to_numpy(), bench.to_numpy(), rf.to_numpy()])
    d_dd, p_dd = boot_p(lambda a, b: mdd(a) - mdd(b), [r.to_numpy(), bench.to_numpy()])
    k = {'K1 夏普显著更高': p_sr < alpha, 'K2 收益显著更高': p_ret < alpha and t > 0,
         'K3 回撤显著更浅且收益不显著更低': p_dd < alpha and p_worse > 0.05}
    num = {'年化收益': (1 + r).prod() ** (252 / len(r)) - 1, '基准年化': (1 + bench).prod() ** (252 / len(bench)) - 1,
           '夏普': sharpe(r, rf), '基准夏普': sharpe(bench, rf), '夏普差 p': p_sr, '年化超额(算术)': x.mean() * 252,
           '超额 NW t': t, '超额 p': p_ret, '收益更差 p': p_worse,
           '最大回撤': mdd(r), '基准最大回撤': mdd(bench), '回撤差 p': p_dd}
    return k, num
