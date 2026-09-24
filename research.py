"""研究脚本：`python3 research.py` 复现 RESEARCH.md 中的全部数字。

流程（与 RESEARCH.md 一致）
  1. 所有配置（CONFIGS，按尝试顺序编号，含失败的）在开发期 2010–2020 上评估；
  2. 至多 5 个候选（CANDIDATES）进入验证期 2021–2025，只在它们之间做选择；
  3. 选定后，为完整披露，再事后给出全部配置的验证期指标（未参与选择）；
  4. 最终策略 strategy_final.py 与对应配置逐日对账，并给出各期完整 stats 与逐年超额。
回测一律用 harness.backtest（月末调仓、时点成分股、单边 10bp）。
另：开发期信号诊断（截面秩 IC）见 ic_table，与组合配置一并披露。
"""
import warnings
import numpy as np, pandas as pd
from harness import backtest, stats, load_members, nw_test

warnings.filterwarnings('ignore', category=RuntimeWarning)

DEV, VAL, FULL = ('2010-01-01', '2020-12-31'), ('2021-01-01', '2025-12-31'), ('2010-01-01', '2025-12-31')


# ---------------------------------------------------------------- 特征（只用引擎传入的窗口）
def features(close, dvol, members):
    """调仓日截面特征。股票池：当月时点成分股，且最近 253 个交易日收盘价完整。"""
    c = close.astype(float)
    r = c.pct_change(fill_method=None)
    mkt = r['SPY']
    u = [t for t in c.columns if t in members and t != 'SPY']
    ok = c[u].iloc[-253:].notna().all() if len(c) >= 253 else pd.Series(False, u)
    u = ok[ok].index
    c, r, dv = c[u], r[u], dvol[u].astype(float).where(lambda x: x > 0)
    # 市场模型 r = a + b·SPY + e，用窗口内全部可得日线估计（至多 599 天，要求 >= 252 天）
    R, M = r.iloc[1:].values, mkt.iloc[1:].values[:, None]
    X = np.where(np.isnan(R), np.nan, M)
    mx, my = np.nanmean(X, 0), np.nanmean(R, 0)
    beta = np.nansum((X - mx) * (R - my), 0) / np.nansum((X - mx) ** 2, 0)
    E = pd.DataFrame(R - (my - beta * mx) - beta * M, r.index[1:], u)
    e = E.iloc[-252:-21]                                           # t-12 ~ t-1 月
    return pd.DataFrame({
        'mom': c.iloc[-22] / c.iloc[-253] - 1,                     # 12-1 动量
        'high': c.iloc[-1] / c.iloc[-252:].max(),                  # 52 周新高接近度
        'resmom': e.sum() / e.std(),                               # 残差动量（标准化）
        'vol': r.iloc[-252:].std() * 252 ** .5,                    # 总波动
        'beta': pd.Series(beta, u),
        'ivol': E.iloc[-63:].std() * 252 ** .5,                    # 特质波动（近 3 个月）
        'illiq': (r.abs() / dv).iloc[-252:].mean(),                # Amihud 非流动性
        'rev': c.iloc[-1] / c.iloc[-22] - 1,                       # 1 个月收益（短期反转用）
        'rrev': -E.iloc[-21:].sum() / E.iloc[-252:].std(),         # 残差反转：近 1 月残差取负（标准化）
        'dv': dv.iloc[-252:].mean(),                               # 平均成交额（规模代理）
    }), r, mkt, E


def zs(s):
    """截面秩标准化，抗极值"""
    q = s.rank(pct=True)
    return (q - q.mean()) / q.std()


# ---------------------------------------------------------------- 组合优化（Grinold & Kahn 主动风险框架）
def proj(v, ub):
    """投影到 {0 <= w <= ub, sum w = 1}（二分阈值）"""
    lo, hi = v.min() - 1, v.max()
    for _ in range(50):
        t = (lo + hi) / 2
        lo, hi = (t, hi) if np.clip(v - t, 0, ub).sum() > 1 else (lo, t)
    return np.clip(v - hi, 0, ub)


def optimize(alpha, f, E, mkt, lam, k=10, ub=0.1, it=300, bt=1.0):
    """max a'w - lam·TE²(w)，w>=0、sum=1。TE² = σm²(β'w-1)² + |B'w|² + w'Dw：
    市场模型 + 残差 PCA 前 k 因子 + 特异方差（近 252 日，年化）。无需知道基准市值权重。"""
    X = E[f.index].iloc[-252:].values
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    B = Vt[:k].T * S[:k] * (252 / len(X)) ** .5
    D = ((X - U[:, :k] * S[:k] @ Vt[:k]) ** 2).mean(0) * 252
    s2m, beta, a = mkt.iloc[-252:].var() * 252, f.beta.values, alpha.values
    grad = lambda w: a - lam * 2 * (s2m * (beta @ w - bt) * beta + B @ (B.T @ w) + D * w)
    L = 2 * lam * (s2m * beta @ beta + np.linalg.norm(B, 2) ** 2 + D.max())
    w = y = np.full(len(a), 1 / len(a)); t = 1
    for _ in range(it):                                             # FISTA
        wn = proj(y + grad(y) / L, ub)
        tn = (1 + (1 + 4 * t * t) ** .5) / 2
        y, w, t = wn + (t - 1) / tn * (wn - w), wn, tn
    w = pd.Series(w, f.index)
    return w[w > 1e-4] / w[w > 1e-4].sum()


# ---------------------------------------------------------------- 通用策略工厂
def make(sig, n=50, wt='ew', overlay=None, te=0.03, lb=126, lam=25, ic=0.02, tilt=0.0, cap=1.0):
    """sig: {特征: 符号权重}；选综合得分前 n；wt: ew|ivol|dv|sqrtdv；
    overlay: None | 'state'（近 24 个月 SPY 收益<0 转 SPY）| 'trend'（SPY<200 日均线转 SPY）
             | 'te'（按事前跟踪误差缩放股票篮子，余下持 SPY）| 'state+te'"""
    def strat(close, dvol, members):
        f, r, mkt, E = features(close, dvol, members)
        score = sum(k * zs(f[s]) for s, k in sig.items()).fillna(0)
        if wt == 'opt':                                              # 全池优化：alpha = IC·√12·特质波动·z
            up = close['SPY'].iloc[-1] > close['SPY'].iloc[-253]             # 时序动量：SPY 近 12 月收益 > 0
            w = optimize(ic * 12 ** .5 * f.ivol * zs(score).fillna(0) if ic else 0 * score, f, E, mkt, lam,
                         bt=1 + (tilt if up else -tilt))
            top = w.index
        else:
            top = score.nlargest(n).index
            w = {'ew': pd.Series(1.0, top), 'ivol': 1 / f.vol[top], 'dv': f.dv[top], 'sqrtdv': f.dv[top] ** .5}[wt]
            w = w / w.sum()
            for _ in range(30):                                      # 单股权重上限
                w = w.clip(upper=cap); w = w / w.sum()
        x = 1.0
        if overlay in ('state', 'state+te') and close['SPY'].iloc[-1] < close['SPY'].iloc[-min(505, len(close))]:
            x = 0.0
        if overlay == 'trend' and close['SPY'].iloc[-1] < close['SPY'].iloc[-200:].mean():
            x = 0.0
        if overlay in ('te', 'state+te') and x > 0:
            act = r[top].iloc[-lb:].fillna(0) @ w - mkt.iloc[-lb:]
            x = min(1.0, te / (act.std() * 252 ** .5))
        out = w * x
        if x < 1:
            out['SPY'] = 1 - x
        return out
    return strat


def blend(*parts):
    """按资金比例混合多个子策略：parts = (比例, 配置字典), ..."""
    subs = [(a, make(**cfg)) for a, cfg in parts]
    return lambda close, dvol, members: pd.concat([a * s(close, dvol, members) for a, s in subs]).groupby(level=0).sum()


# ---------------------------------------------------------------- 全部尝试（按时间顺序，编号即尝试序号）
CONFIGS = [
    ('C01', '基线：全体成分股等权', dict(sig={'dv': 0}, n=10 ** 4)),
    ('C02', '基线：全体成分股按成交额加权（市值代理）', dict(sig={'dv': 0}, n=10 ** 4, wt='dv')),
    ('C03', '12-1 动量 前50 等权', dict(sig={'mom': 1})),
    ('C04', '52周新高接近度 前50 等权', dict(sig={'high': 1})),
    ('C05', '残差动量 前50 等权', dict(sig={'resmom': 1})),
    ('C06', '低波动 前50 等权', dict(sig={'vol': -1})),
    ('C07', '低特质波动 前50 等权', dict(sig={'ivol': -1})),
    ('C08', 'Amihud 非流动性高 前50 等权', dict(sig={'illiq': 1})),
    ('C09', '短期反转（1月输家）前50 等权', dict(sig={'rev': -1})),
    ('C10', '残差反转 前50 等权', dict(sig={'rrev': 1})),
    ('C11', '残差反转+残差动量 前50 等权', dict(sig={'rrev': 1, 'resmom': 1})),
    ('C12', '残差反转+残差动量 前100 等权', dict(sig={'rrev': 1, 'resmom': 1}, n=100)),
    ('C13', '残差反转+残差动量 前100 成交额加权', dict(sig={'rrev': 1, 'resmom': 1}, n=100, wt='dv')),
    ('C14', '残差反转+残差动量 前250 成交额加权', dict(sig={'rrev': 1, 'resmom': 1}, n=250, wt='dv')),
    ('C15', '最小跟踪误差复制组合（优化器，无 alpha）', dict(sig={'dv': 0}, wt='opt', ic=0)),
    ('C16', '优化器：残差反转+残差动量 alpha', dict(sig={'rrev': 1, 'resmom': 1}, wt='opt')),
    ('C17', '优化器：残差动量 alpha', dict(sig={'resmom': 1}, wt='opt')),
    ('C18', '优化器复制 + 时序动量择贝塔（目标β=1±0.2）', dict(sig={'dv': 0}, wt='opt', ic=0, tilt=0.2)),
    ('C19', '优化器：低特质波动 alpha（β=1）', dict(sig={'ivol': -1}, wt='opt')),
    ('C20', '全体成分股成交额加权，单股上限 4%', dict(sig={'dv': 0}, n=10 ** 4, wt='dv', cap=0.04)),
    ('C21', '全体成分股 √成交额 加权', dict(sig={'dv': 0}, n=10 ** 4, wt='sqrtdv')),
    ('C22', '12-1 动量 前50 等权 + 市场状态过滤 + 跟踪误差缩放', dict(sig={'mom': 1}, overlay='state+te')),
    ('C23', '12-1 动量 前250 成交额加权，上限 4%', dict(sig={'mom': 1}, n=250, wt='dv', cap=0.04)),
    ('C24', '成交额加权上限 4% + 跟踪误差缩放（目标 3%）', dict(sig={'dv': 0}, n=10 ** 4, wt='dv', cap=0.04, overlay='te')),
    ('C25', '混合：50% C20 + 50% C19', dict(blend=((0.5, 'C20'), (0.5, 'C19')))),
]
CANDIDATES = ['C25', 'C20', 'C14', 'C23', 'C19']          # 开发期结束后、看验证期之前确定
FINAL = 'C20'                                              # 验证期唯一 IR > 0 的候选
CFG = {c[0]: c[2] for c in CONFIGS}
NAME = {c[0]: c[1] for c in CONFIGS}


def strategy(cfg):
    if 'blend' in cfg:
        return blend(*[(a, CFG[k]) for a, k in cfg['blend']])
    return make(**cfg)


def load():
    return (pd.read_parquet('data/train_close.parquet'), pd.read_parquet('data/train_dvol.parquet'), load_members())


DATA = None


def run(cfg, period=FULL, data=None, cost=0.001):
    close, dvol, members = data or DATA
    r = backtest(cfg if callable(cfg) else strategy(cfg), close, dvol, members, *period, cost=cost)
    return r, close['SPY'].pct_change().loc[r.index]


def brief(r, b):
    s, x = stats(r, b), r - b
    return {'年化超额': x.mean() * 252, '跟踪误差': s['跟踪误差'], 'IR': s['IR'], 'NW_t': s['NW_t']}


# ---------------------------------------------------------------- 开发期信号诊断（截面秩 IC，不涉及验证期）
def extra_signals(close, dvol, members):
    """features 之外、在开发期考察过的信号"""
    f, r, mkt, E = features(close, dvol, members)
    u, c = f.index, close[f.index].astype(float)
    dv = dvol[u].astype(float).where(lambda x: x > 0)
    g = f.copy()
    C = np.corrcoef(E.iloc[-252:].fillna(0).values.T)                  # 统计行业：残差相关最高的 10 只
    np.fill_diagonal(C, -np.inf)
    nb = np.argsort(-C, 1)[:, :10]
    r1, r12, e1 = f.rev.values, f.mom.values, E.iloc[-21:].sum().values
    g['peer1'] = r1[nb].mean(1)                                        # 同业 1 月动量（Moskowitz & Grinblatt 1999）
    g['own-peer1'] = -(r1 - g.peer1)                                   # 同业内反转
    g['peer12'] = r12[nb].mean(1)                                      # 同业 12-1 动量
    g['peerE1'] = e1[nb].mean(1)                                       # 同业 1 月残差
    g['avol'] = np.log(dv.iloc[-21:].mean() / dv.iloc[-252:].mean())  # 异常成交额（Gervais et al. 2001）
    g['max'] = -r.iloc[-21:].max()                                     # MAX 效应取负（Bali et al. 2011）
    w = r.iloc[-252:-21]
    g['fip'] = -np.sign(f.mom) * ((w < 0).mean() - (w > 0).mean())     # 信息离散度取负（Da et al. 2014）
    g['mom_fip'] = zs(f.mom) + zs(g.fip)
    g['ma200'] = c.iloc[-1] / c.iloc[-200:].mean()                    # 趋势（Han, Zhou & Zhu 2016）
    mo = c.resample('ME').last().pct_change(fill_method=None)          # 季节性（Heston & Sadka 2008）
    g['season'] = pd.concat([mo.iloc[-k] for k in (12, 24) if len(mo) > k], axis=1).mean(axis=1)
    ab = (r - np.outer(mkt, f.beta)).values                            # 类财报日（成交额峰值）[-1,+1] 超额收益
    dvr = np.nan_to_num((dv / dv.rolling(63, min_periods=20).median()).values, nan=0)

    def ear(nq):
        out = np.zeros(len(u))
        for q in range(nq):
            lo = len(r) - 63 * (q + 1)
            j = dvr[lo:lo + 63].argmax(0) + lo
            out += [np.nansum(ab[max(k - 1, 0):k + 2, i]) for i, k in enumerate(j)]
        return pd.Series(out, u)
    g['ear1'], g['ear4'] = ear(1), ear(4)                              # Brandt et al. 2008；Chan et al. 1996
    g['ear1z'] = g.ear1 / f.ivol
    g['ear1+resmom'] = zs(g.ear1) + zs(f.resmom)
    return g


def _ic_month(e):
    close, dvol, members = DATA
    days = close.index
    i = days.get_loc(e)
    g = extra_signals(close.iloc[max(0, i - 599):i + 1], dvol.iloc[max(0, i - 599):i + 1], members[e + pd.offsets.MonthEnd(0)])
    nxt = days[days.to_period('M') == e.to_period('M') + 1][-1]
    fwd = (close.loc[nxt, g.index] / close.loc[e, g.index] - 1).rank()
    return pd.Series({k: g[k].rank().corr(fwd) for k in g.columns}, name=e)


def ic_table(pool):
    days = DATA[0].index
    ends = pd.Series(days, days).groupby(days.to_period('M')).last()
    ends = [e for e in ends if pd.Timestamp('2009-12-31') <= e <= pd.Timestamp('2020-11-30')]
    ic = pd.DataFrame(pool.map(_ic_month, ends))
    return pd.DataFrame({'月均IC': ic.mean(), 't': ic.mean() / ic.std() * len(ic) ** .5, '月数': ic.count()})


# ---------------------------------------------------------------- 输出
def _job(a):
    cid, period, cost = a
    return a, run(CFG[cid], period, cost=cost)


def md(df, fmt='{:.3f}', head='配置'):
    cols = list(df.columns)
    rows = ['| ' + ' | '.join([head] + [str(c) for c in cols]) + ' |', '|' + '---|' * (len(cols) + 1)]
    cell = lambda v: v if isinstance(v, str) else str(v) if isinstance(v, (int, np.integer)) else fmt.format(v)
    rows += ['| ' + ' | '.join([str(i)] + [cell(v) for v in row]) + ' |'
             for i, row in zip(df.index, df.astype(object).values.tolist())]
    return '\n'.join(rows)


def rolling_hits(x, T=180, step=21):
    """历史上任意 T 日窗口（每 step 日取一个起点）的 NW 单侧 p < 0.05 比例"""
    ps = [nw_test(x.iloc[k:k + T])[1] for k in range(0, len(x) - T + 1, step)]
    return np.mean(np.array(ps) < 0.05), len(ps)


if __name__ == '__main__':
    import multiprocessing as mp
    from math import erfc, sqrt
    import strategy_final
    DATA = load()
    close = DATA[0]
    cov = pd.Series({d: np.mean([t in close.columns and pd.notna(close[t].asof(d)) for t in m])
                     for d, m in DATA[2].items() if d <= pd.Timestamp('2025-12-31')})
    print('## 0. 成分股行情覆盖率（月末成分股中当日有收盘价的比例，按年平均）\n')
    print(md(cov.groupby(cov.index.year).mean().to_frame('覆盖率').T, '{:.2f}', '年份') + '\n')
    with mp.get_context('fork').Pool() as pool:
        print('## 1. 开发期信号诊断：截面秩 IC（月末信号 vs 下月收益，2010-01 ~ 2020-12）\n')
        print(md(ic_table(pool), head='信号'))
        jobs = [(c, p, k) for p, k in [(DEV, .001), (VAL, .001), (DEV, 0)] for c in CFG] + [(FINAL, FULL, .001)]
        res = {(c, p, k): rb for (c, p, k), rb in pool.map(_job, jobs)}
    tab = pd.DataFrame({c: {**{f'开发·{k}': v for k, v in brief(*res[c, DEV, .001]).items()},
                            '开发·成本拖累': (res[c, DEV, 0][0] - res[c, DEV, .001][0]).mean() * 252,
                            **{f'验证·{k}': v for k, v in brief(*res[c, VAL, .001]).items()}} for c in CFG}).T
    tab.insert(0, '说明', pd.Series(NAME))
    dev_cols = ['说明'] + [c for c in tab.columns if c.startswith('开发')]
    val_cols = [c for c in tab.columns if c.startswith('验证')]
    print('\n## 2. 全部配置·开发期 2010–2020（年化超额 = 日均超额×252，已扣成本；成本拖累 = 零成本回测 − 实际）\n')
    print(md(tab[dev_cols]))
    top = ['C02', 'C14', 'C19', 'C20', 'C21', 'C23', 'C24', 'C25']
    act = pd.DataFrame({c: res[c, DEV, .001][0] - res[c, DEV, .001][1] for c in top}).resample('ME').sum()
    print('\n开发期月度主动收益相关系数（候选筛选用）：\n\n' + md(act.corr(), '{:.2f}'))
    print('\n## 3. 候选·验证期 2021–2025（用于选择）\n')
    print(md(tab.loc[CANDIDATES, ['说明'] + val_cols]))
    print('\n## 4. 非候选·验证期（事后披露，未参与选择）\n')
    print(md(tab.loc[[c for c in CFG if c not in CANDIDATES], ['说明'] + val_cols]))

    # ---- 最终策略
    r, b = run(strategy_final.target_weights, FULL)
    gap = (r - res[FINAL, FULL, .001][0]).abs().max()
    print(f'\n## 5. 最终策略 strategy_final.py（= {FINAL}）\n\n与研究配置逐日对账：最大日收益差 {gap:.2e}')
    assert gap < 1e-10
    full = {'开发 2010–2020': (r.loc[:'2020'], b.loc[:'2020']), '验证 2021–2025': (r.loc['2021':], b.loc['2021':]),
            '全期 2010–2025': (r, b)}
    st = pd.DataFrame({k: stats(*v) for k, v in full.items()})
    for k, (rr, bb) in full.items():
        st.loc['年化超额(算术)', k] = (rr - bb).mean() * 252
        st.loc['贝塔', k] = np.cov(rr, bb)[0, 1] / bb.var()
        e = rr - st.loc['贝塔', k] * bb
        st.loc['残差α(年化)', k] = e.mean() * 252
        st.loc['残差IR', k] = e.mean() / e.std() * 252 ** .5
    print('\n' + md(st.T.astype(float), '{:.4f}', '区间'))
    x = r - b
    yr = pd.DataFrame({'策略': (1 + r).groupby(r.index.year).prod() - 1, 'SPY': (1 + b).groupby(b.index.year).prod() - 1})
    yr['超额'] = yr['策略'] - yr['SPY']
    yr['年内IR'] = x.groupby(x.index.year).mean() / x.groupby(x.index.year).std() * 252 ** .5
    yr['年内NW_t'] = [nw_test(g)[0] for _, g in x.groupby(x.index.year)]
    print('\n逐年（全期连续回测）：\n\n' + md(yr, head='年份'))

    # ---- 持仓与换手
    days = close.index
    ends = pd.Series(days, days).groupby(days.to_period('M')).last()
    hold = []
    for d in [e for e in ends if pd.Timestamp('2009-12-31') <= e <= pd.Timestamp('2025-12-31')][::12]:
        i = days.get_loc(d)
        w = strategy_final.target_weights(close.iloc[max(0, i - 599):i + 1], DATA[1].iloc[max(0, i - 599):i + 1],
                                          DATA[2][d + pd.offsets.MonthEnd(0)])
        hold.append({'调仓日': f'{d:%Y-%m-%d}', '持股数': len(w), '前10权重和': w.nlargest(10).sum(),
                     '触及上限只数': int((w > 0.0399).sum()), '有效持股数(1/Σw²)': 1 / (w ** 2).sum()})
    print('\n持仓集中度（每年 12 月末调仓日）：\n\n' + md(pd.DataFrame(hold).set_index('调仓日'), '{:.3f}', '调仓日'))
    r0 = backtest(strategy_final.target_weights, *DATA, *FULL, cost=0)
    print(f'\n成本拖累（10bp 单边）：{(r0 - r).mean() * 252:.4%}/年')

    # ---- 功效
    print('\n## 6. 2026 功效（T=180 日，正态近似：P = 1 − Φ(1.645 − IR·√(180/252))）\n')
    print('| 真实 IR | 0 | 0.2 | 0.33 | 0.5 | 0.69 | 1.0 | 1.9 |\n|---|---|---|---|---|---|---|---|')
    print('| P(p<0.05) | ' + ' | '.join(f'{0.5 * erfc((1.645 - ir * sqrt(180 / 252)) / sqrt(2)):.1%}'
                                         for ir in (0, .2, .33, .5, .69, 1, 1.9)) + ' |')
    for k, (rr, bb) in list(full.items())[:2]:
        h, n = rolling_hits(rr - bb)
        print(f'\n历史滚动 180 日窗口中 p<0.05 的比例（{k}，{n} 个窗口，每 21 日一个起点）：{h:.1%}')

    # ---- 工程冒烟测试：模拟 eval_2026.py 的数据形态（2023 起、仅期末成分股），跑 2025 年（样本内，非业绩证据）
    tick = sorted(set().union(*(v for d, v in DATA[2].items() if d >= pd.Timestamp('2024-12-31'))) & set(close.columns))
    cc = close.loc['2023-01-01':, tick + ['SPY']].dropna(axis=1, how='all')
    rr, bb = run(strategy_final.target_weights, ('2025-01-01', '2025-12-31'), (cc, DATA[1].loc['2023-01-01':, cc.columns], DATA[2]))
    print(f'\n## 7. 冒烟测试（eval 数据形态，2025 年，{len(rr)} 日）：运行正常；IR {stats(rr, bb)["IR"]:.2f}（样本内，仅供核对）')
