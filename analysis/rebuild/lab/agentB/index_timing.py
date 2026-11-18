"""agentB 指数择时：只交易标普500指数本身。
规则同框架：信号日 t 收盘后决定，t+1 收盘买入，持有 hold 天，不重叠，扣 0.10%，聚类 t（按信号日）。
excess = 平均每笔(扣费) − 同期同持有期无条件(每天都买)的平均收益(不扣费)。

index_eval(s1, s2, hold, lo, hi) 可以算任意时期（主会话用它复算保留期 2025–2026）。
"""
import os, sys, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from lab_common import _trades, _cluster_t, COST, DATA, OUT, TRAIN, VALID

CACHE = os.path.join(HERE, 'cache')
HOLDS = (1, 3, 5, 10, 20)
_D = {}


def _data():
    if not _D:
        st = pd.read_parquet(os.path.join(CACHE, 'states.parquet'))     # 由 features.py 生成，只用 t 日及以前
        spx = pd.read_parquet(os.path.join(DATA, 'index_gspc.parquet')).iloc[:, 0].reindex(st.index).ffill().values
        _D.update(st=st, spx=np.ascontiguousarray(spx.astype(np.float64))[:, None], year=st.index.year.values)
    return _D


def index_trades(s1, s2, hold, lag=1):
    d = _data(); st = d['st']
    sig = np.ones(len(st), bool)
    for s in (s1, s2):
        if s:
            sig &= st[s].values
    t, _, r = _trades(np.ascontiguousarray(sig[:, None]), d['spx'], int(hold), int(lag))
    return t, r


def index_eval(s1, s2, hold, lo, hi, lag=1):
    """任意时期 [lo, hi]（按信号日年份）的成绩。s1/s2 为状态名或 None（None,None = 每天都买）"""
    d = _data(); spx = d['spx'][:, 0]; yr = d['year']
    t, r = index_trades(s1, s2, hold, lag)
    m = (yr[t] >= lo) & (yr[t] <= hi)
    x = r[m] - COST; g = t[m]
    # 无条件基准：同期每天买入持有 hold 天的平均收益（重叠，不扣费）
    T = len(spx); tt = np.arange(T - lag - hold)
    tt = tt[(yr[tt] >= lo) & (yr[tt] <= hi)]
    base = ((spx[tt + lag + hold] / spx[tt + lag] - 1) * 100).mean() if len(tt) else np.nan
    if len(x) == 0:
        return {'n': 0}
    return {'n': int(len(x)), 'days': int(len(np.unique(g))), 'per_year': len(x) / (hi - lo + 1),
            'win': float((x > 0).mean() * 100), 'avg': float(x.mean()), 'med': float(np.median(x)),
            'p5': float(np.percentile(x, 5)), 't': float(_cluster_t(x, g)), 'hold': float(hold),
            'per_day': float(x.mean() / hold), 'excess': float(x.mean() - base),
            't_excess': float(_cluster_t(x - base, g)), 'time_in_mkt': float(len(x) * hold / max(len(tt), 1) * 100)}


def run():
    from features import KEY_STATES
    st = _data()['st']; yr = st.index.year
    tr = (yr >= TRAIN[0]) & (yr <= TRAIN[1]); va = (yr >= VALID[0]) & (yr <= VALID[1])
    combos = [(None, None)] + [(s, None) for s in st.columns]
    for a, b in itertools.combinations(KEY_STATES, 2):
        x = (st[a] & st[b]).values
        if x[tr].sum() >= 30 and x[va].sum() >= 10 and x.sum() != min(st[a].sum(), st[b].sum()):
            combos.append((a, b))
    rows = []
    for s1, s2 in combos:
        for h in HOLDS:
            cond = ' & '.join(x for x in (s1, s2) if x) or '每天买(基准)'
            row = dict(name=f'SPX指数 | {cond} | 持{h}天', family='指数择时', params=f'state={cond};hold={h}',
                       hold=h, state1=s1 or '', state2=s2 or '', n_states=(s1 is not None) + (s2 is not None))
            for lab, (lo, hi) in (('train', TRAIN), ('valid', VALID)):
                for k, v in index_eval(s1, s2, h, lo, hi).items():
                    row[f'{lab}_{k}'] = v
            rows.append(row)
    df = pd.DataFrame(rows); df['agent'] = 'agentB'
    p = os.path.join(OUT, 'agentB_index.parquet'); df.to_parquet(p, index=False)
    print(len(df), 'rows ->', p)


if __name__ == '__main__':
    run()
