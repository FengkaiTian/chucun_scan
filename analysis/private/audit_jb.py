"""核对新数据回测的流程：条件计算 / 收益 / 退市 / 成分股 / 去重。"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE)); RB = os.path.join(ROOT, 'analysis', 'rebuild')
sys.path.insert(0, RB); sys.path.insert(0, ROOT)
import daily_scan as ds
from prepare_data import COND_NAMES, conditions_frame
PANEL = os.path.join(RB, 'data2000', 'gh_johnbrick', 'panel')
P = {k: pd.read_parquet(os.path.join(PANEL, f'{k}.parquet')).loc[:'2026-11-17'] for k in 'OHLCV'}
s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
mem = pd.read_parquet(os.path.join(PANEL, 'member_days.parquet')); memset = set(zip(mem.date, mem.ticker))
rng = np.random.default_rng(1)

# 1. 条件：和扫描器原函数逐条比（截至当天最近 315 根）
pos = s[s.has_open].sample(800, random_state=1)[['ticker', 'signal_day']].assign(k='sig')
neg = []
cols = [t for t in P['C'].columns if P['C'][t].notna().sum() > 600]
while len(neg) < 800:
    t = cols[rng.integers(len(cols))]; c = P['C'][t].dropna()
    d = c.index[rng.integers(320, len(c) - 1)]
    if P['O'].at[d, t] == P['O'].at[d, t]: neg.append((t, d))
neg = pd.DataFrame(neg, columns=['ticker', 'signal_day']).assign(k='rand')
bm, cm, n = 0, {x: 0 for x in COND_NAMES}, 0
cache = {}
for t, d, k in pd.concat([pos, neg]).itertuples(index=False):
    if t not in cache:
        df = pd.DataFrame({x: P[x][t] for x in 'OHLCV'}).dropna(subset=['C']).astype(float)
        cache[t] = (df, conditions_frame(df.O, df.H, df.L, df.C, df.V))
    df, cf = cache[t]
    td = df.loc[:d].tail(315)
    if td.O.isna().any(): continue          # 只核对 315 根都有开盘价的
    r = ds.compute_conditions(td.C, td.H, td.L, td.V, td.O)
    if (r is not None) != bool(cf.at[d, 'base']): bm += 1; continue
    if r is None: continue
    n += 1
    for x in COND_NAMES:
        if bool(r[0][x]) != bool(cf.at[d, x]): cm[x] += 1
print(f'[1] 基础信号与扫描器不一致 {bm} 个；条件核对 {n} 条，不一致：', {k: v for k, v in cm.items() if v} or '无')

# 2. 收益手算抽查
x = s.dropna(subset=['r5']).sample(300, random_state=2); bad = 0
for t, d, r5 in zip(x.ticker, x.signal_day, x.r5):
    c = P['C'][t].dropna(); i = c.index.get_loc(d)
    bad += abs((c.iloc[i + 6] / c.iloc[i + 1] - 1) * 100 - r5) > 1e-3
print(f'[2] 5 日收益手算 300 笔，不一致 {bad}')

# 3. 成分股：信号日都在指数里
print(f'[3] 信号日不在成分股的：{sum((d, t) not in memset for d, t in zip(s.signal_day, s.ticker))}')

# 4. 退市：持有期内没了行情的信号（会被丢掉 → 偏乐观）
last = P['C'].apply(lambda c: c.last_valid_index())
s['last_px'] = s.ticker.map(last)
gone = s[s.r5.isna() & (s.last_px < pd.Timestamp('2026-11-01'))]
print(f'[4] 持有 5 天内退市（r5 缺失）：{len(gone)} 笔 / {len(s)}；', gone[['ticker', 'signal_day', 'last_px']].head(8).to_string(index=False))

# 5. 价格跳变：信号日前后单日 |涨跌| > 50% 且不是退市
C = P['C']; R = C.pct_change(fill_method=None)
big = (R.abs() > 0.5).stack(); big = big[big]
print(f'[5] 全部单日涨跌超 50% 的股票日：{len(big)}（样例）', list(big.index[:10]))
