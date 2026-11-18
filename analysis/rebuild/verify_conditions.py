"""抽样核对：向量化的 base 信号和 37 个条件，是否与 daily_scan.compute_conditions 逐条一致。
做法和每日扫描一样：截至当天、取最近 315 根 K 线调用原函数。"""
import os, sys, warnings
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import daily_scan as ds
from prepare_data import DATA, FIELDS, COND_NAMES, conditions_frame

warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
px = {f: pd.read_parquet(os.path.join(DATA, f'ohlcv_{f.lower()}.parquet')) for f in FIELDS}
sig = pd.read_parquet(os.path.join(DATA, 'signals_20y.parquet'))

pos = sig.sample(min(1500, len(sig)), random_state=0)[['ticker', 'signal_day']].assign(kind='base')
neg = []
tickers = list(px['Close'].columns)
while len(neg) < 1500:
    t = tickers[rng.integers(len(tickers))]
    s = px['Close'][t].dropna()
    if len(s) < 400:
        continue
    neg.append((t, s.index[rng.integers(300, len(s) - 1)]))
neg = pd.DataFrame(neg, columns=['ticker', 'signal_day']).assign(kind='random')

base_mis, cond_mis, checked = 0, {n: 0 for n in COND_NAMES}, 0
cache = {}
for t, day, kind in pd.concat([pos, neg]).itertuples(index=False):
    if t not in cache:
        df = pd.DataFrame({f: px[f][t] for f in FIELDS}).dropna(subset=['Close'])
        cache[t] = (df, conditions_frame(*(df[f] for f in FIELDS)))
    df, cf = cache[t]
    td = df.loc[:day].tail(315)
    res = ds.compute_conditions(td['Close'], td['High'], td['Low'], td['Volume'], td['Open'])
    vec_base = bool(cf.at[day, 'base']) and day >= pd.Timestamp('2006-01-01')
    if kind == 'random' and day < pd.Timestamp('2006-01-01'):
        continue
    if (res is not None) != vec_base:
        base_mis += 1
        continue
    if res is None:
        continue
    checked += 1
    for n in COND_NAMES:
        if bool(res[0][n]) != bool(cf.at[day, n]):
            cond_mis[n] += 1

print(f'base 信号：抽 {len(pos)} 条信号日 + {len(neg)} 个随机日，不一致 {base_mis} 个')
print(f'37 个条件：在 {checked} 条信号上逐条核对')
bad = {k: v for k, v in cond_mis.items() if v}
print('全部一致' if not bad else '不一致：' + ', '.join(f'{k} {v}/{checked}' for k, v in bad.items()))
