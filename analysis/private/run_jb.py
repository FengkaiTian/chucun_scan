"""私人回测：用 Johnbrick123/sp500-data（无幸存者偏差）重跑基础信号和星级。结果只留在本机。
数据：analysis/rebuild/data2000/gh_johnbrick/panel，2000-01-03 ~ 2026-11-17。
信号日必须是当天的成分股；次日收盘入场；成本 0.10% 来回；基准 = 同进同出的 SPY 含分红。
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RB = os.path.join(ROOT, 'analysis', 'rebuild')
sys.path.insert(0, RB); sys.path.insert(0, ROOT)
import prepare_data as pdx

PANEL = os.path.join(RB, 'data2000', 'gh_johnbrick', 'panel')
END = pd.Timestamp('2026-11-17')
COST = 0.10

P = {k: pd.read_parquet(os.path.join(PANEL, f'{k}.parquet')).loc[:END] for k in 'OHLCV'}
mem = pd.read_parquet(os.path.join(PANEL, 'member_days.parquet'))
mem = mem[mem.date <= END]
memset = set(zip(mem.date, mem.ticker))
spy = pd.read_parquet(os.path.join(RB, 'data2000', 'index_spy_tr.parquet')).iloc[:, 0].loc[:END]
vix = pd.read_parquet(os.path.join(RB, 'data2000', 'index_vix.parquet')).iloc[:, 0].loc[:END]
combos = json.load(open(os.path.join(ROOT, 'validated_combos.json'), encoding='utf-8'))
sss = [c['conditions'] for c in combos if c['grade'] == 'SSS']

rows = []; t0 = time.time()
tickers = sorted(set(mem.ticker))
for i, t in enumerate(tickers):
    if t not in P['C'].columns: continue
    df = pd.DataFrame({k: P[k][t] for k in 'OHLCV'}).dropna(subset=['C'])
    if len(df) < 300: continue
    o, h, l, c, v = (df[k].astype(float) for k in 'OHLCV')
    cf = pdx.conditions_frame(o, h, l, c, v)
    sig = cf[cf.base & (cf.index >= '2000-01-01')]
    sig = sig[[(d, t) in memset for d in sig.index]]
    if sig.empty: continue
    fw = pdx.forward(o, h, l, c).loc[sig.index]
    part = pd.concat([sig, fw], axis=1)
    part.insert(0, 'ticker', t)
    pos = df.index.get_indexer(sig.index)
    part['entry_day'] = [df.index[p + 1] if p + 1 < len(df) else pd.NaT for p in pos]
    part['has_open'] = o.loc[sig.index].notna().values & o.shift(1).loc[sig.index].notna().values
    rows.append(part)
    if (i + 1) % 200 == 0: print(f'{i+1}/{len(tickers)} {time.time()-t0:.0f}s', flush=True)

s = pd.concat(rows).rename_axis('signal_day').reset_index()
s['vix'] = vix.reindex(s.signal_day, method='ffill').values
for n in (5, 20):
    fwd = (spy.shift(-n) / spy - 1) * 100
    s[f'b{n}'] = fwd.reindex(s.entry_day).values
    s[f'x{n}'] = s[f'r{n}'] - COST - s[f'b{n}']        # 扣成本后的超额
s['sss'] = np.any([s[cs].all(axis=1) for cs in sss], axis=0)
s['tier'] = np.select([s.gap_down_close, s.doji_bottom, s.sss], ['★★★', '★★', '★'], '参考')
s['year'] = s.signal_day.dt.year
s.to_parquet(os.path.join(HERE, 'jb_signals.parquet'), index=False)
print('done', len(s), time.time() - t0)
