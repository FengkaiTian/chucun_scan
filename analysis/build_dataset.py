"""回测特征数据集：重放每个交易日的信号（不改动 daily_scan 的任何逻辑，只调用它），
记录信号当天的全部特征 + 之后的真实收益。输出 analysis/signals_dataset.csv。"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import daily_scan as ds

OUT = os.path.dirname(os.path.abspath(__file__))
WINDOW = 315
combos = [c for c in json.load(open(ds.COMBO_FILE, encoding='utf-8')) if c['grade'] in ('SSS', 'S', 'A', 'B')]
sector_map, name_map = ds.get_sp500_info()
tickers = list(sector_map)
print('download...', flush=True)
data = yf.download(tickers, period='5y', group_by='ticker', auto_adjust=True, threads=True, progress=False)
spx = yf.download('^GSPC', period='5y', auto_adjust=True, progress=False)['Close'].squeeze()
vix = yf.download('^VIX', period='5y', auto_adjust=True, progress=False)['Close'].squeeze()
spx_ma200 = spx.rolling(200).mean()

def spx_fwd(entry, n):
    i = spx.index.searchsorted(entry)
    if i + n >= len(spx) or spx.index[i] != entry: return np.nan
    return (spx.iloc[i + n] / spx.iloc[i] - 1) * 100

rows = []
for k, t in enumerate(tickers):
    try:
        df = data[t].dropna()
    except Exception:
        continue
    if len(df) < 300: continue
    c, h, l, v, o = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rsi = ds.calc_rsi(c); _, _, hist = ds.calc_macd(c)
    cand = (hist < 0) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2)) & (rsi.shift(1) < 31) & (rsi > rsi.shift(1))
    atr = ds.calc_atr(h, l, c)
    ma200 = c.rolling(200).mean()
    for idx in np.where(cand.values)[0]:
        if idx < 260: continue
        td = df.iloc[:idx + 1].tail(WINDOW)
        try:
            res = ds.compute_conditions(td['Close'], td['High'], td['Low'], td['Volume'], td['Open'])
        except Exception:
            continue
        if res is None: continue
        conds, disp = res
        grade, combo = ds.match_grade(conds, combos)
        day = df.index[idx]
        cl = float(c.iloc[idx])
        r = dict(ticker=t, sector=sector_map[t], signal_day=day, grade=grade or '',
                 combo_wr5=combo['wr5'] if combo else np.nan, combo=' + '.join(combo['conditions']) if combo else '',
                 n_conds=sum(conds.values()), conds='|'.join(k2 for k2, v2 in conds.items() if v2))
        r.update({k2: disp[k2] for k2 in ('rsi', 'cci', 'cmf', 'wr', 'hist', 'dist_52w', 'deviation')})
        r['vol_ratio5'] = round(float(v.iloc[idx] / v.iloc[idx-5:idx].mean()), 2)
        r['ret5d'] = round((cl / c.iloc[idx-5] - 1) * 100, 2)
        r['ret20d'] = round((cl / c.iloc[idx-20] - 1) * 100, 2)
        r['ret60d'] = round((cl / c.iloc[idx-60] - 1) * 100, 2)
        r['dist_high252'] = round((cl / c.iloc[idx-251:idx+1].max() - 1) * 100, 2)
        r['dist_ma200'] = round((cl / ma200.iloc[idx] - 1) * 100, 2)
        r['atr_pct'] = round(float(atr.iloc[idx]) / cl * 100, 2)
        r['gap_open_pct'] = round((o.iloc[idx] / c.iloc[idx-1] - 1) * 100, 2)
        j = spx.index.searchsorted(day)
        if j < len(spx) and spx.index[j] == day and j > 20 and not np.isnan(spx_ma200.iloc[j]):
            r['spx_vs_ma200'] = round((spx.iloc[j] / spx_ma200.iloc[j] - 1) * 100, 2)
            r['spx_ret5'] = round((spx.iloc[j] / spx.iloc[j-5] - 1) * 100, 2)
            r['spx_ret20'] = round((spx.iloc[j] / spx.iloc[j-20] - 1) * 100, 2)
        vj = vix.index.searchsorted(day)
        if vj < len(vix) and vix.index[vj] == day:
            r['vix'] = round(float(vix.iloc[vj]), 1); r['vix_chg5'] = round(float(vix.iloc[vj] - vix.iloc[vj-5]), 1)
        if idx + 1 < len(df):
            entry = df.index[idx + 1]; e = float(c.iloc[idx + 1])
            r['entry_day'] = entry
            for n in (1, 3, 5, 10):
                r[f'r{n}'] = round((c.iloc[idx+1+n] / e - 1) * 100, 2) if idx + 1 + n < len(df) else np.nan
                r[f'b{n}'] = round(spx_fwd(entry, n), 2)
            if idx + 11 < len(df):
                r['mdd10'] = round((c.iloc[idx+2:idx+12].min() / e - 1) * 100, 2)
                r['max10'] = round((c.iloc[idx+2:idx+12].max() / e - 1) * 100, 2)
        rows.append(r)
    if k % 100 == 0: print(k, len(rows), flush=True)

d = pd.DataFrame(rows)
# 当天信号总数（大盘整体超卖 = 扎堆）
d['n_base_day'] = d.groupby('signal_day')['ticker'].transform('count')
d['n_graded_day'] = d.groupby('signal_day')['grade'].transform(lambda s: (s != '').sum())
d['sector_n_day'] = d.groupby(['signal_day', 'sector'])['ticker'].transform('count')
print('signals', len(d), flush=True)

# 财报日
cache_f = os.path.join(OUT, 'earnings_cache.json')
cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
need = [t for t in d.ticker.unique() if t not in cache]
def get_e(t):
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=40)
        return t, sorted({str(x.date()) for x in e.index})
    except Exception:
        return t, []
print('earnings for', len(need), flush=True)
with ThreadPoolExecutor(8) as ex:
    for t, ds_ in ex.map(get_e, need): cache[t] = ds_
json.dump(cache, open(cache_f, 'w'))
def earn_feats(row):
    ed = pd.to_datetime(cache.get(row.ticker, []))
    if len(ed) == 0: return pd.Series([np.nan, np.nan, np.nan], index=['days_to_earn', 'days_since_earn', 'earn_in_hold'])
    day = row.signal_day
    fut = ed[ed > day]; past = ed[ed <= day]
    dte = (fut.min() - day).days if len(fut) else np.nan
    dse = (day - past.max()).days if len(past) else np.nan
    return pd.Series([dte, dse, int(dte <= 16) if dte == dte else np.nan], index=['days_to_earn', 'days_since_earn', 'earn_in_hold'])
d = pd.concat([d, d.apply(earn_feats, axis=1)], axis=1)
# 是否会被实盘去重过滤（14天内同一只已被推荐过）
d = d.sort_values('signal_day')
g = d[d.grade != ''].copy()
g['prev'] = g.groupby('ticker')['signal_day'].shift(1)
d['live_like'] = 0
d.loc[g.index, 'live_like'] = ((g['signal_day'] - g['prev']).dt.days.fillna(999) > 14).astype(int)
d['oos2026'] = (d.signal_day >= '2026-01-01').astype(int)
d.to_csv(os.path.join(OUT, 'signals_dataset.csv'), index=False, encoding='utf-8-sig')
print('done', d.shape, flush=True)
