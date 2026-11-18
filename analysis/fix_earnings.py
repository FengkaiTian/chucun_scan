import json, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, yfinance as yf
from concurrent.futures import ThreadPoolExecutor
cache = json.load(open('earnings_cache.json'))
def get_e(t):
    for _ in range(3):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=40)
            r = sorted({str(x.date()) for x in e.index})
            if r: return t, r
        except Exception:
            pass
        time.sleep(1.5)
    return t, []
empty = [t for t, v in cache.items() if not v]
print('retry', len(empty), flush=True)
with ThreadPoolExecutor(3) as ex:
    for t, r in ex.map(get_e, empty): cache[t] = r
json.dump(cache, open('earnings_cache.json', 'w'))
print('still empty', sum(1 for v in cache.values() if not v), flush=True)
d = pd.read_csv('signals_dataset.csv', encoding='utf-8-sig', parse_dates=['signal_day'], keep_default_na=False, na_values=[''])
def f(row):
    ed = pd.to_datetime(cache.get(row.ticker, []))
    if len(ed) == 0: return pd.Series([np.nan]*3, index=['days_to_earn','days_since_earn','earn_in_hold'])
    day = row.signal_day; fut = ed[ed > day]; past = ed[ed <= day]
    dte = (fut.min() - day).days if len(fut) else np.nan
    dse = (day - past.max()).days if len(past) else np.nan
    return pd.Series([dte, dse, int(dte <= 16) if dte == dte else np.nan], index=['days_to_earn','days_since_earn','earn_in_hold'])
d = d.drop(columns=['days_to_earn','days_since_earn','earn_in_hold'])
d = pd.concat([d, d.apply(f, axis=1)], axis=1)
d.to_csv('signals_dataset.csv', index=False, encoding='utf-8-sig')
print('coverage', round(d.days_to_earn.notna().mean(), 2))
