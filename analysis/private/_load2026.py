"""所有方法只看 2026-01-01 到最新收盘（2026-11-21 跑）。规则和 other_methods.py 完全一样，不再挑参数。
数据：历史面板到 11-17，之后用 yfinance 补到最新（只补 11-17 当天的成分股）。
主推在这里从价格直接重算（基础信号 + 低开 + 5 天去重），分档用当天 VIX 和当天主推个数。
已平仓的算成绩；还没到卖出日的按最新收盘价算浮动盈亏，单独列出。"""
import os, sys, numpy as np, pandas as pd, yfinance as yf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
P = os.path.join(HERE, '..', 'rebuild', 'data2000', 'gh_johnbrick', 'panel')
O, H, L, C = (pd.read_parquet(os.path.join(P, f'{k}.parquet')) for k in 'OHLC')
md = pd.read_parquet(os.path.join(P, 'member_days.parquet'))
MEM = pd.crosstab(md.date, md.ticker).reindex(index=C.index, columns=C.columns, fill_value=0).astype(bool)

# ---- 补 11-17 之后的日线 ----
last = C.index[-1]; mem_last = [t for t in C.columns if MEM.at[last, t]]
ysym = {t: t.replace('.', '-') for t in mem_last}
raw = yf.download(list(ysym.values()), start=(last - pd.Timedelta(days=10)).strftime('%Y-%m-%d'),
                  auto_adjust=True, group_by='ticker', progress=False, threads=True)
new_days = [d for d in raw.index if d.normalize() > last]
import zoneinfo, datetime as dt
ny = dt.datetime.now(zoneinfo.ZoneInfo('America/New_York'))
if new_days and new_days[-1].date() == ny.date() and (ny.hour, ny.minute) < (16, 30):
    new_days = new_days[:-1]                                    # 当天没收盘的半截线不要
add = {}
for k, F in (('Open', O), ('High', H), ('Low', L), ('Close', C)):
    blk = pd.DataFrame(np.nan, index=pd.DatetimeIndex([d.normalize() for d in new_days]), columns=C.columns)
    for t, y in ysym.items():
        try:
            blk[t] = raw[y][k].reindex(new_days).values
        except Exception:
            pass
    add[k] = blk
# 价格接缝检查：yfinance 和面板在重叠日应一致（复权方式可能不同，按重叠日比例缩放）
for t, y in ysym.items():
    try:
        ov = raw[y]['Close'].reindex([d for d in raw.index if d.normalize() <= last])
        ov.index = ov.index.normalize(); r = (C[t].reindex(ov.index) / ov).dropna()
        if len(r):
            f = float(r.iloc[-1])
            for k in add: add[k][t] *= f
    except Exception:
        pass
O = pd.concat([O, add['Open']]); H = pd.concat([H, add['High']]); L = pd.concat([L, add['Low']]); C = pd.concat([C, add['Close']])
MEM = pd.concat([MEM, pd.DataFrame([MEM.loc[last].values] * len(new_days), index=add['Close'].index, columns=C.columns)])
dates = C.index; END = dates[-1]
print('数据到', END.date(), '补了', [d.date() for d in add['Close'].index])

vix = yf.download('^VIX', start='2025-06-01', auto_adjust=True, progress=False)['Close'].squeeze()
vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize(); vix = vix.reindex(dates).ffill()

