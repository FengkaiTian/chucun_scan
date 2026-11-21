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

COST = 0.10; S26 = pd.Timestamp('2026-01-01')
def rsi_w(c, n):            # Wilder（和 daily_scan 一样）
    d = c.diff(); g = d.clip(lower=0).ewm(com=n - 1, min_periods=n).mean(); l = (-d.clip(upper=0)).ewm(com=n - 1, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)
def rsi_c(c, n):            # Connors 用的版本
    d = c.diff(); g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)
ma200 = C.rolling(200).mean(); ma5 = C.rolling(5).mean(); r2 = rsi_c(C, 2)
ibs = (C - L) / (H - L).replace(0, np.nan)
cv, m5v = C.values, ma5.values

def run(entry, lag, exit_kind, maxd):
    E = (entry & MEM).values; out = []
    first = dates.searchsorted(S26)
    for j in range(E.shape[1]):
        busy = -1
        for i in np.where(E[:, j])[0]:
            if i < first: continue
            b = i + lag
            if b <= busy or b >= len(dates) or not np.isfinite(cv[b, j]): continue
            x = None; status = '平仓'
            for k in range(1, maxd + 1):
                t = b + k
                if t >= len(dates): break
                if not np.isfinite(cv[t, j]): break
                if exit_kind == 'ma5' and cv[t, j] > m5v[t, j]: x = t; break
                if exit_kind == 'up' and cv[t, j] > cv[t - 1, j]: x = t; break
                if k == maxd: x = t
            if x is None:                                       # 还在持有：按最新收盘算
                lastv = np.where(np.isfinite(cv[b:, j]))[0]
                x = b + lastv[-1]; status = '持有中'
            out.append((dates[i], C.columns[j], (cv[x, j] / cv[b, j] - 1) * 100 - COST, x - b, status))
            busy = x if status == '平仓' else len(dates)
    return pd.DataFrame(out, columns=['day', 'ticker', 'net', 'hold', 'status'])

# ---- 主推：从价格重算 ----
rsi14 = rsi_w(C, 14)
e12 = C.ewm(span=12, adjust=False).mean(); e26 = C.ewm(span=26, adjust=False).mean(); dif = e12 - e26
hist = dif - dif.ewm(span=9, adjust=False).mean()
base = (hist < 0) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2)) & (rsi14.shift(1) < 30) & (rsi14 > rsi14.shift(1))
gapdn = O < C.shift(1)
kept = pd.DataFrame(False, index=dates, columns=C.columns)
bv = base.values
for j in range(bv.shape[1]):
    lastd = None
    for i in np.where(bv[:, j])[0]:
        d = dates[i]
        if lastd is None or (d - lastd).days > 5:
            lastd = d; kept.iat[i, j] = True
main = kept & gapdn & MEM
gap_pct = (O / C.shift(1) - 1) * 100
deep = (gap_pct <= -1) | (O < L.shift(1))
nday = main.sum(1)
fear = pd.DataFrame(np.repeat(((vix >= 25) | (nday >= 5)).values[:, None], C.shape[1], 1), index=dates, columns=C.columns)
mainAB = main & (deep | fear)
mainA = main & deep & fear

dn3 = (C < C.shift(1)) & (C.shift(1) < C.shift(2)) & (C.shift(2) < C.shift(3)) & (C / C.shift(3) - 1 < -0.05)
r5 = (C / C.shift(5) - 1).where(MEM)
wk_end = pd.Series(dates, index=dates).groupby(dates.to_period('W')).transform('max') == dates
rank = r5.rank(axis=1, method='first')
wk = pd.DataFrame((rank <= 10).values & wk_end.values[:, None], index=dates, columns=C.columns)

res = {
    'Connors RSI2<10': run((C > ma200) & (r2 < 10), 0, 'ma5', 10),
    'Connors RSI2<5': run((C > ma200) & (r2 < 5), 0, 'ma5', 10),
    'IBS<0.15': run((C > ma200) & (ibs < 0.15), 0, 'up', 5),
    '连跌3天>5%': run((C > ma200) & dn3, 0, 'ma5', 10),
    '每周跌最多10只': run(wk, 0, 'fix', 5),
    '主推 全部': run(main, 1, 'fix', 10),
    '主推 A+B': run(mainAB, 1, 'fix', 10),
    '主推 A': run(mainA, 1, 'fix', 10),
    '主推 A+B 站上5日线卖': run(mainAB, 1, 'ma5', 10),
    '随便买成分股拿10天（对照）': run(MEM & (pd.DataFrame(np.random.default_rng(0).random(C.shape), index=dates, columns=C.columns) < 0.01), 0, 'fix', 10),
}
rows = []; months = {}
for k, r in res.items():
    d = r[r.status == '平仓']; o = r[r.status == '持有中']
    rows.append({'方法': k, '已平仓笔数': len(d), '每笔%': round(d.net.mean(), 2), '中位数%': round(d.net.median(), 2),
                 '赚钱%': round((d.net > 0).mean() * 100), '平均持有天': round(d.hold.mean(), 1),
                 '每天%': round(d.net.mean() / d.hold.mean(), 3), '最差5%笔': round(np.percentile(d.net, 5), 1),
                 '赚钱月份': f'{(d.groupby(d.day.dt.month).net.mean() > 0).sum()}/{d.day.dt.month.nunique()}',
                 '持有中笔数': len(o), '持有中浮动%': round(o.net.mean(), 2) if len(o) else None})
    months[k] = d.groupby(d.day.dt.month).net.mean().round(2)
T = pd.DataFrame(rows); M = pd.DataFrame(months).T
T.to_csv(os.path.join(HERE, 'other_methods_2026.csv'), index=False, encoding='utf-8-sig')
M.to_csv(os.path.join(HERE, 'other_methods_2026_monthly.csv'), encoding='utf-8-sig')
pd.set_option('display.width', 260)
print(T.to_string(index=False)); print('\n每月每笔平均%（按信号月份）'); print(M.to_string())
print('\n主推 2026 信号数（重算）', int(main[dates >= S26].values.sum()))
