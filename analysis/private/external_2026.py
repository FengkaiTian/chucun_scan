"""网上公开的短线/中线选股法，只测 2026-01-01 到最新收盘（不含我们自己的主推）。
数据和口径同 other_methods_2026.py：当天是标普500 成分股才入场；收盘价买卖；扣 0.10% 来回成本；
同一只股票持仓期间不重复开仓；还没到卖出条件的按最新收盘价算浮动盈亏、单独列出。
参数一律用原作者/常见公开版本，不针对 2026 调整。

均值回归（抄底）类：
  1 Connors RSI(2)<10      收盘>200日线，RSI2<10；收盘站上 5 日线卖（最多 10 天）
  2 Connors RSI(2)<5       同上，RSI2<5
  3 Connors R3             收盘>200日线，RSI2 连降 3 天、首日<60、当天<10；RSI2>70 卖
  4 Connors 累积 RSI       收盘>200日线，两天 RSI2 之和<35；RSI2>65 卖
  5 Double 7s              收盘>200日线，收盘创 7 日新低；收盘创 7 日新高卖（最多 20 天）
  6 布林带下轨             收盘>200日线，收盘跌破布林带(20,2)下轨；收盘回到中轨卖（最多 20 天）
  7 IBS<0.15               收盘>200日线，IBS<0.15；第一次收涨卖（最多 5 天）
  8 连跌 3 天累计>5%       收盘>200日线；收盘站上 5 日线卖（最多 10 天）
  9 Williams %R(2)<-90     收盘>200日线；%R>-30 卖（最多 10 天）
 10 每周输家               每周最后一个交易日买 5 日跌幅最大的 10 只，拿 5 天
 11 每月输家               每月最后一个交易日买 21 日跌幅最大的 10 只，拿 21 天
趋势/动量类：
 12 12-1 动量              每月最后一个交易日买过去 12 个月（剔除最近 1 个月）涨最多的 10 只，拿 21 天
 13 52 周新高              收盘创 252 日新高；拿 20 天
 14 均线金叉               50 日线上穿 200 日线；拿 20 天
 15 MACD 零轴上金叉        MACD 线上穿信号线且两者>0；拿 10 天
 16 放量突破               收盘创 20 日新高且成交量>20 日均量 2 倍；拿 10 天
对照：
 17 随便买成分股           每天随机 1% 的成分股，拿 10 天
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'panel_ext.pkl')
if not os.path.exists(CACHE):
    exec(open(os.path.join(HERE, '_load2026.py'), encoding='utf-8').read())
    V = pd.read_parquet(os.path.join(P, 'V.parquet'))
    pd.to_pickle({'O': O, 'H': H, 'L': L, 'C': C, 'MEM': MEM, 'V': V}, CACHE)
d = pd.read_pickle(CACHE); O, H, L, C, MEM, V = (d[k] for k in ('O', 'H', 'L', 'C', 'MEM', 'V'))
V = V.reindex(index=C.index, columns=C.columns)     # 新补的几天没有成交量，放量突破在那几天不出信号
dates = C.index; COST = 0.10; S26 = pd.Timestamp('2026-01-01'); first = dates.searchsorted(S26)
print('数据到', dates[-1].date())

def rsi_c(c, n):
    dd = c.diff(); g = dd.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); l = (-dd.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)
ma200 = C.rolling(200).mean(); ma50 = C.rolling(50).mean(); ma20 = C.rolling(20).mean(); ma5 = C.rolling(5).mean()
sd20 = C.rolling(20).std(); r2 = rsi_c(C, 2); up200 = C > ma200
ibs = (C - L) / (H - L).replace(0, np.nan)
wr2 = (H.rolling(2).max() - C) / (H.rolling(2).max() - L.rolling(2).min()).replace(0, np.nan) * -100
e12 = C.ewm(span=12, adjust=False).mean(); e26 = C.ewm(span=26, adjust=False).mean(); macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
cv = C.values

def run(entry, exitm=None, maxd=10):
    """entry/exitm：布尔表。exitm 为 None 时拿满 maxd 天；否则第一次 exitm 为真的那天收盘卖，最多 maxd 天"""
    E = (entry & MEM).values; X = None if exitm is None else exitm.values; out = []
    for j in range(E.shape[1]):
        busy = -1
        for i in np.where(E[:, j])[0]:
            if i < first or i <= busy or not np.isfinite(cv[i, j]): continue
            x = None; st = '平仓'
            for k in range(1, maxd + 1):
                t = i + k
                if t >= len(dates) or not np.isfinite(cv[t, j]): break
                if (X is not None and X[t, j]) or k == maxd: x = t; break
            if x is None:
                ok = np.where(np.isfinite(cv[i:, j]))[0]; x = i + ok[-1]; st = '持有中'
            out.append((dates[i], C.columns[j], (cv[x, j] / cv[i, j] - 1) * 100 - COST, x - i, st))
            busy = x if st == '平仓' else len(dates)
    return pd.DataFrame(out, columns=['day', 'ticker', 'net', 'hold', 'status'])

def period_end(freq):
    s = pd.Series(dates, index=dates)
    return pd.Series(s.groupby(dates.to_period(freq)).transform('max').values == dates, index=dates)

def top_n(score, when, n=10, asc=True):
    rk = score.where(MEM).rank(axis=1, ascending=asc, method='first')
    return pd.DataFrame((rk <= n).values & when.values[:, None], index=dates, columns=C.columns)

r2d = r2.diff()
r3 = up200 & (r2d < 0) & (r2d.shift(1) < 0) & (r2d.shift(2) < 0) & (r2.shift(2) < 60) & (r2 < 10)
dn3 = (C < C.shift(1)) & (C.shift(1) < C.shift(2)) & (C.shift(2) < C.shift(3)) & (C / C.shift(3) - 1 < -0.05)
cross = lambda a, b: (a > b) & (a.shift(1) <= b.shift(1))
wk, mo = period_end('W'), period_end('M')
rng = np.random.default_rng(0)

M = {
    '1 Connors RSI2<10': run(up200 & (r2 < 10), C > ma5, 10),
    '2 Connors RSI2<5': run(up200 & (r2 < 5), C > ma5, 10),
    '3 Connors R3': run(r3, r2 > 70, 20),
    '4 Connors 累积RSI': run(up200 & (r2 + r2.shift(1) < 35), r2 > 65, 20),
    '5 Double 7s': run(up200 & (C <= C.rolling(7).min()), C >= C.rolling(7).max(), 20),
    '6 布林带下轨': run(up200 & (C < ma20 - 2 * sd20), C >= ma20, 20),
    '7 IBS<0.15': run(up200 & (ibs < 0.15), C > C.shift(1), 5),
    '8 连跌3天>5%': run(up200 & dn3, C > ma5, 10),
    '9 Williams %R2<-90': run(up200 & (wr2 < -90), wr2 > -30, 10),
    '10 每周输家': run(top_n(C / C.shift(5), wk), None, 5),
    '11 每月输家': run(top_n(C / C.shift(21), mo), None, 21),
    '12 12-1动量': run(top_n(C.shift(21) / C.shift(252), mo, asc=False), None, 21),
    '13 52周新高': run(C >= C.rolling(252).max(), None, 20),
    '14 均线金叉': run(cross(ma50, ma200), None, 20),
    '15 MACD零轴上金叉': run(cross(macd, sig) & (macd > 0) & (sig > 0), None, 10),
    '16 放量突破': run((C >= C.rolling(20).max()) & (V > 2 * V.rolling(20).mean()), None, 10),
    '17 随便买（对照）': run(pd.DataFrame(rng.random(C.shape) < 0.01, index=dates, columns=C.columns), None, 10),
}
rows, months = [], {}
for k, r in M.items():
    dd = r[r.status == '平仓']; o = r[r.status == '持有中']
    rows.append({'方法': k, '已平仓': len(dd), '每笔%': round(dd.net.mean(), 2), '中位数%': round(dd.net.median(), 2),
                 '赚钱%': round((dd.net > 0).mean() * 100), '持有天': round(dd.hold.mean(), 1),
                 '每天%': round(dd.net.mean() / dd.hold.mean(), 3), '最差5%': round(np.percentile(dd.net, 5), 1),
                 '赚钱月': f'{(dd.groupby(dd.day.dt.month).net.mean() > 0).sum()}/{dd.day.dt.month.nunique()}',
                 '持有中': len(o), '浮动%': round(o.net.mean(), 2) if len(o) else None})
    months[k] = dd.groupby(dd.day.dt.month).net.mean().round(2)
T = pd.DataFrame(rows).sort_values('每笔%', ascending=False)
T.to_csv(os.path.join(HERE, 'external_2026.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(months).T.to_csv(os.path.join(HERE, 'external_2026_monthly.csv'), encoding='utf-8-sig')
pd.set_option('display.width', 260); print(T.to_string(index=False))
print(pd.DataFrame(months).T.to_string())
