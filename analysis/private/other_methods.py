"""网上常见的高胜率短线选股法，放到我们同一份数据上公平比较（2026-11-20）。
数据：无幸存者偏差的标普500 历史成分股（只在当天是成分股时才允许入场），2000-01 ~ 2026-11。
统一口径：收盘价买、收盘价卖，扣 0.10% 来回成本；赚钱 = 扣费后 > 0；同一只股票持仓期间不重复开仓。
方法：
  A Connors RSI(2)：收盘 > 200 日线，RSI2 < 10，当天收盘买；收盘站上 5 日线就卖（最多 10 天）
  B Connors RSI(2) 严格版：RSI2 < 5，其余同上
  C IBS 均值回归：收盘 > 200 日线，IBS=(收-低)/(高-低) < 0.15，当天收盘买；第一次收涨就卖（最多 5 天）
  D 一周输家反转：每周最后一个交易日，买过去 5 天跌最多的 10 只，拿 5 天
  E 连跌 3 天 + 200 日线上：收盘 > 200 日线，连跌 3 天且累计跌 > 5%，收盘买；站上 5 日线卖（最多 10 天）
  M 我们的主推：信号次日收盘买，拿 10 天
  M5 主推 + Connors 卖法：信号次日收盘买，站上 5 日线就卖（最多 10 天）
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, '..', 'rebuild', 'data2000', 'gh_johnbrick', 'panel')
O, H, L, C = (pd.read_parquet(os.path.join(P, f'{k}.parquet')) for k in 'OHLC')
md = pd.read_parquet(os.path.join(P, 'member_days.parquet'))
MEM = pd.crosstab(md.date, md.ticker).reindex(index=C.index, columns=C.columns, fill_value=0).astype(bool)
dates = C.index; COST = 0.10

def rsi(c, n):
    d = c.diff(); g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)

ma200 = C.rolling(200).mean(); ma5 = C.rolling(5).mean()
r2 = rsi(C, 2)
ibs = (C - L) / (H - L).replace(0, np.nan)
cv, m5v = C.values, ma5.values

def run(entry, lag, exit_kind, maxd):
    """entry: 信号布尔表；lag: 0=当天收盘买，1=次日收盘买"""
    E = (entry & MEM).values
    out = []
    for j in range(E.shape[1]):
        idx = np.where(E[:, j])[0]; busy = -1
        for i in idx:
            b = i + lag
            if b <= busy or b + 1 >= len(dates) or not np.isfinite(cv[b, j]):
                continue
            x = None
            for k in range(1, maxd + 1):
                t = b + k
                if t >= len(dates) or not np.isfinite(cv[t, j]):
                    break
                if exit_kind == 'ma5' and cv[t, j] > m5v[t, j]: x = t; break
                if exit_kind == 'up' and cv[t, j] > cv[t - 1, j]: x = t; break
                x = t if k == maxd else None
            if x is None:
                continue
            out.append((dates[i], C.columns[j], (cv[x, j] / cv[b, j] - 1) * 100 - COST, x - b))
            busy = x
    return pd.DataFrame(out, columns=['day', 'ticker', 'net', 'hold'])

res = {}
res['A Connors RSI2<10'] = run((C > ma200) & (r2 < 10), 0, 'ma5', 10)
res['B Connors RSI2<5'] = run((C > ma200) & (r2 < 5), 0, 'ma5', 10)
res['C IBS<0.15'] = run((C > ma200) & (ibs < 0.15), 0, 'up', 5)
dn3 = (C < C.shift(1)) & (C.shift(1) < C.shift(2)) & (C.shift(2) < C.shift(3)) & (C / C.shift(3) - 1 < -0.05)
res['E 连跌3天>5%'] = run((C > ma200) & dn3, 0, 'ma5', 10)
# D 每周输家
r5 = (C / C.shift(5) - 1).where(MEM)
wk_end = pd.Series(dates, index=dates).groupby(dates.to_period('W')).transform('max') == dates
rank = r5.rank(axis=1, method='first')
res['D 每周跌最多10只'] = run(pd.DataFrame((rank <= 10).values & wk_end.values[:, None], index=dates, columns=C.columns), 0, 'fix', 5)
# 主推
s = pd.read_parquet(os.path.join(HERE, 'main_rule_signals.parquet'))
mm = pd.DataFrame(False, index=dates, columns=C.columns)
for d, t in zip(s.signal_day, s.ticker):
    if t in mm.columns and d in mm.index: mm.at[d, t] = True
deep = (s.gap <= -1) | s.open_lt_pl; fear = (s.vix >= 25) | (s.nday >= 5)
ab = s[deep | fear]
mab = pd.DataFrame(False, index=dates, columns=C.columns)
for d, t in zip(ab.signal_day, ab.ticker):
    if t in mab.columns and d in mab.index: mab.at[d, t] = True
res['M 主推（10天）'] = run(mm, 1, 'fix', 10)
res['M5 主推+站上5日线卖'] = run(mm, 1, 'ma5', 10)
res['MAB 主推A+B（10天）'] = run(mab, 1, 'fix', 10)
res['MAB5 主推A+B+站上5日线卖'] = run(mab, 1, 'ma5', 10)

NY = (pd.Timestamp('2026-01-01') - dates[0]).days / 365.25
rows = []
for k, r in res.items():
    tr = r[r.day < '2026-01-01']; te = r[r.day >= '2026-01-01']
    yr = tr.groupby(tr.day.dt.year).net.mean()
    rows.append({'方法': k, '每年笔数': round(len(tr) / NY), '每笔%': round(tr.net.mean(), 2), '赚钱%': round((tr.net > 0).mean() * 100),
                 '平均持有天': round(tr.hold.mean(), 1), '每天%': round(tr.net.mean() / tr.hold.mean(), 3),
                 '最差5%笔': round(np.percentile(tr.net, 5), 1), '赚钱年份%': round((yr > 0).mean() * 100),
                 '2026笔数': len(te), '2026每笔%': round(te.net.mean(), 2) if len(te) else None,
                 '2026赚钱%': round((te.net > 0).mean() * 100) if len(te) else None})
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, 'other_methods.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); print(T.to_string(index=False))
