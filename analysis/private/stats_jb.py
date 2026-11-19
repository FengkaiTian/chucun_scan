"""汇总 jb_signals.parquet：基础信号 / 星级 / SSS，按时期。只留本机。"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[(s.vix < 30) & s.r5.notna() & s.x5.notna()].sort_values(['ticker', 'signal_day'])
# 同一股票 14 个日历天内只算第一次（和扫描器一致）
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    ok = t not in last or (d - last[t]).days > 14
    keep.append(ok)
    if ok: last[t] = d
s = s[np.array(keep)]
PER = [('2000–09', 2000, 2009), ('2010–19', 2010, 2019), ('2020–24', 2020, 2024), ('2025–26', 2025, 2026)]


def ct(y, g):
    y = np.asarray(y, float); m = y.mean(); e = pd.Series(y - m).groupby(np.asarray(g)).sum().values
    se = np.sqrt((e ** 2).sum()) / len(y); return m / se if se > 0 else np.nan


def row(g, name):
    out = {'组': name}
    for lab, a, b in PER:
        x = g[(g.year >= a) & (g.year <= b)]
        if len(x) < 10: out[lab] = f'{len(x)}笔'; continue
        yrs = b - a + 1 if b < 2026 else 1 + 320 / 365
        out[lab] = (f'{len(x)/yrs:.0f}/年 胜{((x.r5 - 0.10) > 0).mean()*100:.0f}% '
                    f'净{(x.r5 - 0.10).mean():+.2f} 超{x.x5.mean():+.2f}(t{ct(x.x5, x.signal_day):.1f})')
    return out


res = [row(s, '基础信号（全部）')]
o = s[s.has_open]
for t in ['★★★', '★★', '★', '参考']:
    res.append(row(o[o.tier == t], t))
res.append(row(s[s.sss], 'SSS（任一）'))
r = pd.DataFrame(res).set_index('组')
pd.set_option('display.width', 250)
txt = '5 日持有，次日收盘入场，扣 0.10%；超 = 相对同期 SPY 含分红；星级只算有开盘价的信号\n' + r.to_string()
print(txt)
open(os.path.join(HERE, 'jb_result.txt'), 'w', encoding='utf-8').write(txt)
