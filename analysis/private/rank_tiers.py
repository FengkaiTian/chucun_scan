"""主推内部分档：个股深低开（低开≥1% 或 开盘<昨日最低）+ 市场恐慌（VIX≥25 或 同日≥5 只）各 1 分。"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
s = pd.read_parquet(os.path.join(HERE, 'main_rule_signals.parquet'))
deep = (s.gap <= -1) | s.open_lt_pl
fear = (s.vix >= 25) | (s.nday >= 5)
s['score'] = deep.astype(int) + fear.astype(int)
tr = s.signal_day < '2025-12-01'; te = s.year == 2026
w = 0.5 ** ((2026 - (s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366)) / 10)
ny = (pd.Timestamp('2025-12-01') - pd.Timestamp('2000-01-01')).days / 365.25
print('分 | 训练每年 | 训练净% | 训练胜率% | 正年份% | 2026笔数 | 2026净% | 2026胜率%')
for sc in (2, 1, 0):
    m = (s.score == sc) & s.net.notna(); q = m & tr; ww = w[q]
    yr = s[q].groupby(s.year[q]).net.mean()
    e = s[m & te]
    print(f'{sc} | {q.sum()/ny:.0f} | {(ww*s.net[q]).sum()/ww.sum():+.2f} | {(ww*s.win[q]).sum()/ww.sum()*100:.0f} | {(yr>0).mean()*100:.0f} | {len(e)} | {e.net.mean():+.2f} | {e.win.mean()*100:.0f}')
