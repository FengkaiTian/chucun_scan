"""修正历史成分股：按维基"当前成分股"表的 Date added 截断。

问题：prepare_data.membership() 用当前名单按变更表逐月倒推，按 ticker 匹配。
公司改过代码时（如 EchoStar 以 SATS 入指，后改为 ECHO），倒推时删的是 SATS，ECHO 就一直留在名单里，
被当成 1999 年起就是成分股（前视）。
修正：当前成分股在其 Date added 之前的月份一律不算成分股；但如果这个 ticker 在变更表里出现过"被移出"
（说明曾经是成分、后来又加回），保留原来倒推的结果，不截断。
输入：data2000/membership.csv.gz、wiki_current.html、wiki_historical.html
输出：data2000/membership_v2.csv.gz、out_fix/membership_diff.csv
"""
import os, io
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data2000')
OUT = os.path.join(HERE, 'out_fix'); os.makedirs(OUT, exist_ok=True)
fix = lambda s: s.fillna('').astype(str).str.strip().str.replace('.', '-', regex=False)
flat = lambda df: [' '.join(dict.fromkeys(map(str, c if isinstance(c, tuple) else (c,)))).lower() for c in df.columns]

tables = []
for name in ('current', 'historical'):
    tables += pd.read_html(io.StringIO(open(os.path.join(D, f'wiki_{name}.html'), encoding='utf-8').read()))
cur = tables[0]
cur = pd.DataFrame({'ticker': fix(cur['Symbol']), 'added': pd.to_datetime(cur['Date added'], errors='coerce')})
chg = next(df for df in tables if any('added' in c for c in flat(df)) and any('removed' in c for c in flat(df)))
chg.columns = flat(chg)
col = lambda k, alt=('',): next(c for c in chg.columns if k in c and any(a in c for a in alt))
removed_ever = set(fix(chg[col('removed', ('ticker', 'symbol'))]))

m = pd.read_csv(os.path.join(D, 'membership.csv.gz'), parse_dates=['month_end'])
x = m.merge(cur, on='ticker', how='left')
cut = x.added.notna() & (x.month_end < x.added) & ~x.ticker.isin(removed_ever)
v2 = x[~cut][['month_end', 'ticker']]
v2.to_csv(os.path.join(D, 'membership_v2.csv.gz'), index=False)

diff = x[cut].groupby('ticker').agg(first_wrong=('month_end', 'min'), last_wrong=('month_end', 'max'),
                                    months_removed=('month_end', 'size'), date_added=('added', 'first'))
diff = diff.sort_values('months_removed', ascending=False)
diff.to_csv(os.path.join(OUT, 'membership_diff.csv'), encoding='utf-8-sig')
print(f'原名单 {len(m)} 行 → 修正后 {len(v2)} 行；删掉 {cut.sum()} 个"股票-月"，涉及 {len(diff)} 只当前成分股')
print('当前成分股里 Date added 缺失的：', cur.added.isna().sum(), '；曾被移出又加回（不截断）的：',
      cur.ticker.isin(removed_ever).sum())
print('每月成分数（修正后）：', v2.groupby('month_end').size().describe()[['min', '50%', 'max']].to_dict())
print(diff.head(15).to_string())
print('ECHO:', diff.loc['ECHO'].to_dict() if 'ECHO' in diff.index else '未改')
