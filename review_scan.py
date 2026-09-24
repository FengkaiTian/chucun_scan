"""v5 实盘复盘：git 历史里每一次推荐 -> 实际 T+1/3/5/10 收益，对比 SPY；区分盘中/收盘后扫描。"""
import math, re, subprocess
import pandas as pd, yfinance as yf

git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, errors='ignore').stdout

latest = {}  # 扫描日 -> (commit, 是否美股盘中生成)
for line in git('log', '--format=%h|%ai|%s').splitlines():
    sha, ai, s = line.split('|', 2)
    m = re.search(r'scan: (\d{4}-\d{2}-\d{2})', s)
    if m and m[1] not in latest:
        et = pd.Timestamp(ai).tz_convert('America/New_York')
        latest[m[1]] = (sha, et.weekday() < 5 and 9.5 <= et.hour + et.minute / 60 < 16)
df = pd.DataFrame([(d, t, intra) for d, (sha, intra) in latest.items()
                   for t in dict.fromkeys(re.findall(r'<td[^>]*><b>([A-Z][A-Z0-9\-]{0,5})</b>\s*\$',
                                                     git('show', f'{sha}:index.html')))],
                  columns=['date', 'ticker', 'intraday'])

px = yf.download(sorted(set(df.ticker)) + ['SPY'], start=pd.Timestamp(min(df.date)) - pd.Timedelta(days=5),
                 auto_adjust=True, progress=False)['Close']


def ret(t, d, n):  # 与 daily_scan.get_ret 口径一致：入场 = 扫描日当天或之后第一个收盘价
    if t not in px:
        return math.nan
    c = px[t].dropna()
    f = c.index[c.index >= pd.Timestamp(d)]
    return (c[f[n]] / c[f[0]] - 1) * 100 if len(f) > n else math.nan


for n in (1, 3, 5, 10):
    df[f'r{n}'] = [ret(t, d, n) for d, t in zip(df.date, df.ticker)]
    df[f'x{n}'] = df[f'r{n}'] - [ret('SPY', d, n) for d in df.date]


def wilson(k, n, z=1.96):
    p = k / n
    mid, half = (p + z * z / 2 / n) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / (1 + z * z / n)
    return 100 * (mid - half), 100 * (mid + half)


for name, g in [('全部', df), ('盘中扫描(K线未收盘)', df[df.intraday]), ('收盘后扫描', df[~df.intraday])]:
    print(f'\n== {name}: {len(g)} 次推荐, {g.date.nunique()} 个扫描日 ==')
    for n in (1, 3, 5, 10):
        r, x = g[f'r{n}'].dropna(), g[f'x{n}'].dropna()
        if len(r):
            lo, hi = wilson((r > 0).sum(), len(r))
            print(f'T+{n:<2} n={len(r):3d}  上涨 {100 * (r > 0).mean():5.1f}% [{lo:.0f}-{hi:.0f}]  均值 {r.mean():+.2f}%  '
                  f'中位 {r.median():+.2f}%  跑赢SPY {100 * (x > 0).mean():5.1f}%  平均超额 {x.mean():+.2f}%')
