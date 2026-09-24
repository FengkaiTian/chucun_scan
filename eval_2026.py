"""2026 样本外检验——只允许运行一次。
成功判据（见 PREREGISTRATION.md）：2026-01-01 至今，策略日净收益 − SPY 日收益，
Newey-West 单侧检验 p < 0.05 且均值 > 0。结果文件已存在则拒绝重跑。"""
import hashlib, os, subprocess, sys
import pandas as pd, yfinance as yf
from data_pipeline import membership
from harness import backtest, stats
import strategy_final as S

OUT = 'results/test_2026.md'
if os.path.exists(OUT):
    sys.exit(f'{OUT} 已存在：测试集只允许评估一次。')

mem = membership(path=None)
members = {pd.Timestamp(d): set(g.ticker) for d, g in mem.groupby('month_end')}
tickers = sorted(set().union(*(v for d, v in members.items() if d >= pd.Timestamp('2025-12-31'))))
raw = yf.download(tickers + ['SPY'], start='2023-01-01', auto_adjust=True, progress=False, threads=True)
close = raw['Close'].dropna(axis=1, how='all')
dvol = (raw['Close'] * raw['Volume'])[close.columns]
end = close.index[-1]

r = backtest(S.target_weights, close, dvol, members, '2026-01-01', end)
bench = close['SPY'].pct_change().loc[r.index]
st = stats(r, bench)
ok = st['单侧p'] < 0.05 and st['IR'] > 0
sha = hashlib.sha256(open('strategy_final.py', 'rb').read()).hexdigest()
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
rows = '\n'.join(f'| {k} | {v:.4f} |' for k, v in st.items())
report = f"""# 2026 样本外检验结果

- 区间：{r.index[0]:%Y-%m-%d} ~ {end:%Y-%m-%d}（{len(r)} 个交易日）
- 策略文件 sha256：`{sha}`　代码提交：`{head}`
- 行情覆盖：{len(close.columns) - 1}/{len(tickers)} 只 2026 期间成分股

| 指标 | 值 |
|---|---|
{rows}

**结论：{'✅ 显著跑赢标普 500（p < 0.05）' if ok else '❌ 未能显著跑赢标普 500'}**
"""
os.makedirs('results', exist_ok=True)
open(OUT, 'w', encoding='utf-8').write(report)
pd.DataFrame({'strategy': r, 'SPY': bench}).to_csv('results/test_2026_daily.csv')
print(report)
