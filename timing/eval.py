"""择时策略一次性独立检验：标普 500 样本前历史 1961–1997（主），2026 年仅作参考。
结果文件已存在即拒绝重跑。判据见 PREREGISTRATION_timing.md。"""
import hashlib, os, subprocess, sys
import pandas as pd
from timing.data import series
from timing.harness import backtest, criteria
import timing.strategy_final as S

OUT = 'results/timing_test.md'
if os.path.exists(OUT):
    sys.exit(f'{OUT} 已存在：独立测试只允许运行一次。')
os.makedirs('results', exist_ok=True)

old = series('1957-01-01', '1998-01-01', 'index')
r, b, w = backtest(S.target_position, old, '1961-01-01', '1997-12-31')
k, n = criteria(r, b, old['rf'].loc[r.index])

new = series('2024-01-01', pd.Timestamp.today().strftime('%Y-%m-%d'), 'spy')
r6, b6, w6 = backtest(S.target_position, new, '2026-01-01', new.index[-1])
k6, n6 = criteria(r6, b6, new['rf'].loc[r6.index])

sha = hashlib.sha256(open('timing/strategy_final.py', 'rb').read()).hexdigest()
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
tbl = lambda d: '\n'.join(f'| {a} | {v:.4f} |' for a, v in d.items())
ok = any(k.values())
report = f"""# 择时策略独立检验结果

- 策略文件 sha256：`{sha}`　代码提交：`{head}`
- 主检验：{r.index[0]:%Y-%m-%d} ~ {r.index[-1]:%Y-%m-%d}（{len(r)} 个交易日），数据来源：{old.attrs['source']}；平均仓位 {w.mean():.2f}

| 指标 | 值 |
|---|---|
{tbl(n)}

| 判据（各自 p < 0.05/3） | 结果 |
|---|---|
""" + '\n'.join(f'| {a} | {"✅" if v else "❌"} |' for a, v in k.items()) + f"""

**主结论：{'✅ 成功（至少一条判据成立）' if ok else '❌ 三条判据均未成立'}**

## 参考（不计入成败）：2026-01-01 ~ {r6.index[-1]:%Y-%m-%d}，SPY
| 指标 | 值 |
|---|---|
{tbl(n6)}
"""
open(OUT, 'w', encoding='utf-8').write(report)
pd.DataFrame({'strategy': r, 'bench': b, 'position': w}).to_csv('results/timing_test_daily.csv')
pd.DataFrame({'strategy': r6, 'bench': b6, 'position': w6}).to_csv('results/timing_2026_daily.csv')
print(report)
