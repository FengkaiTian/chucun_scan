"""出场方式模拟（止损/止盈/持有期）+ 每日选股方式对比。入场=信号次日收盘；
止损/止盈用日内最低/最高价判断，跳空低开按开盘价成交（较真实）。"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, yfinance as yf
d = pd.read_csv('signals_dataset.csv', encoding='utf-8-sig', parse_dates=['signal_day', 'entry_day'], keep_default_na=False, na_values=[''])
d['grade'] = d['grade'].fillna('')
d = d[d.entry_day.notna() & d.r10.notna()].reset_index(drop=True)
tk = sorted(d.ticker.unique())
px = yf.download(tk, period='5y', group_by='ticker', auto_adjust=True, threads=True, progress=False)
P = {t: px[t].dropna() for t in tk}
print('signals', len(d), flush=True)

def sim(row, H, stop=None, tp=None, stop_px=None):
    df = P[row.ticker]; i = df.index.searchsorted(row.entry_day)
    e = float(df['Close'].iloc[i]); sp = None
    if stop is not None: sp = e * (1 - stop / 100)
    if stop_px is not None: sp = stop_px
    tpx = e * (1 + tp / 100) if tp else None
    for k in range(1, H + 1):
        o, h, l, c = (float(df[x].iloc[i + k]) for x in ('Open', 'High', 'Low', 'Close'))
        if sp is not None and l <= sp: return (min(o, sp) / e - 1) * 100
        if tpx is not None and h >= tpx: return (max(o, tpx) / e - 1) * 100
    return (c / e - 1) * 100

# MA20*0.97 止损价（信号日）
ma20 = {}
for r in d.itertuples():
    df = P[r.ticker]; j = df.index.searchsorted(r.signal_day)
    ma20[r.Index] = float(df['Close'].iloc[max(0, j - 19):j + 1].mean()) * 0.97

variants = []
for H in (3, 5, 10): variants.append((f'固定持有 {H} 日', dict(H=H)))
for H in (5, 10):
    for s in (4, 6, 8, 10): variants.append((f'止损 -{s}% / 最长 {H} 日', dict(H=H, stop=s)))
    variants.append((f'MA20×0.97 止损（现行建议）/ 最长 {H} 日', dict(H=H, ma=True)))
    for k in (1.5, 2, 3): variants.append((f'{k}×ATR 止损 / 最长 {H} 日', dict(H=H, atrk=k)))
    for t in (4, 6, 8): variants.append((f'止盈 +{t}% / 最长 {H} 日', dict(H=H, tp=t)))
    variants.append((f'止损 -6% + 止盈 +6% / 最长 {H} 日', dict(H=H, stop=6, tp=6)))
    variants.append((f'止损 -8% + 止盈 +10% / 最长 {H} 日', dict(H=H, stop=8, tp=10)))

IS = d.signal_day < '2026-01-01'
rows = []
for name, kw in variants:
    res = []
    for r in d.itertuples():
        stop_px = ma20[r.Index] if kw.get('ma') else (None)
        stop = kw.get('stop')
        if kw.get('atrk'): stop = kw['atrk'] * r.atr_pct
        if stop_px is not None:
            ep = float(P[r.ticker]['Close'].iloc[P[r.ticker].index.searchsorted(r.entry_day)])
            if stop_px >= ep: res.append(np.nan); continue          # 入场价已在止损价下方，止损无意义
        res.append(sim(r, kw['H'], stop=stop, tp=kw.get('tp'), stop_px=stop_px))
    s = pd.Series(res)
    for lab, m in (('样本内', IS), ('2026样本外', ~IS)):
        x = s[m & s.notna()]
        rows.append(dict(方式=name, 区间=lab, n=len(x), 平均=x.mean(), 中位=x.median(), 上涨比例=(x > 0).mean() * 100,
                         大亏比例=(x <= -5).mean() * 100, 最差=x.min()))
out = pd.DataFrame(rows).round(2)
out.to_csv('explore_exits.csv', index=False, encoding='utf-8-sig')
print(out.to_string(index=False), flush=True)

# ---------- 每日选股方式 ----------
print('\n=== 选股方式对比（T+5，次日入场） ===')
b = d.sort_values('signal_day').copy()
keep = []; last = {}
for r in b.itertuples():
    if r.ticker in last and (r.signal_day - last[r.ticker]).days <= 14: continue
    last[r.ticker] = r.signal_day; keep.append(r.Index)
dd = b.loc[keep].copy()
dd['ex5'] = dd.r5 - dd.b5; dd['crash'] = ((dd.r5 <= -5) | (dd.r10 <= -8))
def ev(name, x):
    rr = []
    for lab, m in (('样本内', x.signal_day < '2026-01-01'), ('2026', x.signal_day >= '2026-01-01')):
        z = x[m]
        if len(z) < 10: rr.append(f'{lab}: n={len(z)}'); continue
        g = z.groupby('signal_day')['ex5'].mean()
        t = g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 2 else np.nan
        rr.append(f'{lab}: n={len(z)} 上涨{(z.r5>0).mean()*100:.0f}% 均{z.r5.mean():+.2f}% 超额{z.ex5.mean():+.2f}%(按日t={t:.1f}) 大跌{z.crash.mean()*100:.0f}%')
    print(f'{name:<26}', ' | '.join(rr))
ev('全部基础信号(去重)', dd)
ev('仅验证组合(现行推荐)', dd[dd.grade != ''])
ev('SSS 级', dd[dd.grade == 'SSS'])
ev('验证组合 且 无财报', dd[(dd.grade != '') & (dd.earn_in_hold != 1)])
ev('基础信号 且 无财报', dd[dd.earn_in_hold != 1])
ev('基础信号 且 当天≥5只', dd[dd.n_base_day >= 5])
ev('基础信号 无财报 且 ≥5只', dd[(dd.earn_in_hold != 1) & (dd.n_base_day >= 5)])
ev('验证组合 无财报 且 ≥5只', dd[(dd.grade != '') & (dd.earn_in_hold != 1) & (dd.n_base_day >= 5)])
ev('基础信号 ATR<4', dd[dd.atr_pct < 4])
top = dd[dd.grade != ''].sort_values(['signal_day', 'combo_wr5'], ascending=[True, False]).groupby('signal_day').head(1)
ev('每天只取胜率最高1只', top)
top3 = dd[dd.grade != ''].sort_values(['signal_day', 'combo_wr5'], ascending=[True, False]).groupby('signal_day').head(3)
ev('每天取胜率最高3只', top3)
