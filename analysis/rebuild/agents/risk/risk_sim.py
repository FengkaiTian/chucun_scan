# -*- coding: utf-8 -*-
"""出场/过滤/组合层面实测。次日收盘入场 (entry_close @ entry_day)。
模式 A: 收盘价触发止损/MA20/时间止损 -> 次日收盘出场。
模式 B: 盘中最低价触及止损 -> 按 min(止损价, 开盘价) 成交 (跳空穿越按开盘)。
止盈两种模式都按盘中限价: 最高价触及 -> max(止盈价, 开盘价)；同日止损止盈都触及按先止损(保守)。
"""
import json, numpy as np, pandas as pd, os
D = 'C:/Users/ft7b6/Desktop/美股/chucun_scan/analysis/rebuild/data2000/'
OUT = 'C:/Users/ft7b6/Desktop/美股/chucun_scan/analysis/rebuild/agents/risk/'
H = 25
s = pd.read_parquet(D + 'signals_20y.parquet')
s = s[s.in_index & (s.vix < 30) & s.r5.notna()].sort_values(['ticker', 'signal_day']).reset_index(drop=True)
# 14 天去重
keep = []; last = {}
for i, (t, d) in enumerate(zip(s.ticker, s.signal_day)):
    if t in last and (d - last[t]).days < 14: keep.append(False); continue
    last[t] = d; keep.append(True)
print('rows', len(s), 'after dedup', sum(keep))
s = s[keep].reset_index(drop=True)
combos = [c['conditions'] for c in json.load(open('C:/Users/ft7b6/Desktop/美股/chucun_scan/validated_combos.json', encoding='utf-8')) if c['grade'] == 'SSS']
sss = np.zeros(len(s), bool)
for c in combos: sss |= s[c].all(axis=1).values
s['sss'] = sss
s['ge5'] = s.n_base_day_idx >= 5
s['sec_n'] = s.groupby(['signal_day', 'sector_now']).ticker.transform('size')

C = pd.read_parquet(D + 'ohlcv_close.parquet'); O = pd.read_parquet(D + 'ohlcv_open.parquet')
Hh = pd.read_parquet(D + 'ohlcv_high.parquet'); L = pd.read_parquet(D + 'ohlcv_low.parquet')
MA20 = C.rolling(20).mean()
dates = C.index; didx = {d: i for i, d in enumerate(dates)}
cols = {t: j for j, t in enumerate(C.columns)}
Ca, Oa, Ha, La, Ma = [x.values for x in (C, O, Hh, L, MA20)]
n = len(s)
PC = np.full((n, H + 1), np.nan); PO = PC.copy(); PH = PC.copy(); PL = PC.copy(); PM = PC.copy()
ei = np.zeros(n, int)
for k, (t, d) in enumerate(zip(s.ticker, s.entry_day)):
    i = didx[d]; j = cols[t]; ei[k] = i
    e = min(i + H + 1, len(dates))
    sl = slice(i, e); m = e - i
    PC[k, :m] = Ca[sl, j]; PO[k, :m] = Oa[sl, j]; PH[k, :m] = Ha[sl, j]; PL[k, :m] = La[sl, j]; PM[k, :m] = Ma[sl, j]
E = s.entry_close.values[:, None]
E = PC[:, :1]  # use actual close series for consistency
rc, ro, rh, rl = PC / E - 1, PO / E - 1, PH / E - 1, PL / E - 1
below_ma = PC < PM
# 复权差异检查
print('entry_close vs close diff median', np.nanmedian(np.abs(s.entry_close.values / PC[:, 0] - 1)))
# 缺失路径(退市等): 用最后有效收盘向前填
for A in (rc, ro, rh, rl):
    pass
def ffill(a):
    df = pd.DataFrame(a).T.ffill().T.values; return df
rc = ffill(rc); ro = np.where(np.isnan(ro), rc, ro); rh = np.where(np.isnan(rh), rc, rh); rl = np.where(np.isnan(rl), rc, rl)

def sim(hold, stop=None, tp=None, ma=False, tstop=None, mode='A', ma_up=False):
    """stop: 负数(比例) 或 ndarray; 返回 (ret, exit_offset)"""
    ret = rc[:, hold].copy(); off = np.full(n, hold)
    done = np.zeros(n, bool)
    st = None if stop is None else (np.broadcast_to(stop, (n,)).astype(float))
    pend = np.zeros(n, bool)  # 模式A: 昨日收盘触发，今日收盘出
    for t in range(1, hold + 1):
        act = ~done
        if True:
            # 前一日收盘触发 -> 今日收盘出
            x = act & pend
            ret[x] = rc[x, t]; off[x] = t; done |= x; act = ~done
        if st is not None and mode == 'B':
            hit = act & (rl[:, t] <= st)
            ret[hit] = np.minimum(st, ro[:, t])[hit]; off[hit] = t; done |= hit; act = ~done
        if tp is not None:
            hit = act & (rh[:, t] >= tp)
            ret[hit] = np.maximum(tp, ro[:, t])[hit]; off[hit] = t; done |= hit; act = ~done
        if ma_up:
            hit = act & (PC[:, t] > PM[:, t])
            ret[hit] = rc[hit, t]; off[hit] = t; done |= hit; act = ~done
        if t == hold: break
        trig = np.zeros(n, bool)
        if st is not None and mode == 'A': trig |= rc[:, t] <= st
        if ma: trig |= below_ma[:, t]
        if tstop is not None and t == tstop: trig |= rc[:, t] <= 0
        pend = act & trig
    return ret * 100, off

rules = {}
for h in (3, 5, 10, 20): rules[f'hold{h}'] = dict(hold=h)
for h in (5, 10):
    for sp in (-3, -4, -6, -8):
        for m in 'AB': rules[f'h{h}_sl{sp}_{m}'] = dict(hold=h, stop=sp / 100, mode=m)
    for k in (1, 1.5, 2):
        for m in 'AB': rules[f'h{h}_sl{k}ATR_{m}'] = dict(hold=h, stop=-k * s.atr_pct.values / 100, mode=m)
    for tp in (5, 8, 10): rules[f'h{h}_tp{tp}'] = dict(hold=h, tp=tp / 100)
    for tp in (5, 8, 10):
        for m in 'AB': rules[f'h{h}_tp{tp}_sl6_{m}'] = dict(hold=h, tp=tp / 100, stop=-0.06, mode=m)
    rules[f'h{h}_ma20'] = dict(hold=h, ma=True)
    rules[f'h{h}_t3'] = dict(hold=h, tstop=3)
for h in (10, 20):
    rules[f'h{h}_ma20_sl8_A'] = dict(hold=h, ma=True, stop=-0.08, mode='A')
    rules[f'h{h}_t5'] = dict(hold=h, tstop=5)
    rules[f'h{h}_tp10_sl8_B'] = dict(hold=h, tp=.10, stop=-.08, mode='B')

for h in (10, 20):
    rules[f'h{h}_ma20up'] = dict(hold=h, ma_up=True)
    rules[f'h{h}_ma20up_sl8_B'] = dict(hold=h, ma_up=True, stop=-.08, mode='B')
    rules[f'h{h}_ma20up_sl8_A'] = dict(hold=h, ma_up=True, stop=-.08, mode='A')
print('收盘在MA20下方的信号占比', np.nanmean(PC[:,0] < PM[:,0]))
yr = s.year.values; tr = yr <= 2019; te = yr >= 2020
subsets = {'all': np.ones(n, bool), 'SSS': s.sss.values, 'SSS|ge5': (s.sss | s.ge5).values}
R = {}; rows = []
for name, kw in rules.items():
    r, off = sim(**kw); R[name] = (r, off)
    for sn, msk in subsets.items():
        row = dict(rule=name, subset=sn, avg_hold=off[msk].mean())
        for pn, pm in (('全', msk), ('训00-19', msk & tr), ('测20-26', msk & te)):
            x = r[pm]; row[f'{pn}_n'] = len(x); row[f'{pn}_均%'] = x.mean(); row[f'{pn}_胜%'] = (x > 0).mean() * 100
            row[f'{pn}_最大亏%'] = x.min(); row[f'{pn}_每日%'] = x.mean() / off[pm].mean()
        rows.append(row)
G = pd.DataFrame(rows).round(3)
G.to_csv(OUT + 'exit_grid.csv', index=False, encoding='utf-8-sig')
for sn in subsets:
    g = G[G.subset == sn].sort_values('训00-19_均%', ascending=False)
    print('\n== exit grid', sn, '(按训练期均值排序 top12 + 基准)')
    show = pd.concat([g.head(12), g[g.rule.isin(['hold5', 'hold10'])]])
    print(show[['rule', 'avg_hold', '训00-19_均%', '训00-19_胜%', '测20-26_均%', '测20-26_胜%', '测20-26_最大亏%', '全_最大亏%', '训00-19_每日%', '测20-26_每日%']].to_string(index=False))
    # 训练期选出的最优规则(按均值 / 按每日收益)
    b1 = g.iloc[0].rule; b2 = g.sort_values('训00-19_每日%', ascending=False).iloc[0].rule
    print('train-best by avg:', b1, ' by per-day:', b2)

# ---------- 过滤 ----------
base5 = R['hold5'][0]; base10 = R['hold10'][0]
filt = {
    'ATR%>p80剔除': s.atr_pct.values <= np.nanpercentile(s.atr_pct[tr], 80),
    'ATR%>p90剔除': s.atr_pct.values <= np.nanpercentile(s.atr_pct[tr], 90),
    'dist_52w<-40剔除(离高点>40%)': ~(s.dist_high252.values < -40),
    'dist_high<-25剔除': ~(s.dist_high252.values < -25),
    '大盘<MA200剔除': ~(s.spx_dist_ma200.values < 0),
    '大盘>MA200剔除(反向)': ~(s.spx_dist_ma200.values >= 0),
    '同板块当天>=3剔除': s.sec_n.values < 3,
    '同板块当天>=3保留(反向)': s.sec_n.values >= 3,
    'n_base_day_idx>=5': s.ge5.values,
    'SSS': s.sss.values,
}
frows = []
for fn, fm in filt.items():
    for rn, rr in (('hold5', base5), ('hold10', base10)):
        row = dict(filter=fn, rule=rn, 保留比例=fm.mean())
        for pn, pm in (('训', tr), ('测', te)):
            a = rr[pm & fm]; b = rr[pm & ~fm]
            row[f'{pn}_n'] = len(a); row[f'{pn}_保留均%'] = a.mean(); row[f'{pn}_剔除均%'] = b.mean() if len(b) else np.nan
            row[f'{pn}_保留胜%'] = (a > 0).mean() * 100; row[f'{pn}_全体均%'] = rr[pm].mean()
        frows.append(row)
# ATR 减半: 每笔收益 * 权重, 报风险调整 (均/标准差)
F = pd.DataFrame(frows).round(3)
F.to_csv(OUT + 'filters.csv', index=False, encoding='utf-8-sig')
print('\n== filters'); print(F.to_string(index=False))

# ---------- 组合 ----------
spx = pd.read_parquet(D + 'index_gspc.parquet').iloc[:, 0].reindex(dates).ffill()
def portfolio(mask, rule, w=0.10, maxpos=10, cost=0.0005, atr_half=None, prio=None):
    r, off = R[rule]
    idx = np.where(mask)[0]
    # 当天按优先级(SSS优先, 再按 n_base_day_idx), 满仓跳过
    order = sorted(idx, key=lambda k: (ei[k], 0 if prio is None else -prio[k]))
    by_day = {}
    for k in order: by_day.setdefault(ei[k], []).append(k)
    cash = 1.0; pos = []  # (k, alloc, exit_i)
    expo = []; eq = np.full(len(dates), np.nan); start = min(by_day); ntr = 0
    for i in range(start, len(dates)):
        # 出场
        val = 0.0; still = []
        for (k, a, xi) in pos:
            if i >= xi:
                cash += a * (1 + r[k] / 100) * (1 - cost); ntr += 1
            else:
                still.append((k, a, xi))
        pos = still
        for (k, a, xi) in pos: val += a * (1 + rc[k, i - ei[k]])
        E_ = cash + val
        for k in by_day.get(i, []):
            if len(pos) >= maxpos: break
            ww = w * (0.5 if (atr_half is not None and atr_half[k]) else 1)
            a = E_ * ww
            if a > cash: break
            cash -= a; a_net = a * (1 - cost); pos.append((k, a_net, ei[k] + off[k]))
            val += a_net
        eq[i] = cash + val; expo.append(val / eq[i])
    e = pd.Series(eq, dates).dropna()
    portfolio.expo = float(np.mean(expo)); return e, ntr
def stats(e, lo, hi):
    e = e[(e.index.year >= lo) & (e.index.year <= hi)]
    dr = e.pct_change().dropna(); yrs = len(dr) / 252
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    mdd = (e / e.cummax() - 1).min()
    sh = dr.mean() / dr.std() * np.sqrt(252)
    return cagr * 100, mdd * 100, sh
prow = []
prio = (s.sss.values * 100 + s.n_base_day_idx.values).astype(float)
atrh = s.atr_pct.values > np.nanpercentile(s.atr_pct[tr], 80)
import sys
best_rules = sys.argv[1:] if len(sys.argv) > 1 else []
configs = []
for sn in ('all', 'SSS', 'SSS|ge5'):
    for rule in ['hold5', 'hold10'] + best_rules:
        for cost in (0.0005, 0.001):
            configs.append((sn, rule, cost, 0.10, 10, None))
for rule in ['hold5', 'hold10'] + best_rules:
    configs.append(('SSS|ge5', rule, 0.0005, 0.05, 20, None))
    configs.append(('SSS', rule, 0.0005, 0.20, 5, None))
    configs.append(('SSS|ge5', rule, 0.0005, 0.10, 20, None))
    configs.append(('SSS|ge5', rule, 0.0005, 0.10, 10, 'atr_half'))
    configs.append(('all', rule, 0.0005, 0.05, 20, None))
eqs = {}
for sn, rule, cost, w, mp, ah in configs:
    e, ntr = portfolio(subsets[sn], rule, w=w, maxpos=mp, cost=cost, atr_half=atrh if ah else None, prio=prio)
    key = f'{sn}|{rule}|c{cost*100:.2f}%|w{w}|N{mp}|{ah or ""}'; eqs[key] = e
    row = dict(config=key, trades=ntr, 平均仓位=round(portfolio.expo, 3))
    for lo, hi, tag in ((2000, 2026, '全'), (2000, 2019, '训'), (2020, 2026, '测')):
        c_, m_, sh_ = stats(e, lo, hi); row[f'{tag}_年化%'] = c_; row[f'{tag}_MDD%'] = m_; row[f'{tag}_Sharpe'] = sh_
    # 平均仓位占用
    prow.append(row)
b = spx[spx.index >= eqs[list(eqs)[0]].index[0]]
row = dict(config='SPX持有(价格指数,不含股息)', trades=0)
for lo, hi, tag in ((2000, 2026, '全'), (2000, 2019, '训'), (2020, 2026, '测')):
    c_, m_, sh_ = stats(b, lo, hi); row[f'{tag}_年化%'] = c_; row[f'{tag}_MDD%'] = m_; row[f'{tag}_Sharpe'] = sh_
prow.append(row)
P = pd.DataFrame(prow).round(2)
P.to_csv(OUT + 'portfolio.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(eqs).to_parquet(OUT + 'equity_curves.parquet')
print('\n== portfolio'); print(P.to_string(index=False))
# 年度表: 主配置
k0 = [k for k in eqs if k.startswith('SSS|ge5|hold5|c0.05')][0]
ann = eqs[k0].resample('YE').last().pct_change() * 100
spxa = spx.resample('YE').last().pct_change() * 100
pd.DataFrame({'strategy': ann, 'spx': spxa}).round(1).to_csv(OUT + 'annual_SSSge5_hold5.csv', encoding='utf-8-sig')
