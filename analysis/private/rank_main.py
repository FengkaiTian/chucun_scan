"""主推规则内部排序：哪些特征让 10 天净收益更高（涨超 0.1% 算赢）。训练 <2025-12-01，测试 2026。"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'rebuild', 'data2000', 'gh_johnbrick', 'panel')
s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[s.has_open].sort_values(['ticker', 'signal_day']).reset_index(drop=True)
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    k = t not in last or (d - last[t]).days > 5; keep.append(k)
    if k: last[t] = d
s = s[np.array(keep)].reset_index(drop=True)
O = pd.read_parquet(os.path.join(R, 'O.parquet')); C = pd.read_parquet(os.path.join(R, 'C.parquet')); L = pd.read_parquet(os.path.join(R, 'L.parquet'))
ri = O.index.get_indexer(s.signal_day); ci = O.columns.get_indexer(s.ticker)
o = O.values[ri, ci]; pc = C.shift(1).values[ri, ci]; pl = L.shift(1).values[ri, ci]
s['gap'] = (o / pc - 1) * 100
s = s[s.gap < 0].copy()
s['open_lt_pl'] = o[s.index] < pl[s.index]
s['net'] = s.r10 - 0.10; s['win'] = s.r10 > 0.10
nday = s.groupby('signal_day').size(); s['nday'] = s.signal_day.map(nday)
tr = s.signal_day < '2025-12-01'; te = s.year == 2026
w = 0.5 ** ((2026 - (s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366)) / 10)
def row(name, m):
    out = [name]
    for lab, per in (('训练', tr), ('2026', te)):
        q = m & per & s.net.notna(); ww = w[q]
        out += [int(q.sum()), (ww * s.net[q]).sum() / ww.sum() if lab == '训练' else s.net[q].mean(),
                (ww * s.win[q]).sum() / ww.sum() * 100 if lab == '训练' else s.win[q].mean() * 100]
    return out
rows = [row('全部主推', s.net.notna())]
feats = {
 '开盘<昨日最低': s.open_lt_pl, '开盘≥昨日最低': ~s.open_lt_pl,
 '跳空收正': s.gap_down_close.astype(bool), '低开>2%': s.gap < -2, '低开1-2%': (s.gap >= -2) & (s.gap < -1), '低开<1%': s.gap >= -1,
 '收>开(阳线)': s.close > o[s.index] if False else (C.values[ri, ci][s.index] > o[s.index]),
 'RSI<25': s.rsi_lt25.astype(bool), '近52周低': s.near_52w.astype(bool),
 'ATR%≥4': s.atr_pct >= 4, 'ATR%<2.5': s.atr_pct < 2.5,
 '同日≥5只': s.nday >= 5, '同日1只': s.nday == 1,
 'VIX≥25': s.vix >= 25, 'VIX<18': s.vix < 18,
 '距MA200<-15%': s.dist_ma200 < -15, '在MA200上方': s.dist_ma200 > 0,
 '10日跌>5%': s.ret10d < -5, '5日跌>3%': s.ret5d < -3,
}
for k, m in feats.items(): rows.append(row(k, m.values if hasattr(m, 'values') else m))
T = pd.DataFrame(rows, columns=['分组', '训练笔数', '训练净%', '训练胜率%', '2026笔数', '2026净%', '2026胜率%']).round(2)
pd.set_option('display.width', 200); print(T.to_string(index=False))
s.to_parquet(os.path.join(HERE, 'main_rule_signals.parquet'), index=False)
