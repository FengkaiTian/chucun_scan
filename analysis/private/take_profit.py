"""主推 + 止盈：T1 收盘买；之后每天，只要盘中涨到 +3%（相对买入价）就卖（开盘就高于 +3% 则按开盘价卖），
否则拿满 N 个交易日收盘卖。对比不止盈、拿满 10 天。扣 0.10% 来回成本；涨超 0.1% 算赢。
另试 2%、4%、5% 止盈，以及"收盘价涨超 3% 才卖"。"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, '..', 'rebuild', 'data2000', 'gh_johnbrick', 'panel')
O, H, C = (pd.read_parquet(os.path.join(P, f'{k}.parquet')) for k in 'OHC')
s = pd.read_parquet(os.path.join(HERE, 'main_rule_signals.parquet'))
deep = (s.gap <= -1) | s.open_lt_pl
fear = (s.vix >= 25) | (s.nday >= 5)
s['grade'] = np.where(deep & fear, 'A', np.where(deep | fear, 'B', 'C'))

def sim(tp, maxd, mode='intraday'):
    out = np.full(len(s), np.nan); days = np.full(len(s), np.nan)
    for t, g in s.groupby('ticker'):
        c = C[t].dropna(); o = O[t].reindex(c.index).values; h = H[t].reindex(c.index).values; cv = c.values
        pos = c.index.get_indexer(g.signal_day) + 1                      # 买入日 T1
        for k, p in zip(g.index, pos):
            if p <= 0 or p + maxd >= len(c): continue
            e = cv[p]; lim = e * (1 + (tp or 0) / 100); x = None
            if tp is not None:
                for d in range(1, maxd + 1):
                    i = p + d
                    if mode == 'intraday':
                        if np.isfinite(o[i]) and o[i] >= lim: x = o[i]; break
                        if np.isfinite(h[i]) and h[i] >= lim: x = lim; break
                    elif cv[i] >= lim: x = cv[i]; break
            if x is None: x = cv[p + maxd]; d = maxd
            out[k] = (x / e - 1) * 100 - 0.10; days[k] = d
    return out, days

tr = (s.signal_day < '2025-12-01').values; te = (s.year == 2026).values
ny = (pd.Timestamp('2025-12-01') - pd.Timestamp('2000-01-03')).days / 365.25
rows = []
for lab, tp, mx, mode in [('不止盈，拿满10天', None, 10, ''), ('盘中涨3%就卖，否则10天', 3, 10, 'intraday'),
                          ('收盘涨3%才卖，否则10天', 3, 10, 'close'), ('盘中涨2%就卖，否则10天', 2, 10, 'intraday'),
                          ('盘中涨4%就卖，否则10天', 4, 10, 'intraday'), ('盘中涨5%就卖，否则10天', 5, 10, 'intraday'),
                          ('盘中涨3%就卖，否则5天', 3, 5, 'intraday'), ('盘中涨3%就卖，否则20天', 3, 20, 'intraday')]:
    r, d = sim(tp, mx, mode)
    for grp, gm in (('全部主推', np.ones(len(s), bool)), ('A+B', (s.grade != 'C').values)):
        row = {'卖法': lab, '范围': grp}
        for per, pm in (('训练', tr), ('2026', te)):
            m = pm & gm & ~np.isnan(r)
            row[f'{per}笔数'] = int(m.sum()); row[f'{per}每笔净%'] = round(r[m].mean(), 2)
            row[f'{per}胜率%'] = round((r[m] > 0).mean() * 100); row[f'{per}平均持有天'] = round(d[m].mean(), 1)
            row[f'{per}最差5%'] = round(np.percentile(r[m], 5), 1)
            row[f'{per}每天收益%'] = round(r[m].mean() / d[m].mean(), 3)
        rows.append(row); print(row, flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, 'take_profit.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 260); print(T.to_string(index=False))
