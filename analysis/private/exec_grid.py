"""买入价 × 持有期 全部口径统一重跑。
买入：A=T0 收盘（信号当天收盘）、B=T1 开盘、C=T1 收盘。
卖出：买入日之后第 N 个交易日收盘（A：T0+N；B、C：T1+N），N = 1, 3, 5, 10。
赢 = 涨幅 > 1%（不扣成本、不比 SPY）。
每种口径：在 2000–2025 年用全部 2/3/4 条件组合挑 top100（按时间加权胜率相对基础信号的 z 值，≥30 笔，去掉信号集合相同的），
2026 年测试 top100 任一命中的胜率和平均涨幅。
"""
import os, sys, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
COND = list(ds.COND_LABEL); NS = [1, 3, 5, 10]; HL = 10.0
P = os.path.join(ROOT, 'analysis', 'rebuild', 'data2000', 'gh_johnbrick', 'panel')
O = pd.read_parquet(os.path.join(P, 'O.parquet')).loc[:'2026-11-17']
Cp = pd.read_parquet(os.path.join(P, 'C.parquet')).loc[:'2026-11-17']

s = pd.read_parquet(os.path.join(HERE, 'jb_signals.parquet'))
s = s[(s.vix < 30) & s.has_open].sort_values(['ticker', 'signal_day'])
keep, last = [], {}
for t, d in zip(s.ticker, s.signal_day):
    ok = t not in last or (d - last[t]).days > 14; keep.append(ok)
    if ok: last[t] = d
s = s[np.array(keep)].reset_index(drop=True)

ENTRY = {'A_T0收盘': (0, 'C'), 'B_T1开盘': (1, 'O'), 'C_T1收盘': (1, 'C')}
for t, g in s.groupby('ticker'):
    c = Cp[t].dropna(); o = O[t].reindex(c.index).values; cv = c.values
    pos = c.index.get_indexer(g.signal_day); L = len(c)
    for name, (k, f) in ENTRY.items():
        ei = pos + k; ok_e = ei < L
        e = np.full(len(g), np.nan); e[ok_e] = (o if f == 'O' else cv)[ei[ok_e]]
        for n in NS:
            xi = ei + n; ok = xi < L
            x = np.full(len(g), np.nan); x[ok] = cv[xi[ok]]
            s.loc[g.index, f'{name}_{n}'] = (x / e - 1) * 100
s.to_parquet(os.path.join(HERE, 'jb_signals_grid.parquet'), index=False)

tr = (s.signal_day < pd.Timestamp('2025-12-01')).values      # 留出 10 个交易日以上，持有期不跨进 2026
te = (s.year == 2026).values
ty = (s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366).values
w_all = 0.5 ** ((2026 - ty) / HL)
keys = [c for k in (2, 3, 4) for c in itertools.combinations(range(37), k)]
idx = np.array([list(c) + [-1] * (4 - len(c)) for c in keys])
XX = np.hstack([s[COND].values.astype(float), np.ones((len(s), 1))])
St = XX[tr]; St26 = XX[te]

rows, tops = [], {}
for name in ENTRY:
    for n in NS:
        col = f'{name}_{n}'; v = s[col].values
        ok = ~np.isnan(v)
        mtr = tr & ok
        win = (v > 1).astype(float)
        w = w_all * mtr
        p0 = (w * win).sum() / w.sum()                                   # 基础信号加权胜率
        A = np.vstack([w, w * win, w * w])[:, tr]
        st = np.hstack([A @ (St[:, b[:, 0]] * St[:, b[:, 1]] * St[:, b[:, 2]] * St[:, b[:, 3]]) for b in np.array_split(idx, 15)])
        cnt = np.hstack([(mtr[tr].astype(float)) @ (St[:, b[:, 0]] * St[:, b[:, 1]] * St[:, b[:, 2]] * St[:, b[:, 3]]) for b in np.array_split(idx, 15)])
        W, Ww, W2 = st
        wr = np.where(W > 0, Ww / np.maximum(W, 1e-12), np.nan)
        neff = np.where(W2 > 0, W ** 2 / np.maximum(W2, 1e-12), 0)
        z = (wr - p0) / np.sqrt(p0 * (1 - p0) / np.maximum(neff, 1))
        z[cnt < 30] = -np.inf
        order = np.argsort(-z)
        seen, top = set(), []
        for i in order:
            if not np.isfinite(z[i]): break
            m = np.all(St[:, [j for j in keys[i]]] > 0, axis=1)
            h = hash(m.tobytes())
            if h in seen: continue
            seen.add(h); top.append(i)
            if len(top) == 100: break
        tops[col] = top
        hit = np.zeros(te.sum(), bool)
        for i in top: hit |= np.all(St26[:, list(keys[i])] > 0, axis=1)
        v26 = v[te]; ok26 = ~np.isnan(v26)
        b = v26[ok26]; q = v26[hit & ok26]
        vtr = v[mtr]
        rows.append({'买入': name[2:], '持有': n,
                     '训练 全部信号胜率%': round((vtr > 1).mean() * 100),
                     '训练 top100 加权胜率%': round(float(np.mean(wr[top])) * 100),
                     '2026 全部信号': f'{len(b)}笔 胜{(b>1).mean()*100:.0f}% 均{b.mean():+.2f}%',
                     '2026 top100任一': f'{len(q)}笔 胜{(q>1).mean()*100:.0f}% 均{q.mean():+.2f}%' if len(q) else '0笔'})
        pd.DataFrame([{'排名': j + 1, '组合': '＋'.join(ds.COND_LABEL[COND[x]] for x in keys[i]), '训练笔数': int(cnt[i]),
                       '训练加权胜率%': round(wr[i] * 100), 'z': round(z[i], 1)} for j, i in enumerate(top)]
                     ).to_csv(os.path.join(HERE, f'top100_{col}.csv'), index=False, encoding='utf-8-sig')
        print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(HERE, 'exec_grid_result.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(R.to_string(index=False))
