"""买入时机对比：信号出现后，什么时候买最好？结果只留本机（analysis/private/）。
三组信号：(a) 全部基础信号，(b) + 开盘<昨日最低（O_T0 < L_{T-1}），(c) + 跳空收正。
卖出 = 买入日之后第 5 / 10 个交易日收盘。超额 = 股票收益 − 0.10% − 同一买入/卖出时点的 SPY 总回报。
SPY 开盘价：yfinance 下载 SPY OHLC（auto_adjust=True），用当日 O/C、H/C、L/C 比例乘到含分红总回报收盘价上
（同一天 OHLC 复权因子相同，比例不受复权影响），得到与 index_spy_tr 对齐的总回报 O/H/L。
训练 = 信号日 < 2025-12-01（时间加权，半衰期 10 年），测试 = 2026；t 按信号日聚类。
"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
D = os.path.join(ROOT, 'analysis', 'rebuild', 'data2000'); P = os.path.join(D, 'gh_johnbrick', 'panel')
END = '2026-11-17'; HL = 10.0; COST = 0.10; NS = (5, 10)
O, H, L, C = (pd.read_parquet(os.path.join(P, f'{k}.parquet')).loc[:END].astype('float64') for k in 'OHLC')
idx = C.index; NT = len(idx)

# ---------- SPY 总回报 OHLC ----------
spy_c = pd.read_parquet(os.path.join(D, 'index_spy_tr.parquet')).iloc[:, 0]
cache = os.path.join(HERE, 'spy_ohlc_yf.parquet')
if os.path.exists(cache):
    yf_ = pd.read_parquet(cache)
else:
    import yfinance as yf
    yf_ = yf.download('SPY', start='1999-01-01', end='2026-11-19', auto_adjust=True, progress=False)
    if isinstance(yf_.columns, pd.MultiIndex): yf_.columns = yf_.columns.get_level_values(0)
    yf_ = yf_[['Open', 'High', 'Low', 'Close']]; yf_.index = pd.to_datetime(yf_.index).tz_localize(None)
    yf_.to_parquet(cache)
yf_ = yf_.reindex(idx)
SC = spy_c.reindex(idx).values
SO = SC * (yf_.Open / yf_.Close).values; SH = SC * (yf_.High / yf_.Close).values; SL = SC * (yf_.Low / yf_.Close).values
SA = (SO + SH + SL + SC) / 4
print('SPY 缺开盘价的交易日:', int(np.isnan(SO).sum()), '| 对齐检查(收盘总回报/yf收盘 比值 2025-2026 标准差):',
      round(float(np.nanstd((SC / yf_.Close.values)[idx >= '2025-01-01'])), 4), flush=True)

# ---------- 信号 ----------
s = pd.read_parquet(os.path.join(HERE, 'jb_signals_grid.parquet'))
R = idx.get_indexer(s.signal_day); K = C.columns.get_indexer(s.ticker)
assert (R >= 0).all() and (K >= 0).all()
Ov, Hv, Lv, Cv = O.values, H.values, L.values, C.values
def px(M, k):                       # 信号日 + k 个交易日的价格
    r = R + k; ok = r < NT; out = np.full(len(s), np.nan); out[ok] = M[r[ok], K[ok]]; return out
def spx(v, k):
    r = R + k; ok = r < NT; out = np.full(len(s), np.nan); out[ok] = v[r[ok]]; return out
o = {k: px(Ov, k) for k in range(0, 4)}; h1 = px(Hv, 1); l1 = px(Lv, 1)
c = {k: px(Cv, k) for k in range(0, 16)}
lprev = px(Lv, -1)
open_below_low = o[0] < lprev                       # O_T0 < L_{T-1}，T0 信息
chk = (c[1 + 5] / c[1] - 1) * 100 - s.r5.values
print('r5 复核最大误差:', round(float(np.nanmax(np.abs(chk))), 4), flush=True)

GROUPS = {'(a) 全部基础信号': np.ones(len(s), bool), '(b) +开盘<昨日最低': open_below_low,
          '(c) +跳空收正': s.gap_down_close.values.astype(bool)}

# 买入时机：(名称, 买入日偏移 k, 股票买入价, SPY 买入价, 成交掩码)
avg1 = (o[1] + h1 + l1 + c[1]) / 4
def limit(f):
    lim = c[0] * f; fill = l1 <= lim
    price = np.where(fill, np.fmin(lim, o[1]), np.nan)
    at_open = o[1] <= lim
    sp = np.where(at_open, spx(SO, 1), spx(SA, 1))   # 开盘即成交→SPY 开盘；盘中成交→SPY 当日 OHLC 均价
    return price, sp, fill
lim99, sp99, f99 = limit(0.99); lim98, sp98, f98 = limit(0.98)
ALL = np.ones(len(s), bool)
TIMING = [
    ('T0 收盘', 0, c[0], spx(SC, 0), ALL),
    ('T1 开盘', 1, o[1], spx(SO, 1), ALL),
    ('T1 收盘(现行)', 1, c[1], spx(SC, 1), ALL),
    ('T2 开盘', 2, o[2], spx(SO, 2), ALL),
    ('T2 收盘', 2, c[2], spx(SC, 2), ALL),
    ('T3 收盘', 3, c[3], spx(SC, 3), ALL),
    ('T5 收盘', 5, c[5], spx(SC, 5), ALL),
    ('T1 开盘<T0收盘才买', 1, o[1], spx(SO, 1), o[1] < c[0]),
    ('T1 限价 T0收×0.99', 1, lim99, sp99, f99),
    ('T1 限价 T0收×0.98', 1, lim98, sp98, f98),
    ('T1 收涨才T1收盘买', 1, c[1], spx(SC, 1), c[1] > c[0]),
    # 额外 1：反向对照——T1 继续收跌才买（检验"确认反弹"和"买更便宜"哪个对）
    ('T1 收跌才T1收盘买', 1, c[1], spx(SC, 1), c[1] < c[0]),
    # 额外 2：T1 全天分批均价（O+H+L+C)/4，近似 TWAP/VWAP 拆单执行
    ('T1 全天均价(OHLC/4)', 1, avg1, spx(SA, 1), ALL),
]

tr_all = (s.signal_day < pd.Timestamp('2025-12-01')).values
te_all = (s.year == 2026).values
ty = (s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366).values; W = 0.5 ** ((2026 - ty) / HL)
G = s.signal_day.values

def ctw(x, w, g):
    Wt = w.sum(); mu = (w * x).sum() / Wt
    e = pd.Series(w * (x - mu)).groupby(g).sum().values
    return mu, mu / (np.sqrt((e ** 2).sum()) / Wt)

rows = []
for gname, gm in GROUPS.items():
    base_n = {'tr': int((gm & tr_all).sum()), 'te': int((gm & te_all).sum())}
    for tname, k, ep, sep, fm in TIMING:
        for n in NS:
            xp = c[k + n] if k + n < 16 else px(Cv, k + n)
            ret = (xp / ep - 1) * 100
            spr = (spx(SC, k + n) / sep - 1) * 100
            ex = ret - COST - spr
            ok = gm & fm & ~np.isnan(ex)
            mtr, mte = ok & tr_all, ok & te_all
            mu, t = ctw(ex[mtr], W[mtr], G[mtr])
            wr = (W[mtr] * (ret[mtr] > 1)).sum() / W[mtr].sum()
            wret = (W[mtr] * ret[mtr]).sum() / W[mtr].sum()
            # 每个信号平均超额（没成交记 0）：比较有条件买入时用
            per_sig_tr = (W[mtr] * ex[mtr]).sum() / W[gm & tr_all].sum()
            e26 = ex[mte]
            t26 = e26.mean() / (pd.Series(e26 - e26.mean()).groupby(G[mte]).sum().pow(2).sum() ** 0.5 / len(e26)) if len(e26) > 2 else np.nan
            rows.append({'信号组': gname, '买入': tname, '持有': n,
                         '训练笔数': int(mtr.sum()), '训练成交率%': round(mtr.sum() / base_n['tr'] * 100),
                         '训练加权超额%': round(mu, 2), '训练t': round(t, 1),
                         '训练加权均收益%': round(wret, 2), '训练加权涨>1%%': round(wr * 100),
                         '训练每信号超额%': round(per_sig_tr, 2),
                         '2026笔数': int(mte.sum()), '2026超额%': round(e26.mean(), 2) if len(e26) else np.nan,
                         '2026t': round(t26, 1) if len(e26) > 2 else np.nan,
                         '2026均收益%': round(ret[mte].mean(), 2) if len(e26) else np.nan,
                         '2026涨>1%%': round((ret[mte] > 1).mean() * 100) if len(e26) else np.nan,
                         '2026每信号超额%': round(e26.sum() / max(base_n['te'], 1), 2)})
Rt = pd.DataFrame(rows)
Rt.to_csv(os.path.join(HERE, 'entry_timing_result.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)
for gname in GROUPS:
    print('\n====', gname)
    for n in NS:
        print(Rt[(Rt.信号组 == gname) & (Rt.持有 == n)].drop(columns=['信号组']).to_string(index=False))

# ---------- 配对差：各全量买入时机 − T1 收盘（同一批信号，聚类 t） ----------
ref = {n: None for n in NS}
EX = {}
for tname, k, ep, sep, fm in TIMING:
    for n in NS:
        EX[(tname, n)] = ((c[k + n] / ep - 1) * 100) - COST - ((spx(SC, k + n) / sep - 1) * 100)
prow = []
for gname, gm in GROUPS.items():
    for tname, k, ep, sep, fm in TIMING:
        if not fm.all() or tname == 'T1 收盘(现行)': continue
        for n in NS:
            d = EX[(tname, n)] - EX[('T1 收盘(现行)', n)]
            ok = gm & ~np.isnan(d); mtr = ok & tr_all; mte = ok & te_all
            mu, t = ctw(d[mtr], W[mtr], G[mtr])
            prow.append({'信号组': gname, '买入': tname, '持有': n, '训练差%': round(mu, 2), '训练差t': round(t, 1),
                         '2026差%': round(d[mte].mean(), 2)})
Pd = pd.DataFrame(prow); Pd.to_csv(os.path.join(HERE, 'entry_timing_vs_T1C.csv'), index=False, encoding='utf-8-sig')
print(Pd.to_string(index=False))
