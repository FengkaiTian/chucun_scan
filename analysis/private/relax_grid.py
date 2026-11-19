"""放宽基础信号参数的大网格（不要 VIX 过滤），按绝对净收益（扣 0.10% 后 >0）评判，SPY 超额只作参考。结果只留本机。
口径：T1 收盘买，持有 3/5/10 个交易日；训练 = 信号日 2000-01-01..2025-11-30（w=0.5^((2026−年份)/10)），测试 = 2026；t 按信号日聚类。
样本：时点成分股、已有 ≥260 根收盘、T0 与 T−1 的 O/H/L 有数据（触发条件要用）。
去重：先对基础信号做 14 天/5 天"只算第一次"，再筛触发条件（现有做法）；或不去重。
"""
import os, sys, time, itertools, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
D = os.path.join(ROOT, 'analysis', 'rebuild', 'data2000'); P = os.path.join(D, 'gh_johnbrick', 'panel')
END = '2026-11-17'; HL = 10.0; COST = 0.10; HOLDS = (3, 5, 10)
t0 = time.time()
O, H, L, C = (pd.read_parquet(os.path.join(P, f'{k}.parquet')).loc[:END].astype('float64') for k in 'OHLC')
spy = pd.read_parquet(os.path.join(D, 'index_spy_tr.parquet')).iloc[:, 0].loc[:END]
idx = C.index
mem = pd.read_parquet(os.path.join(P, 'member_days.parquet')); mem = mem[mem.date <= END]
Mb = np.zeros(C.shape, bool)
mi, mk = idx.get_indexer(mem.date), C.columns.get_indexer(mem.ticker); ok = (mi >= 0) & (mk >= 0)
Mb[mi[ok], mk[ok]] = True
age = C.notna().cumsum().values >= 260
ohl = (O.notna() & H.notna() & L.notna()).values
U = Mb & age & C.notna().values & (idx >= '2000-01-01')[:, None] & ohl & np.vstack([np.zeros((1, C.shape[1]), bool), ohl[:-1]])

sp = spy.reindex(idx).ffill()
NET, EXC = {}, {}
for n in HOLDS:
    r = ((C.shift(-1 - n) / C.shift(-1) - 1) * 100).values
    b = ((sp.shift(-1 - n) / sp.shift(-1) - 1) * 100).values
    NET[n] = (r - COST).astype(np.float32); EXC[n] = (r - COST - b[:, None]).astype(np.float32)
ty = (idx.year + idx.dayofyear / 366).values; wday = 0.5 ** ((2026 - ty) / HL)
year = idx.year.values
train_day = (idx >= '2000-01-01') & (idx < '2025-12-01'); test_day = year == 2026
NY = (pd.Timestamp('2025-12-01') - pd.Timestamp('2000-01-01')).days / 365.25
dnum = idx.values.astype('datetime64[D]').astype(np.int64)

rsi14 = ds.calc_rsi(C); r1 = rsi14.shift(1)
_, _, hist = ds.calc_macd(C)
RS = {'前日': r1.values, '当日': rsi14.values}; RUP = (rsi14 > r1).values
HNEG = (hist < 0).values
UP = [(hist.shift(i) > hist.shift(i + 1)).values for i in range(3)]
pc, pl, ph = C.shift(1), L.shift(1), H.shift(1)
TRIG = {'无': None, '开盘<昨低': (O < pl).values, '开盘<昨低且收>开': ((O < pl) & (C > O)).values,
        '跳空收正': ((O < pc * 0.99) & (C > O)).values, '外包阳线': ((H > ph) & (L < pl) & (C > O)).values,
        '低开(开<昨收)': (O < pc).values}
print('准备', f'{time.time()-t0:.0f}s', flush=True)

def dedup_keep(rr, cc, days):
    dd = dnum[rr]; keep = np.zeros(len(rr), bool); lc, ld = -1, 0
    for i in range(len(rr)):
        if cc[i] != lc or dd[i] - ld > days: keep[i] = True; lc, ld = cc[i], dd[i]
    return keep

def ctw(x, w, g):
    W = w.sum(); mu = (w * x).sum() / W
    e = np.bincount(g, weights=w * (x - mu))
    return mu, mu / (np.sqrt((e ** 2).sum()) / W)

def stats(rr, cc, n):
    x = NET[n][rr, cc].astype(np.float64); v = ~np.isnan(x)
    tr = v & train_day[rr]; te = v & test_day[rr]
    if tr.sum() < 10: return None
    rt, xt = rr[tr], x[tr]; w = wday[rt]
    mu, t = ctw(xt, w, rt)
    ex = EXC[n][rr, cc].astype(np.float64)[tr]; exm = (w * ex).sum() / w.sum()
    yi = year[rt] - 2000
    ys, yc = np.bincount(yi, xt, 26), np.bincount(yi, minlength=26)
    has = yc > 0; ypos = (ys[has] / yc[has] > 0).mean()
    bi = np.minimum(yi // 5, 4); bs, bc = np.bincount(bi, xt, 5), np.bincount(bi, minlength=5)
    worst = (bs[bc > 0] / bc[bc > 0]).min()
    xe = x[te]
    return {'每年笔数': tr.sum() / NY, '净收益%': mu, 't': t, '胜率%': (xt > 0).mean() * 100, '正年份比例': ypos,
            '最差5年块%': worst, '2026笔数': int(te.sum()), '2026净收益%': xe.mean() if te.any() else np.nan,
            '2026胜率%': (xe > 0).mean() * 100 if te.any() else np.nan, '超额SPY%(参考)': exm}

# ---------- 基准：随机成分股日 ----------
rrU, ccU = np.nonzero(U)
BASE = {}
for n in HOLDS:
    b = stats(rrU, ccU, n); BASE[n] = b
    print(f'随机成分股日 持有{n}: 训练净 {b["净收益%"]:+.3f}% t{b["t"]:.1f} 胜率 {b["胜率%"]:.0f}% 正年份 {b["正年份比例"]:.0%} 最差块 {b["最差5年块%"]:+.2f} | 2026 净 {b["2026净收益%"]:+.3f}% 胜率 {b["2026胜率%"]:.0f}% | 超额 {b["超额SPY%(参考)"]:+.3f}', flush=True)

# ---------- 网格 ----------
rows = []
for thr, rm, rup, k, hneg in itertools.product((30, 35, 40, 45), ('前日', '当日'), (True, False), (2, 3, 4), (True, False)):
    m = RS[rm] < thr
    if rup: m = m & RUP
    if hneg: m = m & HNEG
    for i in range(k - 1): m = m & UP[i]
    rr, cc = np.nonzero(m & U)
    o = np.lexsort((rr, cc)); rr, cc = rr[o], cc[o]
    for dname, days in (('14天', 14), ('5天', 5), ('不去重', None)):
        if days is None: r_, c_ = rr, cc
        else: kp = dedup_keep(rr, cc, days); r_, c_ = rr[kp], cc[kp]
        for tn, tg in TRIG.items():
            if tg is None: r2, c2 = r_, c_
            else: kk = tg[r_, c_]; r2, c2 = r_[kk], c_[kk]
            for n in HOLDS:
                s = stats(r2, c2, n)
                if s is None: continue
                rows.append({'RSI阈值': thr, 'RSI取': rm, 'RSI回升': rup, 'MACD回升根': k, '柱<0': hneg, '去重': dname,
                             '触发': tn, '持有': n, **s})
    print(thr, rm, rup, k, hneg, len(rows), f'{time.time()-t0:.0f}s', flush=True)

T = pd.DataFrame(rows)
bn = T.持有.map({n: BASE[n]['净收益%'] for n in HOLDS}); b26 = T.持有.map({n: BASE[n]['2026净收益%'] for n in HOLDS})
T['稳健'] = ((T['净收益%'] > 0) & (T['t'] >= 2) & (T['正年份比例'] >= 0.7) &
             (T['最差5年块%'] > -0.5) & (T['2026净收益%'] > 0) & (T['每年笔数'] >= 40))
T['胜过随机日'] = (T['净收益%'] > bn) & (T['2026净收益%'] > b26)
T['训练比随机日多%'] = T['净收益%'] - bn
T['每年总收益%'] = T['每年笔数'] * T['净收益%']
T = T.round(3)
T.to_csv(os.path.join(HERE, 'relax_grid_all.csv'), index=False, encoding='utf-8-sig')
R = T[T.稳健].sort_values('每年总收益%', ascending=False)
R.to_csv(os.path.join(HERE, 'relax_grid_robust.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 400); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
print('\n总格子', len(T), '稳健', int(T.稳健.sum()), '其中胜过随机日', int((T.稳健 & T.胜过随机日).sum()))
print(T.groupby('持有').agg(格子=('稳健', 'size'), 稳健=('稳健', 'sum'), 稳健且胜随机=('胜过随机日', lambda s: int((s & T.loc[s.index, '稳健']).sum()))))
print(T[T.稳健].groupby(['触发', '去重']).size().unstack(fill_value=0))
print('\n稳健 top 30：'); print(R.head(30).to_string(index=False))
print('\n稳健且胜过随机日 top 20：'); print(R[R.胜过随机日].head(20).to_string(index=False))
print('done', f'{time.time()-t0:.0f}s')
