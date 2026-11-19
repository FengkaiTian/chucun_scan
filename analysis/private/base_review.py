"""基础信号体检：在整个面板（所有时点成分股日）上逐项拆、调参、与"开盘<昨低"族及简单基础信号比较。结果只留本机。
口径：T1 收盘买，持有 5/10 个交易日，超额 = 收益 − 0.10% − 同期 SPY（同进同出，含分红）。
训练 = 信号日 2000-01-01..2025-11-30（时间加权 w=0.5^((2026−年份)/10)），测试 = 2026。t 按信号日聚类。
统一样本：信号日是成分股、该股已有 ≥260 根收盘、T0 与 T−1 的 O/H/L 都有数据（与现有信号表 has_open 一致）。
"""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
D = os.path.join(ROOT, 'analysis', 'rebuild', 'data2000'); P = os.path.join(D, 'gh_johnbrick', 'panel')
END = '2026-11-17'; HL = 10.0; COST = 0.10
t0 = time.time()
O, H, L, C = (pd.read_parquet(os.path.join(P, f'{k}.parquet')).loc[:END].astype('float64') for k in 'OHLC')
spy = pd.read_parquet(os.path.join(D, 'index_spy_tr.parquet')).iloc[:, 0].loc[:END]
vix = pd.read_parquet(os.path.join(D, 'index_vix.parquet')).iloc[:, 0].loc[:END]
idx = C.index; NT = len(idx)
mem = pd.read_parquet(os.path.join(P, 'member_days.parquet')); mem = mem[mem.date <= END]
Mb = np.zeros(C.shape, bool)
mi, mk = idx.get_indexer(mem.date), C.columns.get_indexer(mem.ticker); ok = (mi >= 0) & (mk >= 0)
Mb[mi[ok], mk[ok]] = True

# ---------- 统一样本 ----------
age = C.notna().cumsum().values >= 260
ohl = (O.notna() & H.notna() & L.notna()).values
U0 = Mb & age & C.notna().values & (idx >= '2000-01-01')[:, None]
U = U0 & ohl & np.vstack([np.zeros((1, C.shape[1]), bool), ohl[:-1]])

# ---------- 前向收益 ----------
sp = spy.reindex(idx).ffill()
fw = {}
for n in (5, 10):
    r = ((C.shift(-1 - n) / C.shift(-1) - 1) * 100).values
    b = ((sp.shift(-1 - n) / sp.shift(-1) - 1) * 100).values
    fw[n] = (r, r - COST - b[:, None])
vx = vix.reindex(idx).ffill().values
ty = (idx.year + idx.dayofyear / 366).values; wday = 0.5 ** ((2026 - ty) / HL)
train_day = (idx >= '2000-01-01') & (idx < '2025-12-01'); test_day = idx.year == 2026
NY = (pd.Timestamp('2025-12-01') - pd.Timestamp('2000-01-01')).days / 365.25
dnum = idx.values.astype('datetime64[D]').astype(np.int64)

# ---------- 指标 ----------
rsi14 = ds.calc_rsi(C); r1 = rsi14.shift(1)
_, _, hist = ds.calc_macd(C)
def rsi_w(x, n):
    d = x.diff(); au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)
rsi2 = rsi_w(C, 2)
pc, pl = C.shift(1), L.shift(1)
ma5, ma20, ma200 = (C.rolling(n, min_periods=n).mean() for n in (5, 20, 200))
ibs = ((C - L) / (H - L).where(H > L))
print('指标', f'{time.time()-t0:.0f}s', flush=True)

def macd_up(k):           # 柱<0 且最近 k 根柱逐根抬高（k=3 即现行：h>h1>h2）
    m = hist < 0
    for i in range(k - 1): m &= hist.shift(i) > hist.shift(i + 1)
    return m.values
def rsi_cond(th=30, prev=True):
    return (((r1 < th) if prev else (rsi14 < th)) & (rsi14 > r1)).values
MACD3 = macd_up(3); RSIB = rsi_cond()
G1 = (O < pl).values                          # 开盘<昨日最低
G2 = G1 & (C > O).values                      # 开盘<昨低 且收>开
G3 = ((O < pc * 0.99) & (C > O)).values       # 跳空收正（现有 gap_down_close）
R14 = (rsi14 < 30).values; R2 = (rsi2 < 10).values; BIAS = (C / ma20 - 1 < -0.08).values
IBSM5 = ((ibs < 0.2) & (C < ma5)).values; ABOVE200 = (C > ma200).values

def dedup_keep(rr, cc):
    dd = dnum[rr]; keep = np.zeros(len(rr), bool); lc, ld = -1, 0
    for i in range(len(rr)):
        if cc[i] != lc or dd[i] - ld > 14: keep[i] = True; lc, ld = cc[i], dd[i]
    return keep

def events(cond, vixth=30, dedup=True, base=U, flags=False):
    m = cond & base
    if vixth is not None: m &= (vx < vixth)[:, None]
    rr, cc = np.nonzero(m)
    o = np.lexsort((rr, cc)); rr, cc = rr[o], cc[o]
    if flags: return rr, cc, dedup_keep(rr, cc)
    if dedup and len(rr):
        keep = dedup_keep(rr, cc); rr, cc = rr[keep], cc[keep]
    return rr, cc

def ctw(x, w, g):
    W = w.sum(); mu = (w * x).sum() / W
    e = np.bincount(g, weights=w * (x - mu))
    return mu, mu / (np.sqrt((e ** 2).sum()) / W)

def stats(rr, cc):
    out = {}
    for n in (5, 10):
        r, x = fw[n][0][rr, cc], fw[n][1][rr, cc]; v = ~np.isnan(x)
        tr = v & train_day[rr]; te = v & test_day[rr]
        mu, t = ctw(x[tr], wday[rr[tr]], rr[tr]) if tr.sum() > 2 else (np.nan, np.nan)
        out[n] = dict(ntr=int(tr.sum()), mu=mu, t=t, win_tr=(r[tr] > 1).mean() * 100,
                      nte=int(te.sum()), xte=x[te].mean() if te.any() else np.nan,
                      win_te=(r[te] > 1).mean() * 100 if te.any() else np.nan)
    return out

ROWS = []
def run(sec, name, cond, **kw):
    run_ev(sec, name, *events(cond, **kw))
def run_ev(sec, name, rr, cc):
    s = stats(rr, cc)
    row = {'部分': sec, '信号': name, '训练笔数': s[10]['ntr'], '每年笔数': round(s[10]['ntr'] / NY), '2026笔数': s[10]['nte']}
    for n in (5, 10):
        q = s[n]
        row.update({f'{n}d训练超额%': round(q['mu'], 2), f'{n}d_t': round(q['t'], 1), f'{n}d训练涨>1%': round(q['win_tr']),
                    f'{n}d_2026超额%': round(q['xte'], 2), f'{n}d_2026涨>1%': round(q['win_te']) if q['win_te'] == q['win_te'] else None})
    ROWS.append(row)
    print(f"{sec:4s} {name:34s} 年{row['每年笔数']:6d} 26年{row['2026笔数']:5d} | 5d {row['5d训练超额%']:+.2f} t{row['5d_t']:5.1f} 26:{row['5d_2026超额%']:+.2f} "
          f"| 10d {row['10d训练超额%']:+.2f} t{row['10d_t']:5.1f} 26:{row['10d_2026超额%']:+.2f} 涨>1% {row['10d_2026涨>1%']}  ({time.time()-t0:.0f}s)", flush=True)

ALL = np.ones(C.shape, bool)
# 1 逐项拆
run('拆', '现行基础信号(全)', MACD3 & RSIB); print('基础信号总笔数(应≈7770):', len(events(MACD3 & RSIB)[0]))
run('拆', '现行·不要O/H/L样本限制(核对)', MACD3 & RSIB, base=U0)
run('拆', '去掉MACD', RSIB)
run('拆', '去掉RSI', MACD3)
run('拆', '去掉VIX', MACD3 & RSIB, vixth=None)
run('拆', '去掉14天去重', MACD3 & RSIB, dedup=False)
run('拆', 'MACD+RSI(无VIX无去重)', MACD3 & RSIB, vixth=None, dedup=False)
run('拆', '只留MACD', MACD3, vixth=None, dedup=False)
run('拆', '只留RSI', RSIB, vixth=None, dedup=False)
run('拆', '只留VIX<30(所有成分股日)', ALL, dedup=False)
run('拆', '所有成分股日(参照)', ALL, vixth=None, dedup=False)
# 2 参数
for th in (25, 35): run('参数', f'RSI阈值{th}', MACD3 & rsi_cond(th))
for k in (2, 4): run('参数', f'MACD回升{k}根', macd_up(k) & RSIB)
run('参数', '当日RSI<30且回升', MACD3 & rsi_cond(30, prev=False))
run('参数', '前日RSI<30(不要求回升)', MACD3 & (r1 < 30).values)
for vt in (25, 35): run('参数', f'VIX<{vt}', MACD3 & RSIB, vixth=vt)
# 3 开盘<昨低族 × 前提（统一 VIX<30 + 14 天去重，去重在合并条件之后）
PRE = {'无前提': ALL, '现行基础(MACD+RSI)': MACD3 & RSIB, 'RSI14<30': R14, 'RSI2<10': R2, '距MA20<-8%': BIAS,
       '只MACD': MACD3, '只RSI(前日<30回升)': RSIB}
GAP = {'开盘<昨低': G1, '开盘<昨低且收>开': G2, '跳空收正(<昨收×0.99且收>开)': G3}
for gn, g in GAP.items():
    for pn, p in PRE.items(): run('跳空', f'{pn}＋{gn}', p & g)
for gn, g in (('开盘<昨低', G1), ('开盘<昨低且收>开', G2)):
    run('跳空', f'无前提＋{gn}(无VIX)', g, vixth=None)
# 现有信号表口径：先对基础信号去重再筛跳空
rr, cc = events(MACD3 & RSIB)
for gn, g in GAP.items():
    k = g[rr, cc]; s_ = stats(rr[k], cc[k])
    print(f'表内口径 基础去重后筛 {gn}: 年{s_[10]["ntr"]/NY:.0f} 10d {s_[10]["mu"]:+.2f} t{s_[10]["t"]:.1f} 26:{s_[10]["xte"]:+.2f}(n={s_[10]["nte"]}) | 5d {s_[5]["mu"]:+.2f} t{s_[5]["t"]:.1f} 26:{s_[5]["xte"]:+.2f}')
# 4 简单基础信号
SIMPLE = {'RSI2<10': R2, 'RSI2<5': (rsi2 < 5).values, 'RSI2<10且>MA200(Connors)': R2 & ABOVE200,
          'IBS<0.2且收<MA5': IBSM5, 'IBS<0.2且收<MA5且>MA200': IBSM5 & ABOVE200, '距MA20<-8%': BIAS, 'RSI14<30': R14}
for sn, sc in SIMPLE.items():
    run('简单', sn, sc)
    run('简单', f'{sn}＋开盘<昨低且收>开', sc & G2)

# 5 先对前提去重、再筛跳空（现有信号表的做法）
def pre_then(sec, name, pre, gap, **kw):
    rr, cc = events(pre, **kw); k = gap[rr, cc]; run_ev(sec, name, rr[k], cc[k])
G4 = G1 & (C > pc).values                     # 开盘<昨低 且 收>昨收（基础信号里 RSI 回升 ⇒ 收>昨收，所以 G1 在基础内自动等于 G4）
PRE2 = {'现行基础(MACD+RSI)': MACD3 & RSIB, 'RSI14<30': R14, 'RSI2<10': R2, '距MA20<-8%': BIAS, '只MACD': MACD3,
        '只RSI(前日<30回升)': RSIB, 'IBS<0.2且收<MA5': IBSM5}
for pn, p in PRE2.items(): pre_then('先去重', f'{pn}→开盘<昨低', p, G1)
for pn, p in PRE2.items(): pre_then('先去重', f'{pn}→开盘<昨低且收>昨收', p, G4)
# 5b 基础参数 × 跳空（先去重后筛）
VAR = {'RSI阈值25': (MACD3 & rsi_cond(25), {}), 'RSI阈值35': (MACD3 & rsi_cond(35), {}), 'MACD回升2根': (macd_up(2) & RSIB, {}),
       'MACD回升4根': (macd_up(4) & RSIB, {}), '当日RSI<30且回升': (MACD3 & rsi_cond(30, False), {}),
       '不要VIX': (MACD3 & RSIB, {'vixth': None}), 'VIX<25': (MACD3 & RSIB, {'vixth': 25}), 'VIX<35': (MACD3 & RSIB, {'vixth': 35}),
       '去掉MACD': (RSIB, {}), '去掉RSI': (MACD3, {})}
for vn, (p, kw) in VAR.items(): pre_then('参数×跳空', f'{vn}→开盘<昨低', p, G1, **kw)
# 5c 基础∩开盘<昨低：是否 14 天内第一次基础信号
rr, cc, kp = events(MACD3 & RSIB, flags=True); g = G1[rr, cc]
run_ev('去重拆', '基础∩开盘<昨低·14天内首次基础', rr[g & kp], cc[g & kp])
run_ev('去重拆', '基础∩开盘<昨低·非首次', rr[g & ~kp], cc[g & ~kp])
# 5d 开盘<昨低且收>昨收 × 前提（合并后去重）
for pn, p in {'无前提': ALL, 'RSI14<30': R14, '前日RSI14<30': (r1 < 30).values, '前日RSI2<10': (rsi2.shift(1) < 10).values,
              '距MA20<-8%(昨日)': (pc / ma20.shift(1) - 1 < -0.08).values, '只MACD': MACD3, '现行基础': MACD3 & RSIB}.items():
    run('收>昨收', f'{pn}＋开盘<昨低且收>昨收', p & G4)

T = pd.DataFrame(ROWS)
T.to_csv(os.path.join(HERE, 'base_review.csv'), index=False, encoding='utf-8-sig')
print('done', f'{time.time()-t0:.0f}s')
