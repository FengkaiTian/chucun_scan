"""扩充条件库（原 37 个 + 网上常见的超卖/反弹/量价/大盘/日历条件），全部 1/2/3 条件组合。
信号 = 基础信号（VIX<30，14 天去重，有开盘价），T1 收盘买，持有 10 个交易日（也给 5 天）。
排序 = 时间加权（半衰期 10 年）扣 0.10% 后相对同期 SPY 的超额 t 值，≥30 笔，去掉信号集合相同的。
训练 = 信号日 < 2025-12-01；测试 = 2026。结果只留本机。
"""
import os, sys, itertools, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds
D = os.path.join(ROOT, 'analysis', 'rebuild', 'data2000'); P = os.path.join(D, 'gh_johnbrick', 'panel')
END = '2026-11-17'; HL = 10.0; MINN = 30; COST = 0.10
t0 = time.time()
O, H, L, C, V = (pd.read_parquet(os.path.join(P, f'{k}.parquet')).loc[:END].astype('float64') for k in 'OHLCV')
spy = pd.read_parquet(os.path.join(D, 'index_spy_tr.parquet')).iloc[:, 0].loc[:END]
vix = pd.read_parquet(os.path.join(D, 'index_vix.parquet')).iloc[:, 0].loc[:END]
s = pd.read_parquet(os.path.join(HERE, 'jb_signals_grid.parquet'))
R = C.index.get_indexer(s.signal_day); K = C.columns.get_indexer(s.ticker)
assert (R >= 0).all() and (K >= 0).all()
F = {}
def add(name, fr): F[name] = np.asarray(fr.values[R, K], dtype=bool)
DAY = set()
def addm(name, ser): DAY.add(name); F[name] = ser.reindex(C.index).fillna(False).astype(bool).values[R]

# ---------- 原 37 个 ----------
for c in ds.COND_LABEL: F[ds.COND_LABEL[c]] = s[c].values.astype(bool)

# ---------- 工具 ----------
def rsi(x, n):
    d = x.diff(); au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)
pc, po = C.shift(1), O.shift(1)
ph, pl = H.shift(1), L.shift(1)
ma = {n: C.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 50, 100, 200)}
tr = pd.DataFrame(np.fmax(np.fmax((H - L).values, (H - pc).abs().values), (L - pc).abs().values), index=C.index, columns=C.columns)
atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
rng = (H - L).where(H > L)
ret1 = C.pct_change()
print('载入', f'{time.time()-t0:.0f}s', flush=True)

# ---------- 震荡指标 ----------
r2, r3, r5, r14 = rsi(C, 2), rsi(C, 3), rsi(C, 5), rsi(C, 14)
for th in (2, 5, 10, 20): add(f'RSI2<{th}', r2 < th)
for th in (10, 20): add(f'RSI3<{th}', r3 < th)
for th in (15, 25): add(f'RSI5<{th}', r5 < th)
add('RSI14<15', r14 < 15); add('RSI14当天回升', r14 > r14.shift(1)); add('RSI14回升2天', (r14 > r14.shift(1)) & (r14.shift(1) > r14.shift(2)))
add('RSI2连续2天<10', (r2 < 10) & (r2.shift(1) < 10))
# Connors RSI
up, dn = (ret1 > 0).values, (ret1 < 0).values
st = np.zeros(C.shape)
for i in range(1, len(C)):
    p = st[i - 1]; st[i] = np.where(up[i], np.where(p > 0, p + 1, 1), np.where(dn[i], np.where(p < 0, p - 1, -1), 0))
streak = pd.DataFrame(st, index=C.index, columns=C.columns)
pr100 = ret1.rolling(100, min_periods=100).rank(pct=True) * 100
crsi = (r3 + rsi(streak, 2) + pr100) / 3
for th in (5, 10, 15, 20): add(f'ConnorsRSI<{th}', crsi < th)
add('当天跌幅百分位<5%', pr100 < 5); add('当天涨幅百分位>80%', pr100 > 80)
for k in (4, 5, 6): add(f'连跌≥{k}天', streak <= -k)
add('今天收涨', ret1 > 0); add('今天涨>1%', ret1 > 0.01); add('今天涨>2%', ret1 > 0.02); add('今天涨>3%', ret1 > 0.03)
add('今天跌', ret1 < 0); add('今天跌>2%', ret1 < -0.02); add('昨天跌>3%', ret1.shift(1) < -0.03)
print('RSI 类', f'{time.time()-t0:.0f}s', flush=True)
# 随机指标 / KDJ
ll14, hh14 = L.rolling(14).min(), H.rolling(14).max()
k14 = 100 * (C - ll14) / (hh14 - ll14); ks = k14.rolling(3).mean(); dsl = ks.rolling(3).mean()
for th in (5, 10, 20): add(f'Stoch%K<{th}', k14 < th)
add('慢速K上穿D(<20)', (ks > dsl) & (ks.shift(1) <= dsl.shift(1)) & (ks < 20))
ll9, hh9 = L.rolling(9).min(), H.rolling(9).max()
rsv = 100 * (C - ll9) / (hh9 - ll9)
Kk = rsv.ewm(alpha=1 / 3, adjust=False).mean(); Dk = Kk.ewm(alpha=1 / 3, adjust=False).mean(); J = 3 * Kk - 2 * Dk
for th in (0, -5, -10): add(f'KDJ_J<{th}', J < th)
add('KDJ金叉', (Kk > Dk) & (Kk.shift(1) <= Dk.shift(1))); add('KDJ_K<20', Kk < 20); add('J值回升', J > J.shift(1))
wr = -100 * (hh14 - C) / (hh14 - ll14)
add('WR<-90', wr < -90); add('WR<-95', wr < -95)
tp = (H + L + C) / 3; cci = (tp - tp.rolling(20).mean()) / (0.015 * 0.7979 * tp.rolling(20).std())
add('CCI<-150', cci < -150); add('CCI<-200', cci < -200)
mf = tp * V; pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum(); neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
mfi = 100 - 100 / (1 + pos / neg); add('MFI<10', mfi < 10)
bp = C - np.fmin(L, pc)
uo = 100 * (4 * bp.rolling(7).sum() / tr.rolling(7).sum() + 2 * bp.rolling(14).sum() / tr.rolling(14).sum() + bp.rolling(28).sum() / tr.rolling(28).sum()) / 7
add('UO<30', uo < 30); add('UO<25', uo < 25)
e12, e26 = C.ewm(span=12, adjust=False).mean(), C.ewm(span=26, adjust=False).mean(); macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
add('MACD线<-3%股价', macd / C < -0.03); add('MACD线<-5%股价', macd / C < -0.05); add('MACD金叉', (macd > sig) & (macd.shift(1) <= sig.shift(1)))
add('MACD柱创20日最低后回升', (macd - sig).shift(3) <= (macd - sig).rolling(20).min())
print('震荡指标', f'{time.time()-t0:.0f}s', flush=True)

# ---------- 乖离率 / 统计 ----------
for n, ths in ((5, (-3, -5, -8)), (10, (-5, -8, -12)), (20, (-8, -12, -15, -20)), (60, (-15, -20, -25))):
    bias = (C / C.rolling(n).mean() - 1) * 100
    for th in ths: add(f'BIAS{n}<{th}%', bias < th)
add('BIAS5/10/20全负', (C < ma[5]) & (C < ma[10]) & (C < ma[20]))
for th in (-25, -30, -40): add(f'距MA200<{th}%', C / ma[200] - 1 < th / 100)
z20 = (C - ma[20]) / C.rolling(20).std()
for th in (-2, -2.5, -3): add(f'Z20<{th}', z20 < th)
p252 = C.rolling(252, min_periods=252).rank(pct=True); add('一年价格分位<5%', p252 < 0.05); add('一年价格分位<2%', p252 < 0.02)
sd20 = C.rolling(20).std()
add('BB(2.5)下轨下', C < ma[20] - 2.5 * sd20); add('BB(3)下轨下', C < ma[20] - 3 * sd20)
add('%B<0连续2天', (C < ma[20] - 2 * sd20) & (pc < ma[20].shift(1) - 2 * sd20.shift(1)))
add('收回BB下轨上', (C > ma[20] - 2 * sd20) & (pc < ma[20].shift(1) - 2 * sd20.shift(1)))
bbw = 4 * sd20 / ma[20]; add('BB带宽扩张1.5x', bbw > 1.5 * bbw.rolling(120).mean())
for n, ths in ((10, (-10, -15)), (20, (-15, -20, -30)), (60, (-20, -30)), (120, (-30,))):
    roc = (C / C.shift(n) - 1) * 100
    for th in ths: add(f'{n}日跌>{-th}%', roc < th)
r252 = C / C.shift(252) - 1
add('一年涨幅>0', r252 > 0); add('一年涨幅>20%', r252 > 0.2); add('一年涨幅>50%', r252 > 0.5); add('一年跌>30%', r252 < -0.3)
for n, ths in ((20, (-10, -15, -20)), (252, (-30, -40, -50))):
    dd = C / H.rolling(n).max() - 1
    for th in ths: add(f'距{n}日高点<{th}%', dd < th / 100)
add('2日跌>5%', C / C.shift(2) - 1 < -0.05); add('2日跌>8%', C / C.shift(2) - 1 < -0.08)
add('今天创20日新低', L <= L.rolling(20).min()); add('今天创252日新低', L <= L.rolling(252).min())
add('破20日低后收回', (L < L.shift(1).rolling(20).min()) & (C > L.shift(1).rolling(20).min()))
add('连续3天更低的低点', (L < pl) & (pl < pl.shift(1)) & (pl.shift(1) < pl.shift(2)))
print('乖离统计', f'{time.time()-t0:.0f}s', flush=True)

# ---------- K 线 ----------
ibs = (C - L) / rng
for th in (0.1, 0.2, 0.3): add(f'IBS<{th}', ibs < th)
for th in (0.5, 0.7, 0.8): add(f'IBS>{th}', ibs > th)
body = (C - O).abs(); lsh = np.fmin(O, C) - L; ush = H - np.fmax(O, C)
add('长下影>2倍实体', (lsh > 2 * body) & (lsh > 0.5 * rng)); add('长上影>2倍实体', (ush > 2 * body) & (ush > 0.5 * rng))
add('阳线', C > O); add('大阳线>2%', C / O - 1 > 0.02); add('阴线', C < O)
add('收盘>昨日最高', C > ph); add('收盘>昨日开盘', C > po); add('内包日', (H < ph) & (L > pl))
add('外包阳线', (H > ph) & (L < pl) & (C > O)); add('开盘<昨日最低', O < pl)
add('跳空低开>2%', O / pc - 1 < -0.02); add('跳空低开>3%', O / pc - 1 < -0.03); add('跳空高开', O > pc)
add('跳空低开>2%收正', (O / pc - 1 < -0.02) & (C > O))
add('振幅>1.5ATR', tr > 1.5 * atr.shift(1)); add('振幅>2ATR', tr > 2 * atr.shift(1)); add('振幅<0.7ATR', tr < 0.7 * atr.shift(1))
add('NR7', rng <= rng.rolling(7).min()); add('NR4', rng <= rng.rolling(4).min())
add('收盘在上半区', ibs > 0.5)
add('昨天大阴今天阳', (pc / po - 1 < -0.02) & (C > O))
add('刺透形态', (pc < po) & (O < pc) & (C > (po + pc) / 2) & (C < po))
add('早晨之星近似', (C.shift(2) < O.shift(2)) & ((pc - po).abs() < 0.3 * (ph - pl)) & (C > O) & (C > (O.shift(2) + C.shift(2)) / 2))
print('K线', f'{time.time()-t0:.0f}s', flush=True)

# ---------- 趋势 / 均线 ----------
add('MA50>MA200', ma[50] > ma[200]); add('MA50<MA200', ma[50] < ma[200])
add('MA200上升', ma[200] > ma[200].shift(20)); add('MA200下降', ma[200] < ma[200].shift(20))
add('MA50上升', ma[50] > ma[50].shift(10)); add('收盘>MA100', C > ma[100]); add('MA20<MA50', ma[20] < ma[50])
add('MA5<MA10<MA20空头排列', (ma[5] < ma[10]) & (ma[10] < ma[20]) & (ma[20] < ma[50]))
add('MA200上方5%内', (C > ma[200]) & (C < 1.05 * ma[200])); add('收盘<MA200', C < ma[200])
add('收盘>MA5', C > ma[5]); add('站回MA5', (C > ma[5]) & (pc < ma[5].shift(1)))

# ---------- 量能 ----------
vma = V.rolling(20).mean(); vr = V / vma
add('放量3x', vr > 3); add('缩量0.5x', vr < 0.5); add('阳线放量1.5x', (C > O) & (vr > 1.5)); add('阴线放量1.5x', (C < O) & (vr > 1.5))
add('量比昨日放大', V > V.shift(1)); add('连续3天缩量', (V < V.shift(1)) & (V.shift(1) < V.shift(2)) & (V.shift(2) < V.shift(3)))
add('5日均量>20日均量1.3x', V.rolling(5).mean() > 1.3 * vma)
obv = (np.sign(ret1) * V).cumsum()
add('OBV创20日新低', obv <= obv.rolling(20).min()); add('OBV底背离', (C <= C.rolling(20).min()) & (obv > obv.rolling(20).min()))
mfm = ((C - L) - (H - C)) / rng; cmf = (mfm * V).rolling(20).sum() / V.rolling(20).sum()
add('CMF<-0.2', cmf < -0.2); add('CMF>0.1', cmf > 0.1)
dvol = (C * V).rolling(20).mean()
add('日成交额>5亿美元', dvol > 5e8); add('日成交额<5000万美元', dvol < 5e7)
add('股价<20', C < 20); add('股价>100', C > 100); add('股价>300', C > 300)

# ---------- 波动 ----------
atrp = atr / C
add('ATR%>4%', atrp > 0.04); add('ATR%>6%', atrp > 0.06); add('ATR%<2%', atrp < 0.02)
hv20, hv100 = ret1.rolling(20).std(), ret1.rolling(100).std()
add('短期波动>长期1.5x', hv20 > 1.5 * hv100); add('短期波动<长期0.8x', hv20 < 0.8 * hv100)
print('趋势量能波动', f'{time.time()-t0:.0f}s', flush=True)

# ---------- 相对大盘 ----------
sp = spy.reindex(C.index).ffill()
for n, ths in ((5, (-5, -8)), (20, (-10, -15, -20)), (60, (-20,))):
    rel = C.pct_change(n).sub(sp.pct_change(n), axis=0) * 100
    for th in ths: add(f'{n}日跑输SPY>{-th}%', rel < th)
rel1 = ret1.sub(sp.pct_change(), axis=0); add('今天跑赢SPY', rel1 > 0); add('今天跑赢SPY>2%', rel1 > 0.02)

# ---------- 大盘环境（同一天） ----------
spma200, spma50 = sp.rolling(200).mean(), sp.rolling(50).mean()
addm('SPY>MA200', sp > spma200); addm('SPY<MA200', sp < spma200); addm('SPY>MA50', sp > spma50); addm('SPY<MA50', sp < spma50)
spr2 = rsi(sp, 2); addm('SPY_RSI2<10', spr2 < 10); addm('SPY_RSI2>70', spr2 > 70)
spr1 = sp.pct_change(); addm('SPY今天跌>1%', spr1 < -0.01); addm('SPY今天涨', spr1 > 0); addm('SPY今天涨>1%', spr1 > 0.01)
sp5 = sp.pct_change(5); addm('SPY5日跌>3%', sp5 < -0.03); addm('SPY5日涨', sp5 > 0)
addm('SPY距高点<-10%', sp / sp.rolling(252).max() - 1 < -0.10); addm('SPY一年涨>10%', sp.pct_change(252) > 0.10)
vx = vix.reindex(C.index).ffill()
addm('VIX<15', vx < 15); addm('VIX15-20', (vx >= 15) & (vx < 20)); addm('VIX20-25', (vx >= 20) & (vx < 25)); addm('VIX25-30', (vx >= 25) & (vx < 30))
addm('VIX今天涨>10%', vx.pct_change() > 0.10); addm('VIX今天跌', vx.pct_change() < 0); addm('VIX>10日均值', vx > vx.rolling(10).mean())
addm('VIX>20日均值1.2x', vx > 1.2 * vx.rolling(20).mean())
mem = pd.read_parquet(os.path.join(P, 'member_days.parquet')); mem = mem[mem.date <= END]
Mb = pd.DataFrame(False, index=C.index, columns=C.columns)
mi, mk = C.index.get_indexer(mem.date), C.columns.get_indexer(mem.ticker); ok = (mi >= 0) & (mk >= 0)
mv = Mb.values; mv[mi[ok], mk[ok]] = True; Mb = pd.DataFrame(mv, index=C.index, columns=C.columns)
br50 = ((C > ma[50]) & Mb).sum(1) / Mb.sum(1); br200 = ((C > ma[200]) & Mb).sum(1) / Mb.sum(1)
addm('宽度:MA50上方<20%', br50 < 0.2); addm('宽度:MA50上方<35%', br50 < 0.35); addm('宽度:MA50上方>60%', br50 > 0.6)
addm('宽度:MA200上方<40%', br200 < 0.4); addm('宽度:MA200上方>60%', br200 > 0.6)
adv = ((ret1 > 0) & Mb).sum(1) / Mb.sum(1); addm('当天上涨家数>60%', adv > 0.6); addm('当天上涨家数<30%', adv < 0.3)
nsig = s.groupby('signal_day').size(); ns = s.signal_day.map(nsig).values
F['同日信号只1个'] = ns == 1; F['同日信号≥5个'] = ns >= 5; F['同日信号≥15个'] = ns >= 15

# ---------- 日历 ----------
d = s.signal_day
for i, n in enumerate(['周一', '周二', '周三', '周四', '周五']): F[n] = (d.dt.weekday == i).values
ix = pd.Series(C.index, index=C.index); g = ix.groupby([ix.dt.year, ix.dt.month])
tdm = g.cumcount().reindex(d).values; tdr = (g.cumcount(ascending=False)).reindex(d).values
F['月初5个交易日'] = tdm < 5; F['月末5个交易日'] = tdr < 5; F['月中'] = (tdm >= 5) & (tdr >= 5)
F['1月'] = (d.dt.month == 1).values; F['9-10月'] = d.dt.month.isin([9, 10]).values; F['11-12月'] = d.dt.month.isin([11, 12]).values
F['5-6月'] = d.dt.month.isin([5, 6]).values; F['季末月'] = d.dt.month.isin([3, 6, 9, 12]).values
fri = d + pd.to_timedelta(4 - d.dt.weekday, unit='D'); F['期权到期周'] = ((fri.dt.day >= 15) & (fri.dt.day <= 21)).values
DAY |= {'同日信号只1个','同日信号≥5个','同日信号≥15个','周一','周二','周三','周四','周五','月初5个交易日','月末5个交易日','月中','1月','9-10月','11-12月','5-6月','季末月','期权到期周'}
SUF = '_stock' if os.environ.get('EXCL_DAY') else ''
if SUF:
    for k in DAY: F.pop(k, None)
print('全部条件', len(F), f'{time.time()-t0:.0f}s', flush=True)

# ---------- 收益 ----------
names = list(F); X = np.column_stack([F[n] for n in names]).astype(np.float32)
fwd = {n: ((spy.shift(-n) / spy - 1) * 100).reindex(s.entry_day).values for n in (5, 10)}
x10 = s.r10.values - COST - fwd[10]; x5 = s.r5.values - COST - fwd[5]
tr_ = (s.signal_day < pd.Timestamp('2025-12-01')).values & ~np.isnan(x10)
te = (s.year == 2026).values & ~np.isnan(x10)
ty = (s.signal_day.dt.year + s.signal_day.dt.dayofyear / 366).values; w = 0.5 ** ((2026 - ty) / HL)
prev = X[tr_].mean(0); keepc = (prev > 0.003) & (prev < 0.99) & (X[tr_].sum(0) >= MINN)
names = [n for n, k in zip(names, keepc) if k]; X = X[:, keepc]; NC = len(names)
pd.DataFrame({'条件': names, '训练出现比例%': (prev[keepc] * 100).round(1)}).to_csv(os.path.join(HERE, 'big_conditions_list' + SUF + '.csv'), index=False, encoding='utf-8-sig')
print('可用条件', NC, '训练信号', tr_.sum(), '测试信号', te.sum(), flush=True)

Xt = X[tr_]; wt = w[tr_].astype(np.float32); xt = x10[tr_].astype(np.float32); G = s.signal_day.values[tr_]

def enum(xx):
    A = np.vstack([wt, wt * xx, wt * xx * xx, wt * wt, np.ones_like(wt)]).astype(np.float32)
    keys, st = [], []
    for a in range(NC):
        keys.append((a,)); st.append(A @ Xt[:, a:a + 1])
    for a in range(NC):
        if a + 1 < NC:
            m = Xt[:, a:a + 1] * Xt[:, a + 1:]; keys += [(a, b) for b in range(a + 1, NC)]; st.append(A @ m)
    for a, b in itertools.combinations(range(NC), 2):
        m2 = Xt[:, a] * Xt[:, b]
        if m2.sum() < MINN or b + 1 >= NC: continue
        keys += [(a, b, c) for c in range(b + 1, NC)]; st.append(A @ (Xt[:, b + 1:] * m2[:, None]))
    st = np.hstack(st).astype(np.float64)
    W, Wx, Wxx, W2, n = st
    mu = Wx / np.maximum(W, 1e-12); var = np.clip(Wxx / np.maximum(W, 1e-12) - mu ** 2, 1e-9, None)
    t = mu / np.sqrt(var / np.maximum(W ** 2 / np.maximum(W2, 1e-12), 1)); t[n < MINN] = -np.inf
    return keys, t, mu, n

keys, t, mu, n = enum(xt)
print('组合总数', f'{len(keys):,}', '≥30 笔', f'{int(np.isfinite(t).sum()):,}', f'{time.time()-t0:.0f}s', flush=True)
order = np.argsort(-t); seen, top = set(), []
for i in order:
    m = np.all(Xt[:, list(keys[i])] > 0, axis=1); h = hash(m.tobytes())
    if h in seen: continue
    seen.add(h); top.append(i)
    if len(top) == 100: break

def ctw(m):
    ww = wt[m]; xx = xt[m]; W = ww.sum(); mu_ = (ww * xx).sum() / W
    e = pd.Series(ww * (xx - mu_)).groupby(G[m]).sum().values
    return mu_ / (np.sqrt((e ** 2).sum()) / W)
Xe = X[te]; xe = x10[te]; re10 = s.r10.values[te]; xe5 = x5[te]
out = []; hit = np.zeros(te.sum(), bool)
for j, i in enumerate(top, 1):
    m = np.all(Xt[:, list(keys[i])] > 0, axis=1); me = np.all(Xe[:, list(keys[i])] > 0, axis=1); hit |= me
    out.append({'排名': j, '组合': '＋'.join(names[k] for k in keys[i]), '条件数': len(keys[i]), '训练笔数': int(n[i]),
                '训练加权超额%': round(mu[i], 2), '训练聚类t': round(ctw(m), 1),
                '2026笔数': int(me.sum()), '2026超额%': round(xe[me].mean(), 2) if me.any() else None,
                '2026涨>1%比例': round((re10[me] > 1).mean() * 100) if me.any() else None})
T = pd.DataFrame(out); T.to_csv(os.path.join(HERE, 'big_top100' + SUF + '.csv'), index=False, encoding='utf-8-sig')
new = set(names) - set(ds.COND_LABEL.values())
print('top100 中含新条件的组合', int(T.组合.apply(lambda c: any(z in new for z in c.split('＋'))).sum()))
# 单条件排名
sg = pd.DataFrame({'条件': names, '训练笔数': n[:NC].astype(int), '训练加权超额%': mu[:NC].round(2), 't': t[:NC].round(1),
                   '2026笔数': Xe.sum(0).astype(int),
                   '2026超额%': [round(xe[Xe[:, k] > 0].mean(), 2) if (Xe[:, k] > 0).any() else None for k in range(NC)]}).sort_values('t', ascending=False)
sg.to_csv(os.path.join(HERE, 'big_singles' + SUF + '.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 300)
print('\n单条件前 30：'); print(sg.head(30).to_string(index=False))
print('\ntop100 前 40：'); print(T.head(40).to_string(index=False))
q = hit
print(f'\n2026 全部基础信号 {te.sum()} 笔：超额 {xe.mean():+.2f}%，涨>1% {(re10>1).mean()*100:.0f}%，平均涨 {re10.mean():+.2f}%')
print(f'2026 top100 任一命中 {q.sum()} 笔：超额 {xe[q].mean():+.2f}%，涨>1% {(re10[q]>1).mean()*100:.0f}%，平均涨 {re10[q].mean():+.2f}%（5 天超额 {np.nanmean(xe5[q]):+.2f}%）')
for k_ in (10, 30):
    hk = np.zeros(te.sum(), bool)
    for i in top[:k_]: hk |= np.all(Xe[:, list(keys[i])] > 0, axis=1)
    print(f'2026 top{k_} 任一命中 {hk.sum()} 笔：超额 {xe[hk].mean():+.2f}%，涨>1% {(re10[hk]>1).mean()*100:.0f}%')
# 运气参照：同日打乱 3 次
rng_ = np.random.default_rng(0); idxs = list(pd.Series(np.arange(len(xt))).groupby(G).indices.values()); null = []
for r_ in range(3):
    xx = xt.copy()
    for ii in idxs: xx[ii] = xx[rng_.permutation(ii)]
    _, tn, _, _ = enum(xx); tn = np.sort(tn[np.isfinite(tn)])[::-1]; null.append((tn[0], tn[99]))
    print('打乱', r_, f'{time.time()-t0:.0f}s', flush=True)
null = np.array(null); tr_sorted = np.sort(t[np.isfinite(t)])[::-1]
print(f'真实：第1名 t={tr_sorted[0]:.1f}，第100名 t={tr_sorted[99]:.1f}｜纯运气：第1名 {null[:,0].mean():.1f}，第100名 {null[:,1].mean():.1f}')
print('done', f'{time.time()-t0:.0f}s')
