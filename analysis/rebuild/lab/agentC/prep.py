"""预计算：入场信号 (K1,T,N) 和出场指标条件 (K2,T,N)，存成 npy 供 worker memmap 共享。"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lab_common import Lab

CACHE = sys.argv[1]
os.makedirs(CACHE, exist_ok=True)
t0 = time.time()
lab = Lab()
C, O, H, L = lab.C, lab.O, lab.H, lab.L
T, N = C.shape
sh = lambda a, k=1: np.vstack([np.full((k, a.shape[1]), np.nan), a[:-k]])
vix = lab.vix[:, None]
with np.errstate(invalid='ignore', divide='ignore'):
    h = lab.macd_hist(); h1, h2, h3 = sh(h), sh(h, 2), sh(h, 3)
    r14 = lab.rsi(14); r14_1 = sh(r14); r2 = lab.rsi(2)
    C1, C2, C3 = sh(C), sh(C, 2), sh(C, 3)
    ma = {n: lab.ma(n) for n in (5, 10, 20, 50, 200)}
    sd20 = lab.df(C).rolling(20).std().values
    bbl, bbu = ma[20] - 2 * sd20, ma[20] + 2 * sd20
    rmin = lambda a, n: lab.df(a).rolling(n).min().values
    rmax = lambda a, n: lab.df(a).rolling(n).max().values
    ret5 = lab.ret(5); ret20 = lab.ret(20); ret126 = lab.ret(126)
    rng = H - L
    ibs = np.where(rng > 0, (C - L) / rng, 0.5)
    doji = (rng > 0) & (np.abs(C - O) <= 0.1 * rng)
    dojib = doji & (L <= rmin(L, 10))
    dojib1 = sh(dojib.astype(float)) > 0
    macd3 = (h < 0) & (h > h1) & (h1 > h2) & (h2 > h3)
    macd2 = (h < 0) & (h > h1) & (h1 > h2)
    rsiturn = (r14_1 < 30) & (r14 > r14_1)
    base = macd3 & rsiturn & (vix < 30)
    mom_rank = lab.df(np.where(lab.member, ret126, np.nan)).rank(axis=1, pct=True).values
    spx5 = lab.spx_series('ret', 5)[:, None]
    gc = (ma[50] > ma[200]) & (sh(ma[50]) <= sh(ma[200]))
    E = [
        ('E01 基础信号(MACD3收窄+RSI14回升+VIX<30)', base),
        ('E02 基础无VIX过滤', macd3 & rsiturn),
        ('E03 基础+跳空低开收阳', base & (O < C1) & (C > O)),
        ('E04 基础+十字星底', base & (dojib | dojib1)),
        ('E05 基础RSI<35', macd3 & (r14_1 < 35) & (r14 > r14_1) & (vix < 30)),
        ('E06 基础MACD2根', macd2 & rsiturn & (vix < 30)),
        ('E07 基础+C>MA200', base & (C > ma[200])),
        ('E08 基础VIX<25', macd3 & rsiturn & (vix < 25)),
        ('E09 基础+收阳', base & (C > O)),
        ('E10 RSI2<5', r2 < 5),
        ('E11 RSI2<10', r2 < 10),
        ('E12 RSI2<10+C>MA200', (r2 < 10) & (C > ma[200])),
        ('E13 RSI2<10+VIX<30', (r2 < 10) & (vix < 30)),
        ('E14 跌破布林下轨', C < bbl),
        ('E15 跌破布林下轨+C>MA200', (C < bbl) & (C > ma[200])),
        ('E16 10日新低', C <= rmin(C, 10)),
        ('E17 20日新低', C <= rmin(C, 20)),
        ('E18 252日新低', C <= rmin(C, 252)),
        ('E19 5日跌>5%', ret5 < -5),
        ('E20 5日跌>10%', ret5 < -10),
        ('E21 5日跌>15%', ret5 < -15),
        ('E22 连跌3天', (C < C1) & (C1 < C2) & (C2 < C3)),
        ('E23 IBS<0.15+C>MA200', (ibs < 0.15) & (C > ma[200])),
        ('E24 RSI14<30', r14 < 30),
        ('E25 20日新高突破', C > sh(rmax(C, 20))),
        ('E26 55日新高突破', C > sh(rmax(C, 55))),
        ('E27 252日新高突破', C > sh(rmax(C, 252))),
        ('E28 126日动量前10%+C>MA50', (mom_rank > 0.9) & (C > ma[50])),
        ('E29 20日涨>15%+C>MA20', (ret20 > 15) & (C > ma[20])),
        ('E30 MA50上穿MA200', gc),
        ('E31 零轴下MACD金叉', (h > 0) & (h1 <= 0) & ((lab.df(C).ewm(span=12, adjust=False).mean() - lab.df(C).ewm(span=26, adjust=False).mean()).values < 0)),
        ('E32 跌破下轨+RSI14<30', (C < bbl) & (r14 < 30)),
        ('E33 跳空低开>3%收阳', (O < C1 * 0.97) & (C > O)),
        ('E34 大盘5日跌>3%+RSI2<10', (spx5 < -3) & (r2 < 10)),
    ]
    bad = ((C < ma[5]).astype(int) + (r14 < r14_1) + (h < h1) + (C < O) + (C < C1))
    X = [
        ('C>MA5', C > ma[5]), ('C>MA10', C > ma[10]), ('C>MA20', C > ma[20]), ('C>MA50', C > ma[50]),
        ('RSI14>=50', r14 >= 50), ('RSI14>=60', r14 >= 60), ('RSI14>=70', r14 >= 70),
        ('RSI2>=50', r2 >= 50), ('RSI2>=70', r2 >= 70), ('RSI2>=90', r2 >= 90),
        ('MACD金叉(柱>0)', h > 0), ('MACD柱变差', h < h1), ('MACD柱连2日变差', (h < h1) & (h1 < h2)),
        ('收盘站上前日高', C > sh(H)), ('首个阳线C>C1', C > C1), ('连涨2天', (C > C1) & (C1 > C2)),
        ('IBS>0.8', ibs > 0.8), ('C>布林上轨', C > bbu),
        ('C<MA5', C < ma[5]), ('C<MA10', C < ma[10]), ('C<MA20', C < ma[20]), ('C<MA50', C < ma[50]),
        ('C<前10日最低收盘', C < sh(rmin(C, 10))), ('C<前20日最低收盘', C < sh(rmin(C, 20))),
        ('坏信号>=2', bad >= 2), ('坏信号>=3', bad >= 3), ('坏信号>=4', bad >= 4),
        ('RSI14下穿70', (r14_1 >= 70) & (r14 < 70)), ('RSI14回落', r14 < r14_1), ('首个阴线C<C1', C < C1),
        ('RSI14>=50且柱变差', (r14 >= 50) & (h < h1)), ('RSI2>=70且C>MA5', (r2 >= 70) & (C > ma[5])),
    ]
    atr = lab.atr_pct(14)

ent = np.zeros((len(E), T, N), bool)
for i, (_, s) in enumerate(E):
    ent[i] = np.nan_to_num(s).astype(bool) & lab.member
ex = np.zeros((len(X), T, N), bool)
for i, (_, s) in enumerate(X):
    ex[i] = np.nan_to_num(s).astype(bool)
np.save(os.path.join(CACHE, 'ent.npy'), ent)
np.save(os.path.join(CACHE, 'ex.npy'), ex)
np.save(os.path.join(CACHE, 'C.npy'), C)
np.save(os.path.join(CACHE, 'atr.npy'), np.nan_to_num(atr))
np.save(os.path.join(CACHE, 'spx.npy'), lab.spx)
np.save(os.path.join(CACHE, 'year.npy'), lab.year)
np.save(os.path.join(CACHE, 'member.npy'), lab.member)
# 只统计 2000–2024 的信号密度（不看 2025+）
pre = lab.year < 2025
dens = [float(ent[i][pre].sum() / lab.member[pre].sum()) for i in range(len(E))]
json.dump({'entries': [e[0] for e in E], 'exits': [x[0] for x in X], 'density': dens},
          open(os.path.join(CACHE, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for (n, _), d in zip(E, dens):
    print(f'{n:40s} density={d:.5f}  n_sig(<2025)={d * lab.member[pre].sum():.0f}')
print('prep done', time.time() - t0)
