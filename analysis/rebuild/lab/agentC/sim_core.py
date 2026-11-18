"""自建模拟器（规则与 lab_common._trades_custom 一致，额外支持和入场价有关的出场、分批卖、网格加仓）。
出场规格 leg = [k, minhold, tp, sl, trail, atp, asl, tsd, tsr, maxh]（全部按收盘价判断，当天收盘卖）
  k: 指标条件下标(-1 无)；minhold: 持有满 m 天后才看指标；tp/sl: 止盈/止损 %；trail: 从持有期最高收盘回撤 %；
  atp/asl: 信号日 ATR14% 的倍数止盈/止损；tsd/tsr: 第 tsd 天收益 < tsr% 就卖（时间止损）；maxh: 最长持有。
"""
import numpy as np
from numba import njit

NL = 10


@njit(cache=True)
def _leg_exit(ex, C, atr_e, j, e, a, p):
    k = int(p[0]); mh = int(p[1]); tp = p[2]; sl = p[3]; tr = p[4]; atp = p[5]; asl = p[6]
    tsd = int(p[7]); tsr = p[8]; maxh = int(p[9])
    peak = a
    for u in range(e + 1, e + maxh + 1):
        c = C[u, j]
        if k >= 0 and (u - e) >= mh and ex[k, u, j]:
            return u
        if not (c > 0):
            continue
        rr = (c / a - 1) * 100
        if c > peak:
            peak = c
        if tp > 0 and rr >= tp:
            return u
        if sl > 0 and rr <= -sl:
            return u
        if tr > 0 and c <= peak * (1 - tr / 100):
            return u
        if atp > 0 and rr >= atp * atr_e:
            return u
        if asl > 0 and rr <= -asl * atr_e:
            return u
        if tsd > 0 and (u - e) == tsd and rr < tsr:
            return u
    return e + maxh


@njit(cache=True)
def sim(sig, ex, C, atr, spx, p1, p2, partial, lag):
    T, N = sig.shape
    maxh = int(p1[9])
    if partial and int(p2[9]) > maxh:
        maxh = int(p2[9])
    cap = 2000000
    out_t = np.empty(cap, np.int64); out_r = np.empty(cap); out_d = np.empty(cap); out_s = np.empty(cap)
    k = 0
    for j in range(N):
        busy = -1
        for t in range(T - lag - maxh):
            if t <= busy or not sig[t, j]:
                continue
            e = t + lag; a = C[e, j]
            if not (a > 0):
                continue
            atr_e = atr[t, j]
            x1 = _leg_exit(ex, C, atr_e, j, e, a, p1)
            b1 = C[x1, j]
            if not (b1 > 0):
                continue
            r = (b1 / a - 1) * 100; d = float(x1 - e); s = (spx[x1] / spx[e] - 1) * 100; xl = x1
            if partial:
                x2 = _leg_exit(ex, C, atr_e, j, e, a, p2)
                b2 = C[x2, j]
                if not (b2 > 0):
                    continue
                r = 0.5 * r + 0.5 * (b2 / a - 1) * 100
                d = 0.5 * d + 0.5 * (x2 - e)
                s = 0.5 * s + 0.5 * (spx[x2] / spx[e] - 1) * 100
                if x2 > xl:
                    xl = x2
            if k >= cap:
                break
            out_t[k] = t; out_r[k] = r; out_d[k] = d; out_s[k] = s
            k += 1
            busy = xl - lag
    return out_t[:k], out_r[:k], out_d[:k], out_s[:k]


@njit(cache=True)
def sim_grid(sig, C, spx, X, K, Y, partial, maxh, lag, cost):
    """网格/加仓：t+1 收盘买 1 份；之后收盘价 ≤ 上次买价×(1−X%) 就再买 1 份（最多 K 次）；
    收盘价 ≥ 均价×(1+Y%) 全卖（partial: Y% 卖一半，2Y% 卖剩下）；到 maxh 天全卖。
    收益 = 总盈亏 / (1+K) 份预留资金；成本按实际买入份数×0.10% 计，返回值已加回 cost（_summ 会统一减 cost）。"""
    T, N = sig.shape
    cap = 2000000
    out_t = np.empty(cap, np.int64); out_r = np.empty(cap); out_d = np.empty(cap); out_s = np.empty(cap)
    k = 0
    for j in range(N):
        busy = -1
        for t in range(T - lag - maxh):
            if t <= busy or not sig[t, j]:
                continue
            e = t + lag; a = C[e, j]
            if not (a > 0):
                continue
            # 每份 = 1 元资金；sh 持股数，inv 当前持仓的成本
            sh = 1.0 / a; inv = 1.0; last = a; adds = 0; used = 1.0
            pnl = 0.0; half_done = False
            x = e + maxh
            for u in range(e + 1, e + maxh + 1):
                c = C[u, j]
                if not (c > 0):
                    continue
                avg = inv / sh
                if partial and (not half_done) and c >= avg * (1 + Y / 100):
                    pnl += 0.5 * (sh * c - inv)
                    sh *= 0.5; inv *= 0.5; half_done = True
                    continue
                tgt = Y * 2 if (partial and half_done) else Y
                if c >= avg * (1 + tgt / 100):
                    x = u; break
                if adds < K and c <= last * (1 - X / 100):
                    sh += 1.0 / c; inv += 1.0; last = c; adds += 1; used += 1
            b = C[x, j]
            if not (b > 0):
                continue
            pnl += sh * b - inv
            r = pnl / (1 + K) * 100 - cost * used / (1 + K) + cost
            if k >= cap:
                break
            out_t[k] = t; out_r[k] = r; out_d[k] = x - e; out_s[k] = (spx[x] / spx[e] - 1) * 100
            k += 1
            busy = x - lag
    return out_t[:k], out_r[:k], out_d[:k], out_s[:k]
