"""agentA 原子条件库：build_atoms(lab) -> list of (id, 类别, 中文名, T×N bool)
所有条件只用 t 日及以前的数据（shift 只向过去）。"""
import numpy as np, pandas as pd


def _sh(a, k=1):
    """向过去平移：out[t] = a[t-k]（前 k 行 NaN）"""
    out = np.full_like(a, np.nan, dtype=np.float64)
    out[k:] = a[:-k]
    return out


def build_atoms(lab):
    C, O, H, L, V = lab.C, lab.O, lab.H, lab.L, lab.V
    df = lambda a: pd.DataFrame(a)
    roll = lambda a, n: df(a).rolling(n, min_periods=n)
    A = []

    def add(cat, name, m):
        m = np.asarray(m, dtype=bool) & np.isfinite(C)
        A.append((f'A{len(A):03d}', cat, name, m))

    with np.errstate(invalid='ignore', divide='ignore'):
        C1 = _sh(C); O1 = _sh(O)
        rsi2, rsi3, rsi14 = lab.rsi(2), lab.rsi(3), lab.rsi(14)
        ma = {n: roll(C, n).mean().values for n in (5, 10, 20, 50, 200)}
        sd20 = roll(C, 20).std().values
        r1 = (C / C1 - 1) * 100
        ret = {n: (C / _sh(C, n) - 1) * 100 for n in (5, 10, 20, 60, 120)}
        mx = {n: roll(C, n).max().values for n in (10, 20, 55, 252)}
        mn = {n: roll(C, n).min().values for n in (10, 20, 60, 252)}
        rng_ = H - L
        body = np.abs(C - O)
        lowsh = np.minimum(O, C) - L
        upsh = H - np.maximum(O, C)
        clv = np.where(rng_ > 0, (C - L) / rng_, np.nan)
        vma20 = roll(V, 20).mean().values
        vr = V / _sh(vma20)                      # 当日量 / 之前20日均量
        atr14 = lab.atr_pct(14); atr5 = lab.atr_pct(5); atr50 = lab.atr_pct(50)
        bbw = 4 * sd20 / ma[20] * 100
        macd = lab.macd_hist()

        # 连跌/连涨天数
        dn = np.zeros_like(C); up = np.zeros_like(C)
        d = C < C1; u = C > C1
        for t in range(1, len(C)):
            dn[t] = np.where(d[t], dn[t - 1] + 1, 0)
            up[t] = np.where(u[t], up[t - 1] + 1, 0)

        # ── 超卖 / 均值回归 ──
        cat = '超卖'
        for k in (5, 10, 20): add(cat, f'RSI2<{k}', rsi2 < k)
        for k in (10, 20): add(cat, f'RSI3<{k}', rsi3 < k)
        for k in (25, 30, 35): add(cat, f'RSI14<{k}', rsi14 < k)
        for k in (2.0, 2.5, 3.0): add(cat, f'收盘<布林下轨(20,{k})', C < ma[20] - k * sd20)
        for n in (10, 20, 60, 252): add(cat, f'收盘创{n}日新低', C <= mn[n])
        for k in (3, 4, 5): add(cat, f'连跌≥{k}天', dn >= k)
        for k in (5, 8, 12): add(cat, f'距MA20<-{k}%', C / ma[20] - 1 < -k / 100)
        for k in (15, 25): add(cat, f'距MA200<-{k}%', C / ma[200] - 1 < -k / 100)
        for k in (8, 12): add(cat, f'5日跌>{k}%', ret[5] < -k)
        for k in (15, 25): add(cat, f'20日跌>{k}%', ret[20] < -k)
        for k in (3, 5): add(cat, f'单日跌>{k}%', r1 < -k)

        # ── 动量 / 突破 ──
        cat = '动量'
        for n in (20, 55, 252): add(cat, f'收盘创{n}日新高', C >= mx[n])
        for k in (2, 5): add(cat, f'距52周高<{k}%', C >= mx[252] * (1 - k / 100))
        add(cat, 'MACD柱翻正(金叉)', (macd > 0) & (_sh(macd) <= 0))
        add(cat, 'MACD柱>0且上升', (macd > 0) & (macd > _sh(macd)))
        add(cat, 'MACD柱<0但回升', (macd < 0) & (macd > _sh(macd)) & (_sh(macd) > _sh(macd, 2)))
        mem = lab.member
        for n in (20, 60, 120):
            x = np.where(mem, ret[n], np.nan)
            pr = df(x).rank(axis=1, pct=True).values
            add(cat, f'{n}日涨幅横截面前20%', pr >= 0.8)
            add(cat, f'{n}日涨幅横截面后20%', pr <= 0.2)
        spr20 = lab.spx_series('ret', 20)[:, None]
        for k in (10, 20): add(cat, f'20日跑赢大盘>{k}%', ret[20] - spr20 > k)
        for k in (3, 5): add(cat, f'单日涨>{k}%', r1 > k)
        for k in (70, 80): add(cat, f'RSI14>{k}', rsi14 > k)
        add(cat, '连涨≥4天', up >= 4)

        # ── 成交量 ──
        cat = '成交量'
        for k in (1.5, 2, 3): add(cat, f'放量>{k}倍20日均量', vr > k)
        for k in (0.5, 0.7): add(cat, f'缩量<{k}倍20日均量', vr < k)
        add(cat, '放量2倍且收跌', (vr > 2) & (r1 < 0))
        add(cat, '放量2倍且收涨', (vr > 2) & (r1 > 0))
        add(cat, '缩量0.7倍且收跌', (vr < 0.7) & (r1 < 0))
        v5 = roll(V, 5).mean().values; v50 = roll(V, 50).mean().values
        for k in (1.5,): add(cat, f'5日均量/50日均量>{k}', v5 / v50 > k)
        for k in (0.7,): add(cat, f'5日均量/50日均量<{k}', v5 / v50 < k)
        p20 = ret[20]; vt = (roll(V, 10).mean().values / _sh(roll(V, 10).mean().values, 10))
        add(cat, '量价背离:20日跌但量增', (p20 < -5) & (vt > 1.3))
        add(cat, '量价背离:创20日新高但量缩', (C >= mx[20]) & (vr < 0.8))
        add(cat, '量增3天', (V > _sh(V)) & (_sh(V) > _sh(V, 2)) & (_sh(V, 2) > _sh(V, 3)))

        # ── K 线形态 ──
        cat = 'K线'
        add(cat, '十字星', (rng_ > 0) & (body < 0.1 * rng_))
        add(cat, '锤子线', (rng_ > 0) & (lowsh > 2 * body) & (upsh < body + 0.1 * rng_) & (body > 0))
        add(cat, '看涨吞没', (C > O) & (C1 < O1) & (C >= O1) & (O <= C1))
        add(cat, '看跌吞没', (C < O) & (C1 > O1) & (C <= O1) & (O >= C1))
        for k in (1, 2, 3): add(cat, f'跳空低开>{k}%收阳', (O < C1 * (1 - k / 100)) & (C > O))
        for k in (2,): add(cat, f'跳空高开>{k}%', O > C1 * (1 + k / 100))
        for k in (0.6, 0.7): add(cat, f'长下影>{int(k*100)}%振幅', (rng_ > 0) & (lowsh / rng_ > k))
        for k in (0.1, 0.2): add(cat, f'IBS<{k}(收在低位)', clv < k)
        add(cat, 'IBS>0.9(收在高位)', clv > 0.9)
        add(cat, '内包日', (H < _sh(H)) & (L > _sh(L)))
        add(cat, '外包日收阳', (H > _sh(H)) & (L < _sh(L)) & (C > O))
        for k in (3, 5): add(cat, f'大阴线(收<开{k}%)', C / O - 1 < -k / 100)
        add(cat, '大阳线(收>开3%)', C / O - 1 > 0.03)

        # ── 波动率 ──
        cat = '波动率'
        for k in (0.6, 0.75): add(cat, f'ATR5/ATR50<{k}(收缩)', atr5 / atr50 < k)
        for k in (1.5, 2.0): add(cat, f'ATR5/ATR50>{k}(扩张)', atr5 / atr50 > k)
        bbmin = roll(bbw, 120).min().values
        add(cat, '布林带宽120日最低(挤压)', bbw <= bbmin * 1.05)
        bbq = df(bbw).rolling(252, min_periods=252).quantile(0.9).values
        add(cat, '布林带宽>252日90分位', bbw > bbq)
        nr7 = rng_ <= roll(rng_, 7).min().values
        add(cat, 'NR7(7日最窄振幅)', nr7)
        for k in (1.5,): add(cat, f'ATR14%<{k}(低波)', atr14 < k)
        for k in (3, 5): add(cat, f'ATR14%>{k}(高波)', atr14 > k)
        x = np.where(mem, atr14, np.nan); pr = df(x).rank(axis=1, pct=True).values
        add(cat, 'ATR%横截面前20%', pr >= 0.8)
        add(cat, 'ATR%横截面后20%', pr <= 0.2)

        # ── 趋势状态 ──
        cat = '趋势'
        add(cat, '收盘>MA200', C > ma[200]); add(cat, '收盘<MA200', C < ma[200])
        add(cat, '收盘>MA50', C > ma[50]); add(cat, '收盘<MA50', C < ma[50])
        add(cat, 'MA50>MA200', ma[50] > ma[200]); add(cat, 'MA50<MA200', ma[50] < ma[200])
        add(cat, '均线多头(MA20>MA50>MA200)', (ma[20] > ma[50]) & (ma[50] > ma[200]))
        add(cat, '均线空头(MA20<MA50<MA200)', (ma[20] < ma[50]) & (ma[50] < ma[200]))
        add(cat, 'MA200上行(较20日前)', ma[200] > _sh(ma[200], 20))
        add(cat, 'MA200下行(较20日前)', ma[200] < _sh(ma[200], 20))
        add(cat, '上升趋势回调(>MA200且<MA20)', (C > ma[200]) & (C < ma[20]))
        add(cat, '下降趋势反弹(<MA200且>MA20)', (C < ma[200]) & (C > ma[20]))
        add(cat, '收盘>MA200超20%', C / ma[200] - 1 > 0.2)
    return A
