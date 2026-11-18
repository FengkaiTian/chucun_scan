"""agentB 特征：个股入场基础集 + 市场状态库。只用 t 日及以前的数据。"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from lab_common import Lab

CACHE = os.path.join(HERE, 'cache'); os.makedirs(CACHE, exist_ok=True)


def sh(a, k=1):
    """向后移 k 天（用过去的值），前 k 行 NaN"""
    out = np.full_like(a, np.nan, dtype=float)
    out[k:] = a[:-k]
    return out


def base_signals(lab):
    C, O, H, L, V = lab.C, lab.O, lab.H, lab.L, lab.V
    df = lab.df
    S = {}
    rsi14 = lab.rsi(14); rsi2 = lab.rsi(2); r1 = sh(rsi14)
    h = lab.macd_hist(); h1 = sh(h); h2 = sh(h, 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        macd_shrink = (h < 0) & (h > h1) & (h1 > h2)
        S['MACD收窄+RSI回升(用户基础)'] = macd_shrink & (r1 < 30) & (rsi14 > r1)
        S['MACD零下3根收窄'] = macd_shrink
        S['RSI2<5'] = rsi2 < 5
        S['RSI2<10'] = rsi2 < 10
        ma200 = lab.ma(200); ma20 = lab.ma(20); ma50 = lab.ma(50)
        S['RSI2<10且>MA200'] = (rsi2 < 10) & (C > ma200)
        S['RSI14<30'] = rsi14 < 30
        S['RSI14<30回升'] = (r1 < 30) & (rsi14 > r1)
        sd20 = df(C).rolling(20).std().values
        S['跌破布林下轨'] = C < ma20 - 2 * sd20
        r5 = lab.ret(5); r3 = lab.ret(3); r1d = lab.ret(1)
        S['5日跌>10%'] = r5 < -10
        S['5日跌>5%'] = r5 < -5
        S['3日跌>8%'] = r3 < -8
        S['距MA20<-10%'] = C / ma20 - 1 < -0.10
        mn20 = df(C).rolling(20).min().values; mx20p = sh(df(C).rolling(20).max().values)
        mn252 = df(C).rolling(252).min().values; mx252p = sh(df(C).rolling(252).max().values)
        S['20日新低'] = C <= mn20
        S['52周新低'] = C <= mn252
        S['20日新高突破'] = C > mx20p
        S['52周新高突破'] = C > mx252p
        C1 = sh(C); L1 = sh(L)
        S['跳空低开收阳'] = (O < C1 * 0.98) & (C > O)
        rng = H - L
        S['十字星底'] = (np.abs(C - O) <= 0.1 * rng) & (rng > 0) & (r5 < -5) & (L <= df(L).rolling(10).min().values)
        lower = np.minimum(O, C) - L; body = np.abs(C - O)
        S['锤子线(跌后)'] = (lower > 2 * body) & (lower > 0.6 * rng) & (r5 < -3)
        dn = (C < C1).astype(float); dn[np.isnan(C1)] = 0
        run = df(dn).rolling(4).sum().values
        S['连跌4天'] = run >= 4
        vma = sh(df(V).rolling(20).mean().values)
        S['放量大跌(V>2均,跌>4%)'] = (V > 2 * vma) & (r1d < -4)
        S['放量突破20日高'] = (V > 2 * vma) & (C > mx20p)
        ibs = (C - L) / np.where(rng > 0, rng, np.nan)
        S['IBS<0.1且跌'] = (ibs < 0.1) & (r1d < 0)
        ma5 = lab.ma(5)
        S['MA5上穿MA20'] = (ma5 > ma20) & (sh(ma5) <= sh(ma20))
        dif_cross = (h > 0) & (h1 <= 0)
        S['MACD金叉(零下)'] = dif_cross & (C < ma50)
        # 横截面：只在当期成分股中排名
        mem = lab.member
        def xs_rank(x):
            x = np.where(mem & np.isfinite(x), x, np.nan)
            return df(x).rank(axis=1, pct=True).values
        r126 = lab.ret(126)
        S['动量前10%(126日)'] = xs_rank(r126) >= 0.9
        S['5日收益后5%(反转)'] = xs_rank(r5) <= 0.05
        atr = lab.atr_pct(14)
        S['低波新高(ATR后30%+20日新高)'] = (xs_rank(atr) <= 0.3) & (C > mx20p)
    names = list(S)
    arr = np.stack([np.nan_to_num(S[k]).astype(bool) & mem for k in names])
    return names, arr


def market_states(lab):
    """返回 dict 名称 -> 长度 T 的 bool 数组"""
    mem = lab.member; C = lab.C
    spx = pd.Series(lab.spx, index=lab.dates); vix = pd.Series(lab.vix, index=lab.dates)
    S = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for n in (50, 100, 200):
            d = lab.spx_series('dist_ma', n)
            S[f'SPX>MA{n}'] = d > 0; S[f'SPX<MA{n}'] = d < 0
        d200 = lab.spx_series('dist_ma', 200); d50 = lab.spx_series('dist_ma', 50)
        S['SPX距MA200<-5%'] = d200 < -5; S['SPX距MA200<-10%'] = d200 < -10; S['SPX距MA200>+8%'] = d200 > 8
        S['SPX距MA50<-5%'] = d50 < -5
        for n in (5, 20, 60):
            r = lab.spx_series('ret', n); S[f'SPX{n}日涨'] = r > 0; S[f'SPX{n}日跌'] = r < 0
        S['SPX5日跌>3%'] = lab.spx_series('ret', 5) < -3
        S['SPX20日跌>5%'] = lab.spx_series('ret', 20) < -5
        S['SPX20日涨>5%'] = lab.spx_series('ret', 20) > 5
        S['SPX1日跌>2%'] = lab.spx_series('ret', 1) < -2
        dn = (spx < spx.shift(1)).astype(int)
        grp = (dn == 0).cumsum(); streak = dn.groupby(grp).cumsum().values
        S['SPX连跌≥3天'] = streak >= 3; S['SPX连跌≥5天'] = streak >= 5
        dsp = spx.diff(); g = dsp.clip(lower=0).ewm(com=1, min_periods=2).mean(); l = (-dsp.clip(upper=0)).ewm(com=1, min_periods=2).mean()
        S['SPX RSI2<10'] = (100 - 100 / (1 + g / l)).values < 10
        S['SPX 52周新高'] = (spx >= spx.rolling(252).max()).values
        # VIX
        v = vix.values
        S['VIX<15'] = v < 15; S['VIX15-20'] = (v >= 15) & (v < 20); S['VIX20-30'] = (v >= 20) & (v < 30)
        S['VIX>30'] = v > 30; S['VIX>40'] = v > 40
        vr = (vix / vix.rolling(20).mean()).values
        S['VIX/MA20>1.2'] = vr > 1.2; S['VIX/MA20<0.9'] = vr < 0.9
        v5 = (vix / vix.shift(5) - 1).values * 100
        S['VIX5日升>20%'] = v5 > 20; S['VIX5日降>15%'] = v5 < -15
        S['VIX从10日高回落20%'] = (vix / vix.rolling(10).max()).values < 0.8
        S['VIX高位回落(>25且1日降>5%)'] = (v > 25) & ((vix / vix.shift(1) - 1).values < -0.05)
        # 宽度（只用当期成分股）
        def frac(cond, valid):
            num = (cond & valid & mem).sum(1); den = (valid & mem).sum(1)
            return np.where(den > 50, num / np.maximum(den, 1), np.nan)
        ma50 = lab.ma(50); ma200 = lab.ma(200)
        p50 = frac(C > ma50, np.isfinite(ma50)); p200 = frac(C > ma200, np.isfinite(ma200))
        S['宽度:站上MA50<20%'] = p50 < 0.2; S['宽度:站上MA50 20-40%'] = (p50 >= 0.2) & (p50 < 0.4)
        S['宽度:站上MA50>60%'] = p50 > 0.6; S['宽度:站上MA50>80%'] = p50 > 0.8
        S['宽度:站上MA200<30%'] = p200 < 0.3; S['宽度:站上MA200>70%'] = p200 > 0.7
        p50_5 = p50 - sh(p50, 5)
        S['宽度:MA50比例5日升>15点'] = p50_5 > 0.15; S['宽度:MA50比例5日降>15点'] = p50_5 < -0.15
        mn20 = lab.df(C).rolling(20).min().values; mx20 = lab.df(C).rolling(20).max().values
        nl = frac(C <= mn20, np.isfinite(mn20)); nh = frac(C >= mx20, np.isfinite(mx20))
        S['宽度:20日新低>10%'] = nl > 0.10; S['宽度:20日新低>25%'] = nl > 0.25
        S['宽度:20日新高>15%'] = nh > 0.15; S['宽度:新高>新低'] = nh > nl
        rs = lab.rsi(14)
        r30 = frac(rs < 30, np.isfinite(rs))
        S['宽度:RSI<30>10%'] = r30 > 0.10; S['宽度:RSI<30>20%'] = r30 > 0.20
        C1 = sh(C); adv = frac(C > C1, np.isfinite(C1))
        S['宽度:上涨家数<25%'] = adv < 0.25; S['宽度:上涨家数>75%'] = adv > 0.75
        adv5 = pd.Series(adv).rolling(5).mean().values
        S['宽度:5日均上涨<40%'] = adv5 < 0.40; S['宽度:5日均上涨>60%'] = adv5 > 0.60
        adv10 = pd.Series(adv).rolling(10).mean().values
        S['宽度:推力(10日均从<40%到>60%)'] = (adv10 > 0.6) & (pd.Series(adv10).rolling(10).min().values < 0.4)
        # 日历
        dts = lab.dates
        for i, nm in enumerate(['周一', '周二', '周三', '周四', '周五']):
            S[f'星期:{nm}'] = dts.weekday.values == i
        for m in range(1, 13):
            S[f'月份:{m}月'] = dts.month.values == m
        ym = dts.year.values * 12 + dts.month.values
        pos = pd.Series(1, index=dts).groupby(ym).cumsum().values           # 当月第几个交易日
        cnt = pd.Series(1, index=dts).groupby(ym).transform('size').values   # 当月交易日数（日历已知）
        S['月初(前3日)'] = pos <= 3; S['月末(后3日)'] = pos > cnt - 3
        S['月中(第8-15日)'] = (pos >= 8) & (pos <= 15)
    out = {k: np.nan_to_num(np.asarray(v, dtype=float)).astype(bool) for k, v in S.items()}
    return out


# 组合用的关键状态（两两配对）
KEY_STATES = ['SPX>MA200', 'SPX<MA200', 'SPX距MA200<-10%', 'SPX5日跌>3%', 'SPX连跌≥3天',
              'VIX<15', 'VIX>30', 'VIX5日升>20%', 'VIX从10日高回落20%',
              '宽度:站上MA50<20%', '宽度:站上MA50>80%', '宽度:20日新低>10%', '宽度:RSI<30>20%',
              '宽度:上涨家数<25%', '月末(后3日)', '月初(前3日)',
              'SPX<MA50', 'VIX/MA20>1.2', '宽度:站上MA200>70%', '宽度:新高>新低', 'SPX20日涨>5%']


def build():
    lab = Lab()
    names, arr = base_signals(lab)
    np.save(os.path.join(CACHE, 'base.npy'), arr)
    st = market_states(lab)
    pd.DataFrame(st, index=lab.dates).to_parquet(os.path.join(CACHE, 'states.parquet'))
    pd.Series(names).to_csv(os.path.join(CACHE, 'base_names.csv'), index=False)
    tr = (lab.year >= 2000) & (lab.year <= 2019)
    print('base signals:', len(names))
    for k, a in zip(names, arr):
        print(f'  {k}: train信号数 {a[tr].sum()}')
    print('states:', len(st))
    for k, v in st.items():
        print(f'  {k}: train天数 {v[tr].sum()}')


if __name__ == '__main__':
    build()
