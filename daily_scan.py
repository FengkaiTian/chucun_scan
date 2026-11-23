"""
Daily Stock Scanner - S&P 500 v5
基于6年(2020-2025)回测，1,626种指标组合穷举验证
只保留T+1/3/5/10全周期跑赢baseline的206种组合
从 winning_combos.json 动态加载信号定义
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
os.environ['MPLBACKEND'] = 'Agg'
import requests, io, sys, json, webbrowser, warnings, subprocess, re, base64
from datetime import datetime, timedelta
from report import render_report            # 旧版页面（保留）
from report_v2 import render_report as render_report_v2, cond_tags
from report_v3 import render_report as render_report_v3
from pandas.tseries.holiday import USFederalHolidayCalendar
BDAY = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())   # 美股交易日（近似，未含耶稣受难日）

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMBO_FILE = os.path.join(SCRIPT_DIR, 'validated_combos.json')
OUTPUT_HTML = os.path.join(SCRIPT_DIR, 'index.html')
HISTORY_FILE = os.path.join(SCRIPT_DIR, 'picks_history.json')

# ── 指标计算 ──────────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100/(1 + gain/loss)

def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif, dea, dif - dea

def calc_mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=close.index)
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=close.index)
    return 100 - 100 / (1 + pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0,1e-10))

def calc_williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, 1e-10)

def calc_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = calc_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, 1e-10)
    k = stoch_rsi.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

def calc_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(period).mean()
    tp_mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - tp_ma) / (0.015 * tp_mad).replace(0, 1e-10)

def calc_cmf(high, low, close, volume, period=20):
    clv = ((close - low) - (high - close)) / (high - low).replace(0, 1e-10)
    return (clv * volume).rolling(period).sum() / volume.rolling(period).sum()

def calc_obv(close, volume):
    return (np.sign(close.diff()) * volume).cumsum()

def calc_keltner_lower(close, high, low, ema_period=20, atr_period=10, mult=2):
    mid = close.ewm(span=ema_period, adjust=False).mean()
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    return mid - mult * atr

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_adx(high, low, close, period=14):
    up   = high.diff()
    down = -low.diff()
    plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0),    index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=close.index)
    atr = calc_atr(high, low, close, period)
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean()  / atr.replace(0, 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    return dx.ewm(span=period, adjust=False).mean()

def is_hammer(open_, high, low, close):
    body = abs(close - open_)
    lower_shadow = np.minimum(open_, close) - low
    upper_shadow = high - np.maximum(open_, close)
    return (lower_shadow >= 2 * body) & (upper_shadow <= body * 0.5) & ((high-low) > 0)

# ── 条件名称映射（用于显示） ──
COND_LABEL = {
    'cci_lt100': 'CCI<-100', 'stoch_cross': 'StochRSI交叉',
    'near_52w': '近52周低', 'rsi_lt25': 'RSI<25', 'rsi_lt20': 'RSI<20',
    'cmf_pos': 'CMF>0', 'below_kelt': 'Keltner下轨',
    'wr_lt80': 'WR<-80', 'mfi_lt20': 'MFI<20', 'mfi_lt30': 'MFI<30',
    'cmf_rising': 'CMF上升', 'obv_rising': 'OBV上升',
    'stoch_rising': 'StochRSI上升', 'hammer': '锤子线',
    'above_ma200': 'MA200上方', 'below_bb': 'BB下轨',
    'vol_gt12': '放量1.2x', 'dist_ma50_lt_m10': '距MA50<-10%',
    'dist_ma200_lt_m15': '距MA200<-15%', 'ret5d_lt_m3': '5日跌>3%',
    'ret10d_lt_m5': '10日跌>5%', 'cci_rising': 'CCI上升',
    'mfi_rising': 'MFI上升', 'wr_cross80': 'WR穿越-80',
    'vol_lt08': '缩量0.8x',
    # 新增条件标签
    'bullish_engulf': '多头吞没', 'rsi_div': 'RSI底背离',
    'gap_down_close': '跳空收正', 'ret3d_lt_m5': '3日跌>5%',
    'doji_bottom': '十字星底', 'adx_lt25': 'ADX<25',
    'bb_squeeze': 'BB收窄', 'vol_spike2x': '恐慌放量2x',
    'atr_low': 'ATR低位', 'near_ma50': '近MA50',
    'wr_rising3': 'WR升3天', 'consec_down3': '连跌3天',
}

# ── 信号检测 ──────────────────────────────────────────────

def get_sp500_info():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return (dict(zip(df['Symbol'], df['GICS Sector'])),
            dict(zip(df['Symbol'], df['Security'])))

def compute_conditions(close, high, low, volume, open_):
    """计算所有37个条件（原25 + 新增12），返回dict"""
    if len(close) < 260:
        return None

    rsi = calc_rsi(close)
    _, _, hist = calc_macd(close)

    r14 = rsi.iloc[-1]
    r14_1 = rsi.iloc[-2]
    h = hist.iloc[-1]; h1 = hist.iloc[-2]; h2 = hist.iloc[-3]

    # 基础条件：MACD收窄 + RSI<30反弹
    if not (h < 0 and h > h1 > h2 and r14_1 < 30 and r14 > r14_1):
        return None

    c = close.iloc[-1]
    mfi = calc_mfi(high, low, close, volume)
    wr = calc_williams_r(high, low, close)
    stoch_k, stoch_d = calc_stoch_rsi(close)
    cci = calc_cci(high, low, close)
    cmf = calc_cmf(high, low, close, volume)
    obv = calc_obv(close, volume)
    obv_ma5 = obv.rolling(5).mean()
    kelt_lower = calc_keltner_lower(close, high, low)
    hammer = is_hammer(open_, high, low, close)
    vol_ma20 = volume.rolling(20).mean()
    ma200 = close.rolling(200).mean()
    ma50 = close.rolling(50).mean()
    # 新增指标
    atr = calc_atr(high, low, close, 14)
    atr_ma20 = atr.rolling(20).mean()
    adx = calc_adx(high, low, close, 14)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_width = (4 * bb_std) / bb_mid.replace(0, 1e-10)
    bb_width_ma = bb_width.rolling(60).mean()
    bb_lower = bb_mid - 2 * bb_std

    # RSI底背离（需在字典外单独计算，因为要查找过去最低点的位置）
    if len(close) >= 21:
        past_slice = close.iloc[-21:-1]
        past_min_idx = past_slice.idxmin()
        past_min_pos = close.index.get_loc(past_min_idx)
        _rsi_div_val = bool(c <= past_slice.min() * 1.02 and r14 > float(rsi.iloc[past_min_pos]) + 3)
    else:
        _rsi_div_val = False

    conds = {
        'cci_lt100': bool(cci.iloc[-1] < -100),
        'stoch_cross': bool(stoch_k.iloc[-1] > stoch_d.iloc[-1] and stoch_k.iloc[-2] <= stoch_d.iloc[-2]),
        'near_52w': bool((c / close.iloc[-252:].min() - 1) < 0.05),
        'rsi_lt25': bool(r14_1 < 25),
        'rsi_lt20': bool(r14_1 < 20),
        'cmf_pos': bool(cmf.iloc[-1] > 0),
        'below_kelt': bool(c < kelt_lower.iloc[-1]),
        'wr_lt80': bool(wr.iloc[-1] < -80),
        'mfi_lt20': bool(mfi.iloc[-1] < 20),
        'mfi_lt30': bool(mfi.iloc[-1] < 30),
        'cmf_rising': bool(cmf.iloc[-1] > cmf.iloc[-2]),
        'obv_rising': bool(obv.iloc[-1] > obv_ma5.iloc[-1]),
        'stoch_rising': bool(stoch_k.iloc[-1] > stoch_k.iloc[-2]),
        'hammer': bool(hammer.iloc[-1] or hammer.iloc[-2]),
        'above_ma200': bool(c > ma200.iloc[-1]),
        'vol_gt12': bool(volume.iloc[-1] > vol_ma20.iloc[-1] * 1.2) if vol_ma20.iloc[-1] > 0 else False,
        'dist_ma50_lt_m10': bool((c / ma50.iloc[-1] - 1) * 100 < -10),
        'dist_ma200_lt_m15': bool((c / ma200.iloc[-1] - 1) * 100 < -15),
        'ret5d_lt_m3': bool((c / close.iloc[-6] - 1) * 100 < -3) if len(close) > 6 else False,
        'ret10d_lt_m5': bool((c / close.iloc[-11] - 1) * 100 < -5) if len(close) > 11 else False,
        'cci_rising': bool(cci.iloc[-1] > cci.iloc[-2]),
        'mfi_rising': bool(mfi.iloc[-1] > mfi.iloc[-2]),
        'wr_cross80': bool(wr.iloc[-1] > -80 and wr.iloc[-2] < -80),
        'vol_lt08': bool(volume.iloc[-1] < vol_ma20.iloc[-1] * 0.8) if vol_ma20.iloc[-1] > 0 else False,
        'below_bb': bool(c < bb_lower.iloc[-1]),
        # ── 新增12个条件 ──
        'adx_lt25': bool(not pd.isna(adx.iloc[-1]) and float(adx.iloc[-1]) < 25),
        'bullish_engulf': bool(
            c > open_.iloc[-1] and
            max(open_.iloc[-1], c) > max(open_.iloc[-2], close.iloc[-2]) and
            min(open_.iloc[-1], c) < min(open_.iloc[-2], close.iloc[-2]) and
            close.iloc[-2] < open_.iloc[-2]
        ),
        'rsi_div': _rsi_div_val,
        'bb_squeeze': bool(
            not pd.isna(bb_width.iloc[-1]) and not pd.isna(bb_width_ma.iloc[-1]) and
            float(bb_width.iloc[-1]) < float(bb_width_ma.iloc[-1]) * 0.8
        ),
        'consec_down3': bool(close.iloc[-1] < close.iloc[-2] < close.iloc[-3] < close.iloc[-4]),
        'vol_spike2x': bool(volume.iloc[-1] > vol_ma20.iloc[-1] * 2.0) if vol_ma20.iloc[-1] > 0 else False,
        'atr_low': bool(
            not pd.isna(atr.iloc[-1]) and not pd.isna(atr_ma20.iloc[-1]) and
            float(atr.iloc[-1]) < float(atr_ma20.iloc[-1]) * 0.8
        ),
        'ret3d_lt_m5': bool((c / close.iloc[-4] - 1) * 100 < -5) if len(close) > 4 else False,
        'gap_down_close': bool(open_.iloc[-1] < close.iloc[-2] * 0.99 and c > open_.iloc[-1]),
        'near_ma50': bool(-5.0 < (c / ma50.iloc[-1] - 1) * 100 < -1.0),
        'doji_bottom': bool(
            (high.iloc[-1] - low.iloc[-1]) > 0 and
            abs(c - open_.iloc[-1]) / (high.iloc[-1] - low.iloc[-1]) < 0.2 and
            low.iloc[-1] < low.iloc[-2]
        ),
        'wr_rising3': bool(wr.iloc[-1] > wr.iloc[-2] > wr.iloc[-3]),
    }

    # 辅助显示数据
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    vol_ma5 = volume.rolling(5).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_ma5 if vol_ma5 > 0 else 1
    low_52w = close.iloc[-252:].min()

    if vol_ratio <= 0.6:
        vol_status = f'极缩 {vol_ratio:.1f}x'
    elif vol_ratio <= 0.9:
        vol_status = f'缩量 {vol_ratio:.1f}x'
    elif vol_ratio <= 1.5:
        vol_status = f'正常 {vol_ratio:.1f}x'
    else:
        vol_status = f'放量 {vol_ratio:.1f}x'

    display = {
        'rsi': round(r14, 1),
        'cci': round(float(cci.iloc[-1]), 1),
        'cmf': round(float(cmf.iloc[-1]), 3),
        'wr': round(float(wr.iloc[-1]), 1),
        'hist': round(h, 4),
        'dist_52w': round((c / low_52w - 1) * 100, 1),
        'deviation': round((c - ma5) / ma5 * 100, 2),
        'volume_status': vol_status,
        'ma20': round(ma20, 2),
        'stop_loss': round(ma20 * 0.97, 2),
        'price': round(c, 2),
        # 回测分析追加的参考信息（只展示，不参与选股）
        'atr_pct': round(float(atr.iloc[-1]) / c * 100, 1),
        'dist_high252': round((c / close.iloc[-252:].max() - 1) * 100, 1),
    }

    return conds, display


def match_grade(conds, combos):
    """匹配最高等级，返回 (grade, best_combo)"""
    grade_order = {'SSS': 0, 'S': 1, 'A': 2, 'B': 3}
    best_grade = None
    best_combo = None
    best_order = 99
    best_wr5 = 0

    for combo_def in combos:
        if all(conds.get(c, False) for c in combo_def['conditions']):
            g = combo_def['grade']
            order = grade_order.get(g, 99)
            wr5 = combo_def['wr5']
            if order < best_order or (order == best_order and wr5 > best_wr5):
                best_grade = g
                best_combo = combo_def
                best_order = order
                best_wr5 = wr5

    return best_grade, best_combo


# ── 星级（2026-11 复查后的新分级，见 analysis/rebuild/NEW_SCHEME.md）──────────
# 逐个验证 334 个组合后发现：起作用的主要是「跳空收正」和「十字星底」两个条件，S/A/B 等级本身没有作用。
# 实测数字：2000–2026 年标普500 基础信号（VIX<30），次日收盘入场持有 5 天。随便买一只的基准是 53.7% / +0.37%。
TIER_ORDER = ['★★★', '★★', '★', '参考']
TIER_STATS = {   # (T+5 上涨比例 %, T+5 平均涨幅 %, 每年约多少次)
    '★★★': (64.7, 2.00, 21),
    '★★':  (59.7, 0.89, 23),
    '★':   (54.5, 0.66, 34),
    '参考': (52.9, 0.27, 343),
}
TIER_RULE = {
    '★★★': '跳空收正（低开 1% 以上、收盘高于开盘）',
    '★★':  '十字星底',
    '★':   '命中 SSS 组合（不含上面两个条件）',
    '参考': '只有基础信号，或只命中 S/A/B 组合',
}


# ── 主推规则（2026-11-19 起）──────────────────────────────
# 基础信号（同上，不变）+ 今天低开（开盘 < 昨收）+ 该股 5 个日历天内第一次出基础信号；不看 VIX。
# 次日收盘买入，持有 10 个交易日。旧星级照常保留，仅作参考。
NEW_RULE_DEDUP_DAYS = 5
NEW_RULE_HOLD = 10


def base_flags(close):
    """每天是否满足基础信号（MACD 柱<0 且连续 3 根抬高、前一日 RSI<30 且今日回升）"""
    rsi = calc_rsi(close)
    _, _, hist = calc_macd(close)
    return ((hist < 0) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))
            & (rsi.shift(1) < 30) & (rsi > rsi.shift(1)))


def new_rule_hit(close, open_):
    """今天是否命中主推规则。去重：基础信号按时间顺序，距上一次「算数」的基础信号超过 5 个日历天才算数"""
    if not (float(open_.iloc[-1]) < float(close.iloc[-2])):
        return False
    b = base_flags(close).iloc[-80:]
    last = None; kept_today = False
    for d, v in b.items():
        if not v:
            continue
        keep = last is None or (d - last).days > NEW_RULE_DEDUP_DAYS
        if keep:
            last = d
        kept_today = keep and d == b.index[-1]
    return kept_today


def main_grade(gap_pct, open_lt_prev_low, vix, n_main_today):
    """主推分档（analysis/private/rank_tiers.py）：个股深低开 1 分 + 市场恐慌 1 分。
    深低开 = 低开 ≥1% 或 开盘 < 昨日最低；恐慌 = VIX ≥ 25 或 当天主推 ≥ 5 只。2 分 A，1 分 B，0 分 C"""
    score = int(gap_pct <= -1 or open_lt_prev_low) + int(vix >= 25 or n_main_today >= 5)
    return 'ABC'[2 - score]


RECORD_START = '2026-01-01'   # 网页记录从这天开始


def replay_main(data, tickers, vix_hist, name_map, since=RECORD_START, spy=None, mf=None):
    """用已下载的日线，把主推规则在最近 lookback 个交易日逐日重放，得到最近的主推记录（回测）。
    每条：信号日、入场日（T1 收盘）、持有 10 天的收益；没满 10 天的给出至今收益和已持有天数。
    只覆盖当前成分股（近几个月影响很小）。"""
    recs = []
    for t in tickers:
        try:
            td = data[t].dropna()
        except Exception:
            continue
        if len(td) < 260:
            continue
        c, o, l, hh, vv = td['Close'], td['Open'], td['Low'], td['High'], td['Volume']
        b = base_flags(c)
        idx = c.index; start = int(idx.searchsorted(pd.Timestamp(since)))
        last = None
        for i in np.where(b.values)[0]:
            d = idx[i]
            keep = last is None or (d - last).days > NEW_RULE_DEDUP_DAYS
            if keep:
                last = d
            if not keep or i < start or not (o.iloc[i] < c.iloc[i - 1]):
                continue
            rec = {'ticker': t, 'name': name_map.get(t, t), 'signal_day': d,
                   'gap': float(o.iloc[i] / c.iloc[i - 1] - 1) * 100, 'open_lt_pl': bool(o.iloc[i] < l.iloc[i - 1])}
            try:   # 信号当天命中的条件（只用到当天为止的数据）
                cr = compute_conditions(c.iloc[:i + 1], hh.iloc[:i + 1], l.iloc[:i + 1], vv.iloc[:i + 1], o.iloc[:i + 1])
                rec['conds'] = [k for k, v in cr[0].items() if v is True] if cr else []
                rec['conds'] += web_flags(td, spy, i)
            except Exception:
                rec['conds'] = []
            e = i + 1
            if e >= len(c):
                rec.update(status='待买入'); recs.append(rec); continue
            rec['entry_day'] = idx[e]; ep = float(c.iloc[e])
            x = e + NEW_RULE_HOLD
            if x < len(c):
                rec.update(status='完成', ret=(float(c.iloc[x]) / ep - 1) * 100, exit_day=idx[x], held=NEW_RULE_HOLD)
            else:
                rec.update(status='持有中', ret=(float(c.iloc[-1]) / ep - 1) * 100, held=len(c) - 1 - e)
            rec['path'] = [round((float(v) / ep - 1) * 100, 2) for v in c.iloc[e:min(x, len(c) - 1) + 1]]   # 网页小走势图
            recs.append(rec)
    if not recs:
        return []
    nday = pd.Series([r['signal_day'] for r in recs]).value_counts()
    vx = vix_hist.reindex(sorted(nday.index), method='ffill')
    for r in recs:
        r['grade'] = main_grade(r['gap'], r['open_lt_pl'], float(vx.get(r['signal_day'], 0) or 0), int(nday[r['signal_day']]))
        if mf is not None:
            r['conds'] = (r.get('conds') or []) + mkt_conds(mf, r['signal_day'], int(nday[r['signal_day']]))
        r['cond_tags'] = cond_tags(r.get('conds') or [])
        r['net'] = r['ret'] - 0.10 if 'ret' in r else None
    recs.sort(key=lambda r: (r['signal_day'], r['ticker']), reverse=True)
    return recs


def watch_list(data, tickers, name_map, entry_day):
    """观察名单：明天可能触发主推的股票。今天 MACD 柱<0 且比昨天高、今天 RSI<30、5 天内没出过算数的基础信号。
    明天要触发还需：开盘低于今天收盘（低开）、收盘高于今天收盘（RSI 回升）、收盘高到让 MACD 柱继续抬高。
    MACD 柱对明天收盘价是线性的，所以能直接解出这个价格。"""
    out = []
    for t in tickers:
        try:
            c = data[t]['Close'].dropna()
        except Exception:
            continue
        if len(c) < 260:
            continue
        rsi = calc_rsi(c); _, _, h = calc_macd(c)
        if not (h.iloc[-1] < 0 and h.iloc[-1] > h.iloc[-2] and rsi.iloc[-1] < 30):
            continue
        b = base_flags(c).iloc[-80:]; last = None
        for d, v in b.items():
            if v and (last is None or (d - last).days > NEW_RULE_DEDUP_DAYS):
                last = d
        if last is not None and (pd.Timestamp(entry_day) - last).days <= NEW_RULE_DEDUP_DAYS:
            continue
        c0 = float(c.iloc[-1]); nxt = pd.Timestamp(entry_day)
        def h_at(p):
            cc = pd.concat([c, pd.Series([p], index=[nxt])])
            return float(calc_macd(cc)[2].iloc[-1])
        a0, a1 = h_at(c0), h_at(c0 * 1.01)
        slope = (a1 - a0) / (c0 * 0.01)
        if slope <= 0:
            continue
        p_macd = c0 + (float(h.iloc[-1]) - a0) / slope          # 明天收盘高于它，MACD 柱才会继续抬高
        p_neg = c0 + (0 - a0) / slope                           # 明天收盘低于它，柱子才还在零轴下方
        need = max(c0, p_macd)
        if need >= p_neg:
            continue
        out.append({'ticker': t, 'name': name_map.get(t, t), 'close': c0, 'rsi': float(rsi.iloc[-1]),
                    'chg': float(c0 / float(c.iloc[-2]) - 1) * 100, 'need': need,
                    'need_pct': (need / c0 - 1) * 100})
    out.sort(key=lambda r: r['need_pct'])
    return out


def market_series(data, spy, vix_hist, n=120):
    """网页走势图：标普500、VIX、站上 50 日线的成分股比例，最近 n 个交易日"""
    closes = data.xs('Close', axis=1, level=1)
    ma50 = closes.rolling(50).mean()
    valid = closes.notna() & ma50.notna()
    breadth = ((closes > ma50) & valid).sum(1) / valid.sum(1).replace(0, np.nan) * 100
    idx = closes.index[-n:]
    sp = spy.copy(); sp.index = pd.to_datetime(sp.index).tz_localize(None) if getattr(sp.index, 'tz', None) else pd.to_datetime(sp.index)
    return {'dates': [d.strftime('%Y-%m-%d') for d in idx],
            'spy': [None if pd.isna(v) else round(float(v), 2) for v in sp.reindex(idx).ffill().values],
            'vix': [None if pd.isna(v) else round(float(v), 2) for v in vix_hist.reindex(idx).ffill().values],
            'breadth': [None if pd.isna(v) else round(float(v), 1) for v in breadth.reindex(idx).values]}


def star_tier(conds, grade):
    """按条件给星级：跳空收正 > 十字星底 > SSS 组合 > 参考"""
    if conds.get('gap_down_close'):
        return '★★★'
    if conds.get('doji_bottom'):
        return '★★'
    if grade == 'SSS':
        return '★'
    return '参考'


# ── 网上常见的条件（analysis/private/web_cond_grades.py 挑出来的，分级见 report_v2.COND_GRADES）──────
def web_flags(td, spy_close, i):
    """个股条件，只用到第 i 根（信号日）为止的数据"""
    c, h, l, o = td['Close'].iloc[:i + 1], td['High'].iloc[:i + 1], td['Low'].iloc[:i + 1], td['Open'].iloc[:i + 1]
    out = []
    def last(x):
        v = x.iloc[-1]
        return bool(v) if pd.notna(v) else False
    if len(c) >= 200 and last(c / c.rolling(200).mean() - 1 < -0.30): out.append('dist200_30')
    if len(c) >= 200 and last(c / h.rolling(252, min_periods=200).max() - 1 < -0.40): out.append('dd252_40')
    if len(c) > 20 and last(c / c.shift(20) - 1 < -0.20): out.append('ret20_20')
    if len(c) > 252 and last(c / c.shift(252) - 1 < -0.30): out.append('yr_dn30')
    if len(c) >= 10 and last(c / c.rolling(10).mean() - 1 < -0.08): out.append('bias10_8')
    pc, po = c.shift(1), o.shift(1)
    if last((pc < po) & (o < pc) & (c > (po + pc) / 2) & (c < po)): out.append('piercing')
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    if len(c) > 15 and last(tr > 1.5 * atr.shift(1)): out.append('range15')
    if spy_close is not None and len(c) > 5:
        sp = spy_close.reindex(c.index).ffill()
        if last((c / c.shift(5) - sp / sp.shift(5)) < -0.08): out.append('under_spy5')
    return out


def market_flags(data, spy, vix_hist):
    """大盘条件，按日期：大盘普跌、标普跌破 200 日线、标普当天跌超 1%、VIX 当天涨超 10%、上涨家数不到 30%"""
    closes = data.xs('Close', axis=1, level=1)
    br50 = (closes > closes.rolling(50).mean()).sum(1) / closes.notna().sum(1)
    adv = (closes.pct_change() > 0).sum(1) / closes.notna().sum(1)
    sp = spy.reindex(closes.index).ffill(); vx = vix_hist.reindex(closes.index).ffill()
    return pd.DataFrame({'breadth35': br50 < 0.35, 'spy_lt200': sp < sp.rolling(200).mean(),
                         'spy_dn1': sp.pct_change() < -0.01, 'vix_up10': vx.pct_change() > 0.10,
                         'adv30': adv < 0.30}, index=closes.index).fillna(False)


def mkt_conds(mf, day, n_main_day):
    out = []
    if day in mf.index:
        out = [k for k, v in mf.loc[day].items() if bool(v)]
    if n_main_day >= 15:
        out.append('nday15')
    return out


# ── 选股记录与实际表现 ────────────────────────────────────
# picks_history.json 是选股记录的唯一来源：每日扫描追加当天结果，backfill.py 负责补算缺失的日子。
# scan_date 的含义：信号出现后的下一个交易日（即可以入场的那天），收益从这天的收盘价起算。

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_history(history):
    history = sorted(history, key=lambda p: (p['scan_date'], p['ticker']))
    with open(HISTORY_FILE, 'w', encoding='utf-8', newline='\n') as f:
        f.write('[\n' + ',\n'.join(json.dumps(p, ensure_ascii=False) for p in history) + '\n]\n')


def git_history_picks():
    """从 git 历史里每天最后一次扫描的 index.html 反查当时推荐的股票（旧版记录方式，仅供迁移用）"""
    r = subprocess.run(['git', '-C', SCRIPT_DIR, 'log', '--format=%h %s'],
                       capture_output=True, text=True, encoding='utf-8', errors='ignore')
    by_date = {}
    for line in r.stdout.splitlines():
        m = re.search(r'^([a-f0-9]+) .*scan: (\d{4}-\d{2}-\d{2})', line)
        if m and m.group(2) not in by_date:
            by_date[m.group(2)] = m.group(1)
    picks = []
    for ds, sha in sorted(by_date.items()):
        src = subprocess.run(['git', '-C', SCRIPT_DIR, 'show', f'{sha}:index.html'],
                             capture_output=True, text=True, encoding='utf-8', errors='ignore').stdout
        seen = set()
        for row_html in re.split(r'<tr[^>]*>', src):
            mt = re.search(r'<td[^>]*><b>([A-Z][A-Z0-9\-]{0,5})</b>\s*\$', row_html)
            if mt and mt.group(1) not in seen:
                seen.add(mt.group(1))
                picks.append({'ticker': mt.group(1), 'scan_date': ds, 'source': 'live'})
    return picks


def pick_performance(history, days=180):
    """给最近 days 天的选股算 T+1/3/5/10 实际收益（r1..r10）和同期标普500收益（b1..b10）。
    None 表示交易日还没到。返回按入场日倒序的列表"""
    cutoff = (datetime.now().date() - timedelta(days=days)).strftime('%Y-%m-%d')
    picks = [dict(p) for p in history if p['scan_date'] >= cutoff]
    if not picks:
        return []
    tickers = sorted(set(p['ticker'] for p in picks)) + ['^GSPC']
    start = (datetime.strptime(min(p['scan_date'] for p in picks), '%Y-%m-%d').date()
             - timedelta(days=7)).strftime('%Y-%m-%d')
    try:
        prices = yf.download(tickers, start=start, auto_adjust=True, progress=False)['Close']
    except Exception:
        return []

    def get_ret(ticker, scan_date, n):
        if ticker not in prices.columns:
            return None
        col = prices[ticker].dropna()
        future = col.index[col.index >= pd.Timestamp(scan_date)]
        if len(future) < n + 1:
            return None
        return round(float((col[future[n]] / col[future[0]] - 1) * 100), 2)

    def now_ret(ticker, scan_date):
        """入场至今的收益和已持有交易日数（还没到入场日收盘则返回 None, 0）"""
        if ticker not in prices.columns:
            return None, 0
        col = prices[ticker].dropna()
        future = col.index[col.index >= pd.Timestamp(scan_date)]
        if len(future) < 1:
            return None, 0
        return round(float((col[future[-1]] / col[future[0]] - 1) * 100), 2), len(future) - 1

    for p in picks:
        for n in (1, 3, 5, 10):
            p[f'r{n}'] = get_ret(p['ticker'], p['scan_date'], n)
            p[f'b{n}'] = get_ret('^GSPC', p['scan_date'], n)
        p['rnow'], p['held'] = now_ret(p['ticker'], p['scan_date'])
        p['bnow'], _ = now_ret('^GSPC', p['scan_date'])
    picks.sort(key=lambda p: (p['scan_date'], p.get('wr5', 0)), reverse=True)
    return picks


def main():
    # 加载组合定义
    with open(COMBO_FILE, 'r', encoding='utf-8') as f:
        combos = json.load(f)
    # 只保留 SSS/S/A/B
    combos = [c for c in combos if c['grade'] in ('SSS','S','A','B')]
    grade_counts = {}
    for c in combos:
        grade_counts[c['grade']] = grade_counts.get(c['grade'], 0) + 1
    print(f"加载 {len(combos)} 种验证组合")
    for g in ['SSS','S','A','B']:
        print(f"  {g}: {grade_counts.get(g,0)}种")
    bl = {'ret1': 53.7, 'ret3': 55.5, 'ret5': 55.7, 'ret10': 56.6}

    print("\n获取标普500成分股...")
    sector_map, name_map = get_sp500_info()
    tickers = list(sector_map.keys())
    print(f"共 {len(tickers)} 只，下载数据...")

    data = yf.download(tickers, period='2y', group_by='ticker',
                       auto_adjust=True, threads=True, progress=True)
    # 美股还没收盘时，yfinance 会给出当天的半截日线，信号会算错；只用完整日线
    import zoneinfo
    ny = datetime.now(zoneinfo.ZoneInfo('America/New_York'))
    if len(data) and data.index[-1].date() == ny.date() and (ny.hour, ny.minute) < (16, 30):
        data = data.iloc[:-1]
        print(f"美股尚未收盘（纽约 {ny:%H:%M}），只用到 {data.index[-1].date()} 的完整日线")
    spy = yf.download('^GSPC', period='2y', auto_adjust=True, progress=False)['Close'].squeeze()
    vix_df = yf.download('^VIX', period='15mo', auto_adjust=True, progress=False)['Close'].squeeze()
    vix_hist = vix_df.copy(); vix_hist.index = pd.to_datetime(vix_hist.index)
    if vix_hist.index.tz is not None:
        vix_hist.index = vix_hist.index.tz_localize(None)
    vix_price = float(vix_df.iloc[-1]) if len(vix_df) > 0 else 0

    spy_price = float(spy.iloc[-1])
    spy_ma200 = float(spy.rolling(200).mean().iloc[-1])
    spy_pct = (spy_price - spy_ma200) / spy_ma200 * 100
    is_bull = spy_price > spy_ma200
    env_label = "牛市" if is_bull else "熊市"
    env_color = "#27ae60" if is_bull else "#e74c3c"
    vix_high = vix_price >= 20
    vix_extreme = vix_price >= 30
    vix_color = "#e74c3c" if vix_extreme else "#e67e22" if vix_high else "#27ae60"

    if vix_extreme:
        print(f"\n⚠️ VIX={vix_price:.1f} ≥ 30，市场极度恐慌，暂停出信号")

    print("扫描信号中...")
    results = []
    n_universe = n_base = 0

    for ticker in tickers:
        try:
            td = data[ticker].dropna()
            close = td['Close']; volume = td['Volume']
            high = td['High']; low = td['Low']; open_ = td['Open']
            if len(close) < 260:
                continue
            n_universe += 1

            result = compute_conditions(close, high, low, volume, open_)
            if result is None:
                continue
            n_base += 1

            conds, display = result
            new_rule = new_rule_hit(close, open_)

            # VIX >= 30 暂停旧星级信号；主推规则不看 VIX
            if vix_extreme and not new_rule:
                continue

            grade, best_combo = match_grade(conds, combos)
            tier = star_tier(conds, grade)   # 新星级；没命中任何组合的基础信号也保留，标为「参考」

            matched_signals = ['MACD收窄', 'RSI<30反弹']
            shown = list(best_combo['conditions']) if best_combo else []
            for key in ('gap_down_close', 'doji_bottom'):   # 决定星级的条件一定显示出来
                if conds.get(key) and key not in shown:
                    shown.append(key)
            for cond_name in shown:
                matched_signals.append(COND_LABEL.get(cond_name, cond_name))

            chg = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)

            r = dict(display)
            bc = best_combo or {}
            r.update({
                'grade': grade or '',
                'tier': tier,
                'new_rule': new_rule,
                'gap': round(float(open_.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
                'open_lt_pl': bool(open_.iloc[-1] < low.iloc[-2]),
                'tier_wr5': TIER_STATS[tier][0],
                'tier_ret5': TIER_STATS[tier][1],
                'matched_signals': matched_signals,
                'conds': [k for k, v in conds.items() if v is True] + web_flags(td, spy, len(td) - 1),
                'wr1': bc.get('wr1', 0),
                'wr3': bc.get('wr3', 0),
                'wr5': bc.get('wr5', 0),
                'wr10': bc.get('wr10', 0),
                'win_rate_5': bc.get('wr5', 0),
                'matched_combo': ' + '.join(bc['conditions']) if bc else '（无组合）',
                'sample': bc.get('n', 0),
                'ticker': ticker,
                'name': name_map.get(ticker, ticker),
                'sector': sector_map.get(ticker, '—'),
                'chg': round(chg, 2),
            })
            results.append(r)
        except Exception:
            continue

    # 财报预警：未来10天内有财报则标注
    today = datetime.now().date()
    deadline = today + timedelta(days=10)
    print("检查财报日期...")
    for r in results:
        r['earnings_warn'] = ''
        try:
            cal = yf.Ticker(r['ticker']).calendar
            dates = []
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                if 'Earnings Date' in cal.columns:
                    dates = [d for d in cal['Earnings Date'] if pd.notna(d)]
            elif isinstance(cal, dict):
                raw = cal.get('Earnings Date', [])
                dates = raw if isinstance(raw, list) else [raw]
            upcoming = [d for d in dates if hasattr(d, 'date') and today <= d.date() <= deadline]
            if upcoming:
                nearest = min(upcoming, key=lambda d: d.date())
                r['earnings_warn'] = nearest.strftime('%m-%d')
        except Exception:
            pass

    # 过滤近10个交易日（约14日历天）已推荐的股票
    # scan_date = 入场日 = 信号日（最后一根完整日线）的下一个交易日；收盘后跑和第二天早上跑结果一致
    signal_day = pd.Timestamp(data.index[-1]).normalize()
    entry_day = signal_day + BDAY
    exit_day = entry_day + NEW_RULE_HOLD * BDAY
    today_str = entry_day.strftime('%Y-%m-%d')
    cutoff_str = (datetime.now().date() - timedelta(days=14)).strftime('%Y-%m-%d')
    history = load_history()
    recent_tickers = {p['ticker'] for p in history if cutoff_str <= p['scan_date'] < today_str
                      and p.get('tier') != '参考'}   # 只按带星的推荐去重，「参考」不占名额
    before = len(results)
    results = [r for r in results if r['new_rule'] or r['ticker'] not in recent_tickers]   # 主推有自己的 5 天去重
    n_deduped = before - len(results)
    if n_deduped:
        print(f"  去重过滤: 移除 {before - len(results)} 只（近10交易日已推荐）")

    mf = market_flags(data, spy, vix_hist)
    # 主推分档 A/B/C；主推排最前（按档、再按低开幅度），其余按旧星级
    n_main = sum(r['new_rule'] for r in results)
    for r in results:
        r['main_grade'] = main_grade(r['gap'], r['open_lt_pl'], vix_price, n_main) if r['new_rule'] else ''
        if r['new_rule']:
            r['conds'] = (r.get('conds') or []) + mkt_conds(mf, signal_day, n_main)
        r['cond_tags'] = cond_tags(r.get('conds') or [])
        r['n_sss'] = sum(g == 'SSS' for g, _ in r['cond_tags']); r['n_a'] = sum(g == 'A' for g, _ in r['cond_tags'])
    # 同一级别里，命中 SSS 条件多的排前，再看 A 条件，再看低开幅度
    results.sort(key=lambda x: (not x['new_rule'], x['main_grade'], -x['n_sss'], -x['n_a'], x['gap'], TIER_ORDER.index(x['tier']), -x.get('win_rate_5', 0)))
    for r in results:
        r['n_base_day'] = n_base   # 今天全市场通过基础信号的股票数

    print("生成近期选股表现...")
    # 把今天的结果记进选股记录（同一天重跑则覆盖）
    history = [p for p in history if p['scan_date'] != today_str]
    history += [{'ticker': r['ticker'], 'scan_date': today_str, 'grade': r['grade'], 'tier': r['tier'],
                 'new_rule': r['new_rule'], 'main_grade': r['main_grade'], 'wr5': r['wr5'], 'source': 'live'} for r in results]
    save_history(history)
    recent = pick_performance(history)
    print("重放最近的主推（回测记录）...")
    replay = replay_main(data, tickers, vix_hist, name_map, spy=spy, mf=mf)
    print("观察名单、市场走势...")
    watch = watch_list(data, tickers, name_map, entry_day)
    mkt = market_series(data, spy, vix_hist, n=int((data.index >= pd.Timestamp(RECORD_START)).sum()))

    date_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    render = (render_report if '--old-page' in sys.argv            # --old-page：最早的旧版页面
              else render_report_v2 if '--v2-page' in sys.argv     # --v2-page：11-20 的简版页面
              else render_report_v3)
    ctx = {
        'date_str': date_str,
        'signal_day': signal_day.strftime('%Y-%m-%d'),
        'entry_day': entry_day.strftime('%Y-%m-%d'),
        'exit_day': exit_day.strftime('%Y-%m-%d'),
        'hold': NEW_RULE_HOLD,
        'replay': replay,
        'watch': watch,
        'mkt': mkt,
        'spy_chg': float(spy.iloc[-1] / spy.iloc[-2] - 1) * 100,
        'n_combos': len(combos),
        'grade_counts': grade_counts,
        'bl': bl,
        'is_bull': is_bull,
        'spy_price': spy_price,
        'spy_pct': spy_pct,
        'vix_price': vix_price,
        'vix_high': vix_high,
        'vix_extreme': vix_extreme,
        'results': results,
        'funnel': {'universe': n_universe, 'base': n_base, 'deduped': n_deduped},
        'recent': recent,
    }
    if '--dump-ctx' in sys.argv:   # 调网页用：把数据存下来，改页面时不用重新下载
        import pickle
        with open(sys.argv[sys.argv.index('--dump-ctx') + 1], 'wb') as f:
            pickle.dump(ctx, f)
    html = render(ctx)

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ 报告已生成：{OUTPUT_HTML}")
    for r in results:
        print(f"   {'主推' + {'A': 'SSS', 'B': 'A', 'C': 'B'}.get(r['main_grade'], '') if r['new_rule'] else '     '} {r['tier']:4} {r['ticker']:6} RSI={r['rsi']:5.1f} 原等级={r['grade'] or '—'} 匹配: {r['matched_combo']}")
    print(f"\n   总计: {len(results)} 只")

    if '--no-open' not in sys.argv:
        webbrowser.open(f'file:///{OUTPUT_HTML}')

    # --no-push：只在本地生成报告，不提交、不发布到 GitHub Pages
    if '--no-push' in sys.argv:
        print("已跳过推送（--no-push）")
        return

    repo_dir = SCRIPT_DIR
    try:
        subprocess.run(['git', '-C', repo_dir, 'add', 'index.html', 'picks_history.json'], check=True)
        subprocess.run(['git', '-C', repo_dir, 'commit', '-m', f'v5 signal scan: {date_str}'], check=True)
        subprocess.run(['git', '-C', repo_dir, 'push', 'origin', 'main'], check=True)
        print("✅ 已推送到 GitHub Pages")
    except Exception as e:
        print(f"⚠️ 推送失败：{e}")

if __name__ == '__main__':
    main()
