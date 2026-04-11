"""
Daily Stock Scanner - S&P 500 v5
基于6年(2020-2025)回测，1,626种指标组合穷举验证
只保留T+1/3/5/10全周期跑赢baseline的206种组合
从 winning_combos.json 动态加载信号定义
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests, io, sys, json, webbrowser, warnings, os, subprocess
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMBO_FILE = os.path.join(SCRIPT_DIR, 'validated_combos.json')
OUTPUT_HTML = os.path.join(SCRIPT_DIR, 'index.html')

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


# ── HTML ──────────────────────────────────────────────────

GRADE_COLOR = {
    'SSS': '#ff6f00',
    'S':   '#6c3483',
    'A':   '#c0392b',
    'B':   '#2980b9',
}

GRADE_DESC = {
    'SSS': '高收益区 · 样本少但均收>2% · T+1/3/5/10全超baseline',
    'S':   '最强 · 5日胜率≥80%',
    'A':   '强 · 5日胜率70-80%',
    'B':   '有效 · 5日胜率65-70%',
}

def make_table(rows):
    if not rows:
        return '<p style="color:#aaa;padding:12px">暂无</p>'
    cols = ['股票/板块', '级别', '匹配信号', 'T+1', 'T+3', 'T+5', 'T+10', 'RSI', 'CCI', 'CMF', 'WR', '距52W低', '成交量', '日涨跌', '止损']
    th = ''.join(f'<th>{c}</th>' for c in cols)
    body = ''
    for r in rows:
        gc = GRADE_COLOR.get(r['grade'], '#888')
        chg_c = '#27ae60' if r['chg'] >= 0 else '#e74c3c'

        sig_tags = ''
        for s in r['matched_signals']:
            sig_tags += f'<span style="background:#3498db;color:white;padding:1px 6px;border-radius:8px;font-size:10px;margin:1px 2px;display:inline-block">{s}</span>'

        def wr_cell(val):
            return f'<td style="font-size:12px;font-weight:bold;color:#27ae60">{val:.1f}%</td>'

        earn_tag = ''
        if r.get('earnings_warn'):
            earn_tag = f'<br><span style="background:#e67e22;color:white;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:bold">⚠️ 财报 {r["earnings_warn"]}</span>'

        body += f'''<tr>
          <td><b>{r["ticker"]}</b> ${r["price"]}<br><small style="color:#888">{r["sector"][:22]}</small>{earn_tag}</td>
          <td><span style="background:{gc};color:white;padding:3px 10px;border-radius:12px;font-size:13px;font-weight:bold">{r["grade"]}</span></td>
          <td style="font-size:11px">{sig_tags}</td>
          {wr_cell(r.get("wr1",0))}
          {wr_cell(r.get("wr3",0))}
          {wr_cell(r.get("wr5",0))}
          {wr_cell(r.get("wr10",0))}
          <td style="font-size:13px;font-weight:bold;color:#e74c3c">{r["rsi"]}</td>
          <td style="font-size:12px;color:#8e44ad">{r["cci"]}</td>
          <td style="font-size:12px;color:{'#27ae60' if r['cmf']>0 else '#e74c3c'}">{r["cmf"]:+.3f}</td>
          <td style="font-size:12px">{r["wr"]}</td>
          <td style="font-size:12px;color:#8e44ad">+{r["dist_52w"]:.1f}%</td>
          <td style="font-size:12px">{r["volume_status"]}</td>
          <td style="color:{chg_c};font-weight:bold">{r["chg"]:+.2f}%</td>
          <td style="font-size:12px">${r["stop_loss"]}</td>
        </tr>'''
    return f'<table class="data-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


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

    data = yf.download(tickers, period='15mo', group_by='ticker',
                       auto_adjust=True, threads=True, progress=True)
    spy = yf.download('^GSPC', period='1y', auto_adjust=True, progress=False)['Close'].squeeze()
    vix_df = yf.download('^VIX', period='5d', auto_adjust=True, progress=False)['Close'].squeeze()
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

    for ticker in tickers:
        try:
            td = data[ticker].dropna()
            close = td['Close']; volume = td['Volume']
            high = td['High']; low = td['Low']; open_ = td['Open']
            if len(close) < 260:
                continue

            result = compute_conditions(close, high, low, volume, open_)
            if result is None:
                continue

            conds, display = result

            # VIX >= 30 暂停出信号
            if vix_extreme:
                continue

            grade, best_combo = match_grade(conds, combos)

            if grade is None:
                continue

            matched_signals = ['MACD收窄', 'RSI<30反弹']
            for cond_name in best_combo['conditions']:
                matched_signals.append(COND_LABEL.get(cond_name, cond_name))

            chg = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)

            r = dict(display)
            r.update({
                'grade': grade,
                'matched_signals': matched_signals,
                'wr1': best_combo.get('wr1', 0),
                'wr3': best_combo.get('wr3', 0),
                'wr5': best_combo.get('wr5', 0),
                'wr10': best_combo.get('wr10', 0),
                'win_rate_5': best_combo['wr5'],
                'matched_combo': ' + '.join(best_combo['conditions']),
                'sample': best_combo['n'],
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

    # 按级别排序
    grade_order = {'SSS': 0, 'S': 1, 'A': 2, 'B': 3}
    results.sort(key=lambda x: (grade_order.get(x['grade'], 99), -x.get('win_rate_5', 0)))

    by_grade = {}
    for g in ['SSS','S','A','B']:
        by_grade[g] = [r for r in results if r['grade'] == g]

    date_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    grade_legend = ''
    for g, desc in GRADE_DESC.items():
        gc = GRADE_COLOR[g]
        cnt = len(by_grade.get(g, []))
        grade_legend += f'<span style="background:{gc};color:white;padding:2px 8px;border-radius:10px;font-size:11px;margin:2px 4px;display:inline-block"><b>{g}</b> {desc}</span> '

    badge_html = ''
    for g in ['SSS','S','A','B']:
        gc = GRADE_COLOR[g]
        cnt = len(by_grade.get(g, []))
        badge_html += f'<span class="badge" style="background:{gc}">{g}级 {cnt}</span>\n  '
    badge_html += f'<span class="badge" style="background:#2c3e50">总计 {len(results)}</span>'

    # ── 板块统计 ──
    from collections import defaultdict
    sector_stats = defaultdict(lambda: {'count': 0, 'wr5_sum': 0, 'grades': []})
    for r in results:
        s = r['sector']
        sector_stats[s]['count'] += 1
        sector_stats[s]['wr5_sum'] += r.get('wr5', 0)
        sector_stats[s]['grades'].append(r['grade'])

    sector_rows = []
    for sec, st in sorted(sector_stats.items(), key=lambda x: -x[1]['count']):
        avg_wr5 = st['wr5_sum'] / st['count'] if st['count'] else 0
        grade_cnt = {g: st['grades'].count(g) for g in ['SSS','S','A','B'] if st['grades'].count(g)}
        grade_str = ' '.join(
            f'<span style="background:{GRADE_COLOR[g]};color:white;padding:1px 6px;'
            f'border-radius:8px;font-size:10px">{g}×{n}</span>'
            for g, n in grade_cnt.items()
        )
        sector_rows.append((sec, st['count'], avg_wr5, grade_str))

    sector_html = ''
    if sector_rows:
        bar_max = sector_rows[0][1]
        rows_html = ''
        for sec, cnt, avg_wr5, grade_str in sector_rows:
            bar_w = int(cnt / bar_max * 180)
            rows_html += f'''<tr>
              <td style="font-size:12px;white-space:nowrap">{sec[:32]}</td>
              <td style="text-align:center;font-weight:bold">{cnt}</td>
              <td><div style="background:#3498db;height:12px;width:{bar_w}px;border-radius:3px;display:inline-block"></div></td>
              <td style="text-align:center;color:#27ae60;font-weight:bold">{avg_wr5:.1f}%</td>
              <td>{grade_str}</td>
            </tr>'''
        sector_html = f'''
<div class="card">
  <h2>📊 板块分布</h2>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr style="background:#2c3e50;color:white">
      <th style="padding:6px 10px;text-align:left">板块</th>
      <th style="padding:6px 10px">信号数</th>
      <th style="padding:6px 10px;text-align:left">占比</th>
      <th style="padding:6px 10px">均T+5胜率</th>
      <th style="padding:6px 10px;text-align:left">等级构成</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
'''

    section_html = ''
    section_info = {
        'SSS': ('🔥', 'SSS高收益区', '均收>2%，样本较少，高弹性'),
        'S':   ('🟣', 'S级 — 最强信号', '5日胜率≥80%'),
        'A':   ('🔴', 'A级 — 强信号', '5日胜率70-80%'),
        'B':   ('🔵', 'B级 — 有效信号', '5日胜率65-70%'),
    }
    for g in ['SSS','S','A','B']:
        icon, title, desc = section_info[g]
        rows = by_grade.get(g, [])
        section_html += f'''
<div class="card">
  <h2>{icon} {title}（{desc}）</h2>
  {make_table(rows)}
</div>
'''

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>信号扫描 {date_str}</title>
<style>
  * {{ box-sizing:border-box }}
  body {{ font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#f0f2f5;margin:0;padding:16px }}
  .container {{ max-width:1600px;margin:auto }}
  .card {{ background:white;border-radius:10px;padding:20px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.07) }}
  h1 {{ color:#2c3e50;margin:0 0 4px;font-size:22px }}
  h2 {{ color:#2c3e50;margin:0 0 14px;font-size:15px;border-left:4px solid #3498db;padding-left:10px }}
  .stat {{ display:inline-block;margin-right:20px;font-size:13px;color:#666 }}
  .stat b {{ color:#2c3e50;font-size:15px }}
  .badge {{ display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:bold;color:white;margin-right:6px }}
  table.data-table {{ width:100%;border-collapse:collapse;font-size:13px }}
  table.data-table th {{ background:#2c3e50;color:white;padding:8px 10px;text-align:left;white-space:nowrap }}
  table.data-table td {{ padding:8px 10px;border-bottom:1px solid #f0f0f0;vertical-align:top }}
  table.data-table tr:hover td {{ background:#fafafa }}
  .legend {{ font-size:11px;line-height:2;padding:10px;background:#f8f9fa;border-radius:6px }}
  .note {{ color:#888;font-size:11px;margin-top:6px }}
</style>
</head>
<body>
<div class="container">

<div class="card">
  <h1>SP500 Signal Scanner v5</h1>
  <p style="color:#aaa;margin:2px 0 10px;font-size:13px">{date_str} &nbsp;·&nbsp; 87种回测验证组合(去除C级) · T+1/3/5/10全周期超baseline</p>
  <div class="stat">SPY <b style="color:{env_color}">{env_label}</b></div>
  <div class="stat">SPY <b>${spy_price:.2f}</b></div>
  <div class="stat">vs MA200 <b style="color:{env_color}">{spy_pct:+.1f}%</b></div>
  <div class="stat">VIX <b style="color:{vix_color}">{vix_price:.1f}</b></div>
  <br>
  {'<div style="margin:8px 0;padding:10px 14px;background:#ffebee;border-left:4px solid #c62828;border-radius:4px;font-size:14px"><b style="color:#c62828">VIX ≥ 30 — 市场极度恐慌，信号暂停</b><br><span style="font-size:12px;color:#666">回测显示极端恐慌期超卖信号失败率极高，等VIX回落至30以下再恢复扫描</span></div>' if vix_extreme else '<div style="margin:8px 0;padding:8px 14px;background:#fff3e0;border-left:4px solid #e67e22;border-radius:4px;font-size:13px"><b style="color:#e67e22">VIX > 20 — 恐慌信号增强</b>　回测显示VIX>20时超卖反弹信号胜率提升约5-10个百分点，当前信号可信度更高</div>' if vix_high else ''}
  <br>
  {badge_html}
</div>

<div class="card">
  <div class="legend">
    <b>信号分级（6年2020-2025标普500全成分股回测，1,626种组合穷举，全周期T+1/3/5/10验证）</b><br>
    {grade_legend}
    <p class="note">基础信号: MACD收窄(柱状图零轴下连续3根递增) + RSI&lt;30反弹 · 每个等级在此基础上叠加不同附加条件</p>
  </div>
</div>

{sector_html}

{section_html}

<div class="card" style="border-top:3px solid #2c3e50;margin-top:30px">
  <h2 style="border-left:4px solid #2c3e50;padding-left:10px">方法论</h2>
  <div style="font-size:13px;line-height:1.9;color:#444">

  <h3 style="font-size:14px;margin:16px 0 8px;color:#2c3e50">策略</h3>
  <p>在MACD金叉之前入场。当MACD柱状图在零轴下方连续收窄（空头力量减弱），同时RSI从超卖区反弹，形成基础信号。叠加多项技术指标过滤后，只推荐在T+1、T+3、T+5、T+10全周期均跑赢随机baseline的组合。</p>

  <h3 style="font-size:14px;margin:16px 0 8px;color:#2c3e50">分级</h3>
  <table style="font-size:12px;border-collapse:collapse;margin:6px 0 12px;width:auto">
    <tr style="background:#2c3e50;color:white">
      <th style="padding:6px 14px">等级</th>
      <th style="padding:6px 14px">5日胜率</th>
      <th style="padding:6px 14px">5日中位收益</th>
    </tr>
    <tr style="background:#fff3e0">
      <td style="padding:5px 14px"><b style="color:#ff6f00">SSS</b> 高收益区</td>
      <td style="padding:5px 14px">~74%</td>
      <td style="padding:5px 14px">+2.0%</td>
    </tr>
    <tr style="background:#f3e5f5">
      <td style="padding:5px 14px"><b style="color:#6c3483">S</b> 最强</td>
      <td style="padding:5px 14px">≥ 80%</td>
      <td style="padding:5px 14px">+1.2%</td>
    </tr>
    <tr>
      <td style="padding:5px 14px"><b style="color:#c0392b">A</b> 强</td>
      <td style="padding:5px 14px">70–80%</td>
      <td style="padding:5px 14px">+1.7%</td>
    </tr>
    <tr style="background:#f8f8f8">
      <td style="padding:5px 14px"><b style="color:#2980b9">B</b> 有效</td>
      <td style="padding:5px 14px">65–70%</td>
      <td style="padding:5px 14px">+1.7%</td>
    </tr>
  </table>

  <p style="margin-top:16px;padding-top:12px;border-top:1px solid #eee;color:#aaa;font-size:11px">
    建议持仓：5个交易日 · 止损参考：MA20 × 0.97<br>
    仅供研究参考，不构成投资建议。历史回测不代表未来表现。
  </p>
  </div>
</div>

</div>
</body>
</html>'''

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ 报告已生成：{OUTPUT_HTML}")
    for g in ['SSS','S','A','B']:
        cnt = len(by_grade.get(g, []))
        if cnt:
            print(f"\n   {g}级 ({cnt}只):")
            for r in by_grade[g]:
                print(f"      {r['ticker']:6} RSI={r['rsi']:5.1f} T+5胜率={r['win_rate_5']:.1f}% 匹配: {r['matched_combo']}")

    print(f"\n   总计: {len(results)} 只")

    webbrowser.open(f'file:///{OUTPUT_HTML}')

    repo_dir = SCRIPT_DIR
    try:
        subprocess.run(['git', '-C', repo_dir, 'add', 'index.html'], check=True)
        subprocess.run(['git', '-C', repo_dir, 'commit', '-m', f'v5 signal scan: {date_str}'], check=True)
        subprocess.run(['git', '-C', repo_dir, 'push', 'origin', 'main'], check=True)
        print("✅ 已推送到 GitHub Pages")
    except Exception as e:
        print(f"⚠️ 推送失败：{e}")

if __name__ == '__main__':
    main()
