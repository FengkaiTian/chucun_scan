"""
Daily Stock Scanner - S&P 500
复刻 ZhuLinsen/daily_stock_analysis 评分体系
本地计算，输出 HTML，无需 LLM / 推送
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import sys
import webbrowser
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_HTML = r"C:\Users\ft7b6\OneDrive\Desktop\STOCK\index.html"

# ── 数据 ──────────────────────────────────────────────────

def get_sp500_info():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return (dict(zip(df['Symbol'], df['GICS Sector'])),
            dict(zip(df['Symbol'], df['Security'])))

# ── 指标 ──────────────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# ── 核心评分（参考 bull_trend 策略）─────────────────────
# 总分 100，分五维度

def score_stock(close, volume):
    result = {
        'score': 0, 'signal': '', 'signal_en': '',
        'trend': '', 'deviation': 0.0, 'rsi': 0.0,
        'macd_status': '', 'volume_status': '',
        'risks': [], 'positives': [],
        'stop_loss': 0.0, 'ideal_buy': 0.0,
        'ma5': 0.0, 'ma10': 0.0, 'ma20': 0.0,
    }

    if len(close) < 30:
        return None

    price   = close.iloc[-1]
    ma5     = close.rolling(5).mean().iloc[-1]
    ma10    = close.rolling(10).mean().iloc[-1]
    ma20    = close.rolling(20).mean().iloc[-1]
    ma60    = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else np.nan
    rsi14   = calc_rsi(close).iloc[-1]
    macd, sig_line, hist = calc_macd(close)
    macd_val  = macd.iloc[-1]
    hist_val  = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    vol_ma5   = volume.rolling(5).mean().iloc[-1]
    vol_now   = volume.iloc[-1]
    deviation = (price - ma5) / ma5 * 100  # 乖离率 vs MA5

    result['ma5']  = round(ma5, 2)
    result['ma10'] = round(ma10, 2)
    result['ma20'] = round(ma20, 2)
    result['rsi']  = round(rsi14, 1)
    result['deviation'] = round(deviation, 2)
    result['stop_loss'] = round(ma20 * 0.98, 2)
    result['ideal_buy'] = round(ma5 * 1.005, 2)

    score = 0

    # ── 1. 趋势强度（30分）────────────────────────────────
    if ma5 > ma10 > ma20:                        # 完美多头排列
        score += 30
        result['trend'] = '强势多头 ▲▲'
        result['positives'].append('MA5>MA10>MA20 多头排列')
        if not np.isnan(ma60) and ma20 > ma60:
            score += 5                           # 加分：MA20也在MA60上
            result['positives'].append('MA20>MA60 中长期趋势向上')
    elif ma5 > ma10:
        score += 18
        result['trend'] = '弱势多头 ▲'
    elif ma5 < ma10 < ma20:
        score += 0
        result['trend'] = '空头排列 ▼▼'
        result['risks'].append('MA5<MA10<MA20 空头排列')
    elif ma5 < ma10:
        score += 8
        result['trend'] = '弱势空头 ▼'
    else:
        score += 12
        result['trend'] = '盘整震荡 ─'

    # ── 2. 乖离率（20分）─────────────────────────────────
    # 偏离MA5的距离，越近越好（适合买入），追高扣分
    abs_dev = abs(deviation)
    if abs_dev <= 2:
        score += 20
        result['positives'].append(f'乖离率 {deviation:+.1f}%（贴近MA5，最佳买点）')
    elif abs_dev <= 5:
        score += 12
    elif abs_dev <= 8:
        score += 5
        if deviation > 0:
            result['risks'].append(f'乖离率 {deviation:+.1f}%（偏高，追高风险）')
    else:
        score += 0
        if deviation > 0:
            result['risks'].append(f'乖离率 {deviation:+.1f}%（严重偏高，禁止追高）')
        else:
            result['risks'].append(f'乖离率 {deviation:+.1f}%（严重偏离，加速下跌中）')

    # ── 3. 成交量（15分）─────────────────────────────────
    vol_ratio = vol_now / vol_ma5 if vol_ma5 > 0 else 1
    if 0.6 <= vol_ratio <= 0.9:
        score += 15
        result['volume_status'] = f'缩量 {vol_ratio:.1f}x（理想回调）'
        result['positives'].append('缩量回调，筹码稳定')
    elif 0.9 < vol_ratio <= 1.5:
        score += 10
        result['volume_status'] = f'正常量 {vol_ratio:.1f}x'
    elif vol_ratio > 1.5:
        score += 7
        result['volume_status'] = f'放量 {vol_ratio:.1f}x'
        if deviation > 3:
            result['risks'].append(f'放量追高（量比{vol_ratio:.1f}x，乖离{deviation:+.1f}%）')
        else:
            result['positives'].append(f'放量突破（量比{vol_ratio:.1f}x）')
    else:
        score += 3
        result['volume_status'] = f'极度缩量 {vol_ratio:.1f}x'
        result['risks'].append('成交量极度萎缩')

    # ── 4. MACD（15分）───────────────────────────────────
    if macd_val > 0 and hist_val > 0:
        score += 15
        result['macd_status'] = '零轴上方 金叉'
        result['positives'].append('MACD 零轴上方，多头动能强')
    elif macd_val < 0 and hist_val > hist_prev:
        score += 10
        result['macd_status'] = '零轴下方 底背离'
        result['positives'].append('MACD 底背离，潜在反转信号')
    elif macd_val > 0 and hist_val < hist_prev:
        score += 8
        result['macd_status'] = '零轴上方 动能减弱'
        result['risks'].append('MACD 顶背离风险')
    elif macd_val < 0 and hist_val < 0:
        score += 2
        result['macd_status'] = '零轴下方 死叉'
        result['risks'].append('MACD 零轴下方，空头主导')
    else:
        score += 5
        result['macd_status'] = '中性'

    # ── 5. RSI（10分）────────────────────────────────────
    if 40 <= rsi14 <= 60:
        score += 10
        result['positives'].append(f'RSI {rsi14:.0f}（中性健康区间）')
    elif 30 <= rsi14 < 40:
        score += 10
        result['positives'].append(f'RSI {rsi14:.0f}（超卖区域，潜在买点）')
    elif rsi14 < 30:
        score += 8
        result['positives'].append(f'RSI {rsi14:.0f}（深度超卖）')
    elif 60 < rsi14 <= 70:
        score += 6
    elif rsi14 > 70:
        score += 2
        result['risks'].append(f'RSI {rsi14:.0f}（超买，短期回调风险）')

    # ── 价格在MA20下方 ────────────────────────────────────
    if price < ma20:
        score -= 10
        result['risks'].append('价格跌破MA20支撑')

    result['score'] = max(0, min(100, score))

    # ── 信号映射 ──────────────────────────────────────────
    s = result['score']
    if s >= 80:
        result['signal'] = '强烈买入 ⭐⭐⭐'
        result['signal_en'] = 'STRONG BUY'
    elif s >= 65:
        result['signal'] = '买入 ⭐⭐'
        result['signal_en'] = 'BUY'
    elif s >= 50:
        result['signal'] = '观望 ⚪'
        result['signal_en'] = 'WATCH'
    elif s >= 35:
        result['signal'] = '谨慎 ⚠️'
        result['signal_en'] = 'CAUTION'
    else:
        result['signal'] = '回避 ❌'
        result['signal_en'] = 'AVOID'

    return result

# ── HTML 生成 ──────────────────────────────────────────────

SIGNAL_COLOR = {
    'STRONG BUY': '#27ae60',
    'BUY':        '#2ecc71',
    'WATCH':      '#f39c12',
    'CAUTION':    '#e67e22',
    'AVOID':      '#e74c3c',
}

def score_bar(score):
    color = '#27ae60' if score >= 65 else '#f39c12' if score >= 50 else '#e74c3c'
    return f'''<div style="background:#eee;border-radius:4px;height:8px;width:80px;display:inline-block;vertical-align:middle">
      <div style="background:{color};width:{score}%;height:100%;border-radius:4px"></div></div>
      <span style="font-size:12px;margin-left:4px">{score}</span>'''

def make_table(rows, cols):
    if not rows:
        return '<p style="color:#aaa;padding:12px">暂无</p>'
    th = ''.join(f'<th>{c}</th>' for c in cols)
    body = ''
    for r in rows:
        sc  = r['score']
        sig = r['signal_en']
        color = SIGNAL_COLOR.get(sig, '#888')
        chg_color = '#27ae60' if r['chg'] >= 0 else '#e74c3c'
        risks_str    = '<br>'.join(f'⚠️ {x}' for x in r['risks'][:2])    or '—'
        pos_str      = '<br>'.join(f'✅ {x}' for x in r['positives'][:2]) or '—'
        body += f'''<tr>
          <td><b>{r["ticker"]}</b><br><small style="color:#888">{r["sector"][:18]}</small></td>
          <td>{score_bar(sc)}</td>
          <td><span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:12px;white-space:nowrap">{r["signal"]}</span></td>
          <td style="font-size:12px">{r["trend"]}</td>
          <td style="font-size:12px">{r["macd_status"]}</td>
          <td style="font-size:12px">RSI {r["rsi"]}<br>乖离 {r["deviation"]:+.1f}%</td>
          <td style="font-size:12px">{r["volume_status"]}</td>
          <td style="color:{chg_color};font-weight:bold">{r["chg"]:+.2f}%</td>
          <td style="font-size:11px;color:#666">{pos_str}</td>
          <td style="font-size:11px;color:#c0392b">{risks_str}</td>
          <td style="font-size:12px">买入 ${r["ideal_buy"]}<br>止损 ${r["stop_loss"]}</td>
        </tr>'''
    return f'<table class="data-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'

COLS = ['股票/板块','评分','信号','趋势','MACD','RSI/乖离','成交量','日涨跌','优势','风险','操作参考']

def main():
    print("获取标普500成分股...")
    sector_map, name_map = get_sp500_info()
    tickers = list(sector_map.keys())
    print(f"共 {len(tickers)} 只，下载近3个月数据...")

    data = yf.download(tickers, period='3mo', group_by='ticker',
                       auto_adjust=True, threads=True, progress=True)
    spy  = yf.download('^GSPC', period='1y', auto_adjust=True, progress=False)['Close'].squeeze()

    # 大盘环境
    spy_price = float(spy.iloc[-1])
    spy_ma200 = float(spy.rolling(200).mean().iloc[-1])
    spy_pct   = (spy_price - spy_ma200) / spy_ma200 * 100
    is_bull   = spy_price > spy_ma200
    env_label = "牛市 📈" if is_bull else "熊市 📉"
    env_color = "#27ae60" if is_bull else "#e74c3c"

    print("评分中...")
    all_rows = []

    for ticker in tickers:
        try:
            close  = data[ticker]['Close'].dropna()
            volume = data[ticker]['Volume'].dropna()
            if len(close) < 30:
                continue
            r = score_stock(close, volume)
            if r is None:
                continue
            chg = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
            r.update({
                'ticker': ticker,
                'name':   name_map.get(ticker, ticker),
                'sector': sector_map.get(ticker, '—'),
                'price':  round(float(close.iloc[-1]), 2),
                'chg':    round(chg, 2),
            })
            all_rows.append(r)
        except Exception:
            continue

    all_rows.sort(key=lambda x: x['score'], reverse=True)

    strong_buy = [r for r in all_rows if r['signal_en'] == 'STRONG BUY']
    buy        = [r for r in all_rows if r['signal_en'] == 'BUY']
    watch      = [r for r in all_rows if r['signal_en'] == 'WATCH']
    avoid      = [r for r in all_rows if r['signal_en'] == 'AVOID']

    date_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>每日分析报告 {date_str}</title>
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
  .tab-btn {{ padding:6px 14px;border:1px solid #ddd;border-radius:6px;cursor:pointer;font-size:13px;background:white;margin-right:6px }}
  .tab-btn.active {{ background:#2c3e50;color:white;border-color:#2c3e50 }}
</style>
</head>
<body>
<div class="container">

<div class="card">
  <h1>📊 标普500 每日分析报告</h1>
  <p style="color:#aaa;margin:2px 0 14px;font-size:13px">{date_str} &nbsp;·&nbsp; 基于多头趋势策略评分（满分100）</p>
  <div class="stat">大盘环境 <b style="color:{env_color}">{env_label}</b></div>
  <div class="stat">SPY <b>${spy_price:.2f}</b></div>
  <div class="stat">MA200 <b>${spy_ma200:.2f}</b></div>
  <div class="stat">偏离MA200 <b style="color:{env_color}">{spy_pct:+.1f}%</b></div>
  <br><br>
  <span class="badge" style="background:#27ae60">强烈买入 {len(strong_buy)}</span>
  <span class="badge" style="background:#2ecc71">买入 {len(buy)}</span>
  <span class="badge" style="background:#f39c12">观望 {len(watch)}</span>
  <span class="badge" style="background:#e74c3c">回避 {len(avoid)}</span>
</div>

<div class="card">
  <h2>⭐⭐⭐ 强烈买入（评分 ≥80）</h2>
  {make_table(strong_buy, COLS)}
</div>

<div class="card">
  <h2>⭐⭐ 买入（评分 65–79）</h2>
  {make_table(buy[:50], COLS)}
  {'<p style="color:#aaa;font-size:12px">只显示前50条</p>' if len(buy)>50 else ''}
</div>

<div class="card">
  <h2>⚪ 观望（评分 50–64）</h2>
  {make_table(watch[:30], COLS)}
  {'<p style="color:#aaa;font-size:12px">只显示前30条</p>' if len(watch)>30 else ''}
</div>

<div class="card">
  <h2>❌ 回避（评分 &lt;35）— 末尾50只</h2>
  {make_table(avoid[-50:], COLS)}
</div>

</div>
</body>
</html>'''

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ 报告已生成：{OUTPUT_HTML}")
    print(f"   强烈买入 {len(strong_buy)} | 买入 {len(buy)} | 观望 {len(watch)} | 回避 {len(avoid)}")
    webbrowser.open(f'file:///{OUTPUT_HTML}')

    # 自动推送到 GitHub Pages
    import subprocess
    repo_dir = r"C:\Users\ft7b6\OneDrive\Desktop\STOCK"
    try:
        subprocess.run(['git', '-C', repo_dir, 'add', 'index.html'], check=True)
        subprocess.run(['git', '-C', repo_dir, 'commit', '-m', f'report: {date_str}'], check=True)
        subprocess.run(['git', '-C', repo_dir, 'push', 'origin', 'main'], check=True)
        print("✅ 已推送到 GitHub Pages")
    except Exception as e:
        print(f"⚠️ 推送失败：{e}")

if __name__ == '__main__':
    main()
