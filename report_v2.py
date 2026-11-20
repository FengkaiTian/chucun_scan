"""新版报告页面（2026-11-19）：以「主推」规则为中心。旧版页面仍在 report.py（daily_scan.py --old-page）。

页面结构（2026-11-20 改版）：今天买不买 → 最近 20 笔 → 规则和历史回测（折叠）。
注意：信号行的第一格保持 `<td ...><b>TICKER</b> $价格` 形状，daily_scan.git_history_picks 靠它从 git 历史反查。
"""
from html import escape

# 主推规则回测（analysis/private/relax_grid.py；T1 收盘买、持有 10 个交易日、扣 0.10% 来回成本）
BT = {'per_year': 120, 'net': 1.21, 't': 4.5, 'win': 57, 'pos_years': 81, 'worst5': 0.59, 'excess': 0.25,
      'n26': 156, 'net26': 2.43, 'win26': 65, 'random': 0.40}
# 主推分档（analysis/private/rank_tiers.py）：(说明, 每年笔数, 训练每笔净%, 训练胜率%, 2026 笔数, 2026 每笔净%, 2026 胜率%)
GRADES = {
    'A': ('个股深低开 + 市场恐慌', 29, 2.94, 67, 26, 2.46, 69),
    'B': ('只满足其中一项', 49, 1.30, 57, 67, 3.20, 72),
    'C': ('两项都不满足', 41, -0.05, 53, 63, 1.60, 57),
}
GRADE_NOTE = {'A': '优先买', 'B': '正常买', 'C': '历史上不赚钱，只观察'}

CSS = '''
:root{
  color-scheme:light dark;
  --bg:#ffffff; --ink:#1c1c1c; --ink2:#5c5c5c; --ink3:#8f8f8f; --rule:#e3e3e3; --tint:#f5f5f2;
  --up:#c8102e; --down:#087a4c; --hi:#c8102e;
  --gA:#c8102e; --gB:#1f4e8c; --gC:#8f8f8f;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#1b1c1e; --ink:#ececec; --ink2:#b0b0b0; --ink3:#7d7d7d; --rule:#333539; --tint:#242528;
  --up:#ff5c6c; --down:#3ccf8e; --hi:#ff5c6c;
  --gA:#ff5c6c; --gB:#7aa7ff; --gC:#8a8a8a;
}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 "PingFang SC","Hiragino Sans GB","Microsoft YaHei",-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:32px 16px 56px}
.num,.n{font-variant-numeric:tabular-nums}
.up{color:var(--up)} .down{color:var(--down)} .muted{color:var(--ink3)}

.mast{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;
  padding-bottom:12px;border-bottom:2px solid var(--ink)}
.mast h1{margin:0;font-size:17px;font-weight:700}
.mast .mk{font-size:14px;color:var(--ink2)}
.mast .mk b{color:var(--ink);font-weight:600}

.today{padding:34px 0 30px;border-bottom:1px solid var(--rule)}
.today .date{font-size:15px;color:var(--ink2)}
.today .verdict{margin:4px 0 14px;font-size:44px;line-height:1.15;font-weight:800;letter-spacing:-.01em}
.today .verdict.buy{color:var(--hi)}
.today p{margin:0 0 6px;color:var(--ink2);max-width:36em}
.when{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;margin:18px 0 0;font-size:15px}
.when dt{color:var(--ink3)} .when dd{margin:0}
.when dd b{font-weight:700}

h2{font-size:19px;margin:40px 0 6px}
.lead{margin:0 0 14px;color:var(--ink2)}
.lead b{color:var(--ink)}

.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:15px}
th{font-size:13px;font-weight:500;color:var(--ink3);text-align:left;padding:8px 8px 8px 0;border-bottom:1px solid var(--ink2);white-space:nowrap}
td{padding:11px 8px 11px 0;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:baseline}
th:last-child,td:last-child{padding-right:0}
th.n,td.n{text-align:right}
td .co{display:block;font-size:12.5px;color:var(--ink3);max-width:180px;overflow:hidden;text-overflow:ellipsis}
.gr{display:inline-block;width:22px;height:22px;line-height:22px;text-align:center;border-radius:3px;font-weight:800;font-size:13px;color:#fff}
.gA{background:var(--gA)} .gB{background:var(--gB)} .gC{background:transparent;color:var(--gC);box-shadow:inset 0 0 0 1.5px var(--gC)}
.note{font-size:13px;color:var(--ink3);margin:10px 0 0}
.wrapc{white-space:normal}
.wl{display:block;font-size:13px;color:var(--up)}
.prog{display:inline-block;width:46px;height:4px;background:var(--rule);vertical-align:middle;margin-right:6px}
.prog i{display:block;height:100%;background:var(--ink2)}

details{margin-top:40px;border-top:1px solid var(--rule)}
details>summary{cursor:pointer;padding:16px 0;font-size:17px;font-weight:700;list-style:none}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"+";display:inline-block;width:18px;color:var(--ink3);font-weight:400}
details[open]>summary::before{content:"−"}
details>summary:focus-visible{outline:2px solid var(--gB);outline-offset:2px}
.rules{font-size:15px;color:var(--ink2)}
.rules h3{font-size:15px;color:var(--ink);margin:22px 0 6px}
.rules ol,.rules ul{margin:0;padding-left:22px} .rules li{margin:4px 0}
.rules table{font-size:14px} .rules td,.rules th{white-space:normal}

footer{color:var(--ink3);font-size:12.5px;margin-top:44px;padding-top:14px;border-top:1px solid var(--rule)}
@media (max-width:640px){
  .wrap{padding:22px 16px 44px}
  .today .verdict{font-size:36px}
  table{font-size:14px}
  .hide-sm{display:none}
  td .co{display:none}
}
'''

def _sign(v, eps=0.0):
    return 'up' if v is not None and v > eps else 'down' if v is not None and v < -eps else ''


def _pct(v, digits=2):
    return '<span class="muted">—</span>' if v is None else f'<span class="{_sign(v)}">{v:+.{digits}f}%</span>'


def _holding(picks, hold):
    rows = [p for p in picks if p.get('new_rule') and (p.get('held') or 0) < hold]
    if not rows:
        return '<div class="empty" style="padding:18px">目前没有持有中的主推</div>'
    body = ''
    for p in sorted(rows, key=lambda p: p['scan_date']):
        held = p.get('held') or 0
        rn = p.get('rnow')
        state = '<span class="muted">—</span>' if rn is None else ('<span class="up">赢</span>' if rn > 0.10 else '<span class="down">输</span>')
        body += (f'<tr><td><b>{escape(p["ticker"])}</b></td><td class="num">{p["scan_date"]}</td>'
                 f'<td><span class="prog"><i style="width:{held / hold * 100:.0f}%"></i></span><span class="num">{held}/{hold}</span></td>'
                 f'<td class="n">{_pct(rn)}</td><td class="n">{state}</td></tr>')
    return f'''<div class="scroll"><table>
<thead><tr><th>股票</th><th>买入日</th><th>已持有（交易日）</th><th class="n">至今涨跌</th><th class="n">目前（涨超 0.1% 算赢）</th></tr></thead>
<tbody>{body}</tbody></table></div>'''


def _live_record(picks, hold):
    done = [p for p in picks if p.get('new_rule') and p.get(f'r{hold}') is not None]
    n_all = sum(bool(p.get('new_rule')) for p in picks)
    if not done:
        return (f'<div class="kpis">'
                f'<div class="kpi"><div class="k">已出结果</div><div class="v num">0</div><div class="d">共 {n_all} 只主推，2026-11-19 开始记录</div></div>'
                f'<div class="kpi"><div class="k">回测每笔净收益</div><div class="v num up">+{BT["net"]:.2f}%</div><div class="d">2000–2025，t {BT["t"]}</div></div>'
                f'<div class="kpi"><div class="k">回测胜率</div><div class="v num">{BT["win"]}%</div><div class="d">涨超 0.1% 算赢 · 2026 年 {BT["win26"]}%</div></div>'
                f'<div class="kpi"><div class="k">随便买一只</div><div class="v num">+{BT["random"]:.2f}%</div><div class="d">同样持有 10 天</div></div></div>')
    raw = [p[f'r{hold}'] for p in done]
    vals = [v - 0.10 for v in raw]
    avg = sum(vals) / len(vals); win = sum(v > 0.10 for v in raw) / len(raw) * 100   # 涨超 0.1% 算赢
    total = sum(vals)
    return (f'<div class="kpis">'
            f'<div class="kpi"><div class="k">已出结果</div><div class="v num">{len(done)}</div><div class="d">共 {n_all} 只主推</div></div>'
            f'<div class="kpi"><div class="k">实盘每笔净收益</div><div class="v num {_sign(avg)}">{avg:+.2f}%</div><div class="d">回测 +{BT["net"]:.2f}%</div></div>'
            f'<div class="kpi"><div class="k">实盘胜率</div><div class="v num">{win:.0f}%</div><div class="d">涨超 0.1% 算赢 · 回测 {BT["win"]}%</div></div>'
            f'<div class="kpi"><div class="k">累计净收益</div><div class="v num {_sign(total)}">{total:+.1f}%</div><div class="d">每笔收益相加</div></div></div>')


def _rules(ctx):
    return f'''<div class="body"><div class="rules">
  <div>
    <h3>规则（一条都不能少）</h3>
    <ol>
      <li>MACD 柱在零轴下方，连续 3 根抬高（跌势在减弱）</li>
      <li>前一天 RSI14 低于 30，今天回升</li>
      <li>今天低开：开盘价低于昨天收盘价</li>
      <li>这只股票 5 天内第一次出现 1、2 两条</li>
      <li>当天是标普500 成分股；不看 VIX</li>
    </ol>
    <h3 style="margin-top:14px">怎么做</h3>
    <ul>
      <li>信号日的下一个交易日<b>收盘</b>买入（晚一天买也差不多，不要等到第 5 天）</li>
      <li>持有 <b>{ctx["hold"]} 个交易日</b>后收盘卖出</li>
      <li>大跌日可能一次出几十只：事先定好每天最多买几只，单只仓位要小</li>
      <li>财报落在持有期内、或波动 ATR% ≥ 4 的，建议减半</li>
    </ul>
  </div>
  <div>
    <h3>分档（同一天有多只时先买 A）</h3>
    <table>
      <thead><tr><th>档</th><th>条件</th><th class="n">每年</th><th class="n">每笔净收益</th><th class="n">胜率</th><th class="n">2026</th></tr></thead>
      <tbody>{"".join(f'<tr><td><span class="gr g{k}">{k}</span></td><td>{v[0]}</td><td class="n">{v[1]}</td><td class="n {_sign(v[2])}">{v[2]:+.2f}%</td><td class="n">{v[3]}%</td><td class="n"><span class="{_sign(v[5])}">{v[5]:+.2f}%</span> · {v[6]}%</td></tr>' for k, v in GRADES.items())}</tbody>
    </table>
    <p style="margin:6px 0 14px">个股深低开 = 低开 ≥1% 或 开盘低于昨日最低；市场恐慌 = VIX ≥ 25 或 当天主推 ≥ 5 只。2000–2025 年挑、2026 年检验。</p>
    <h3>整体回测（T1 收盘买、持有 10 天、扣 0.10% 来回成本）</h3>
    <table>
      <thead><tr><th></th><th class="n">2000–2025</th><th class="n">2026</th></tr></thead>
      <tbody>
        <tr><td>笔数</td><td class="n">约 {BT["per_year"]} / 年</td><td class="n">{BT["n26"]}</td></tr>
        <tr><td>每笔净收益</td><td class="n up">+{BT["net"]:.2f}%</td><td class="n up">+{BT["net26"]:.2f}%</td></tr>
        <tr><td>胜率（涨超 0.1% 算赢）</td><td class="n">{BT["win"]}%</td><td class="n">{BT["win26"]}%</td></tr>
        <tr><td>赚钱的年份</td><td class="n">{BT["pos_years"]}%</td><td class="n">—</td></tr>
        <tr><td>最差连续 5 年每笔</td><td class="n up">+{BT["worst5"]:.2f}%</td><td class="n">—</td></tr>
      </tbody>
    </table>
    <p style="margin-top:10px">参照：随便挑一天随便买一只成分股，同样拿 10 天每笔 +{BT["random"]:.2f}%。主推多出来的部分主要来自大跌后的反弹，收益集中在少数恐慌日。数据为无幸存者偏差的历史成分股，训练 2000–2025、测试 2026。</p>
  </div>
</div></div>'''


def _others(rows):
    if not rows:
        return '<div class="empty" style="padding:18px">今天没有其他信号</div>'
    body = ''
    for r in rows:
        extra = [s for s in r['matched_signals'] if s not in ('MACD收窄', 'RSI<30反弹')]
        chips = ''.join(f'<span class="chip">{escape(s)}</span>' for s in extra) or '<span class="muted">—</span>'
        body += (f'<tr><td class="stock"><b>{r["ticker"]}</b> $<span class="num">{r["price"]}</span> '
                 f'<span class="{_sign(r["chg"])} num">{r["chg"]:+.2f}%</span></td>'
                 f'<td><span class="tier">{r["tier"]}</span></td><td>{chips}</td>'
                 f'<td class="n">{r["rsi"]}</td><td class="n">{r.get("atr_pct", 0):.1f}%</td>'
                 f'<td>{escape(r["sector"])}</td></tr>')
    return f'''<div class="scroll"><table>
<thead><tr><th>股票</th><th>旧星级</th><th>命中的条件</th><th class="n">RSI</th><th class="n">ATR%</th><th>板块</th></tr></thead>
<tbody>{body}</tbody></table></div>'''


def _all_picks(picks):
    if not picks:
        return '<div class="empty" style="padding:18px">暂无记录</div>'
    body = ''
    for p in picks:
        g = p.get('main_grade')
        tag = f'<span class="gr g{g}">{g}</span>' if g in GRADES else '<span class="tier main">主推</span>'
        body += (f'<tr><td><span style="font-weight:700">{escape(p["ticker"])}</span></td><td>{tag}</td>'
                 f'<td class="num muted">{p["scan_date"]}</td>'
                 + ''.join(f'<td class="n">{_pct(p.get(f"r{n}"))}</td>' for n in (1, 3, 5, 10)) + '</tr>')
    return f'''<div class="scroll tall"><table>
<thead><tr><th>股票</th><th>档</th><th>买入日</th><th class="n">T+1</th><th class="n">T+3</th><th class="n">T+5</th><th class="n">T+10</th></tr></thead>
<tbody>{body}</tbody></table></div>'''


WD = '一二三四五六日'


def _ts(d):
    import pandas as pd
    return pd.Timestamp(d)


def _day(d, wd=True):
    t = _ts(d)
    return f'{t.month}月{t.day}日' + (f'（周{WD[t.weekday()]}）' if wd else '')


def _close_ct(d):
    """美股收盘时间（美国中部时间）。感恩节次日、7/3、12/24 提前到 12:00 收盘。"""
    t = _ts(d)
    if t.month == 11 and t.weekday() == 4 and 23 <= t.day <= 29:
        return '12:00'
    if (t.month, t.day) in ((7, 3), (12, 24)):
        return '12:00'
    return '15:00'


def _buy_time(d):
    """收盘价买入：给出下单的时间窗口（美中）"""
    c = _close_ct(d)
    start = '11:50' if c == '12:00' else '14:50'
    return f'{start}–{c}'


def _pick_cards(rows, ctx):
    """今日主推：一张表；提醒放在股票名下面，手机上不撑宽"""
    body = ''
    for r in rows:
        g = r.get('main_grade') or 'C'
        warn = []
        if r.get('open_lt_pl'):
            warn.append('开盘跌破前一天最低价')
        if r.get('earnings_warn'):
            warn.append(f'持有期内有财报（{escape(r["earnings_warn"])}），买一半')
        if (r.get('atr_pct') or 0) >= 4:
            warn.append(f'波动大（ATR {r["atr_pct"]:.1f}%），买一半')
        w = ''.join(f'<span class="wl">{x}</span>' for x in warn)
        body += (f'<tr><td class="wrapc"><b>{escape(r["ticker"])}</b> <span class="muted">{escape(r["name"])}</span>{w}</td>'
                 f'<td><span class="gr g{g}">{g}</span></td>'
                 f'<td class="n">{r.get("gap", 0):.1f}%</td>'
                 f'<td class="n">${r["price"]}</td></tr>')
    return f'''<div class="scroll"><table>
<thead><tr><th>股票</th><th>档</th><th class="n">开盘跌幅</th><th class="n">收盘价</th></tr></thead>
<tbody>{body}</tbody></table></div>'''


def _last20(replay, hold):
    rows = [r for r in replay if r.get('status') in ('完成', '持有中')][:20]
    if not rows:
        return '<p class="lead">最近没有主推。</p>'
    done = [r for r in rows if r['status'] == '完成']

    def summ(rs, lead):
        if not rs:
            return ''
        v = [r['net'] for r in rs]
        avg = sum(v) / len(v)
        return (f'{lead} <b>{len(v)}</b> 笔，赚钱 <b>{sum(x > 0 for x in v)}</b> 笔，'
                f'平均每笔 <b class="{_sign(avg)}">{avg:+.2f}%</b>。')
    ab = [r for r in done if r['grade'] != 'C']
    lead = summ(done, '已卖出') + (summ(ab, '其中 A、B 档') if ab else '')
    hold_n = len(rows) - len(done)
    if hold_n:
        lead += f'还有 {hold_n} 笔在持有。'
    body = ''
    for r in rows:
        g = r['grade']
        if r['status'] == '完成':
            st = f'<span class="{"up" if r["net"] > 0 else "down"}">{"赚" if r["net"] > 0 else "亏"}</span>'
        else:
            st = (f'<span class="prog"><i style="width:{r["held"] / hold * 100:.0f}%"></i></span>'
                  f'<span class="num muted">第 {r["held"]} 天</span>')
        body += (f'<tr><td><b>{escape(r["ticker"])}</b><span class="co">{escape(r["name"])}</span></td>'
                 f'<td><span class="gr g{g}">{g}</span></td>'
                 f'<td class="num hide-sm muted">{r["signal_day"]:%m-%d}</td>'
                 f'<td class="n hide-sm">{r["gap"]:.1f}%</td>'
                 f'<td class="num">{r["entry_day"]:%m-%d} {_close_ct(r["entry_day"])}</td>'
                 f'<td class="n">{_pct(r["net"])}</td><td>{st}</td></tr>')
    return f'''<p class="lead">{lead}</p><div class="scroll"><table>
<thead><tr><th>股票</th><th>档</th><th class="hide-sm">信号日</th><th class="n hide-sm">开盘跌幅</th><th>买入（美中）</th><th class="n">收益</th><th></th></tr></thead>
<tbody>{body}</tbody></table></div>
<p class="note">开盘跌幅：信号当天开盘价比前一天收盘价低多少。买入按收盘价，所以时间写的是收盘时刻。收益已减去 0.10% 的买卖成本。</p>'''


def _rules_v3(ctx):
    rows = ''.join(
        f'<tr><td><span class="gr g{k}">{k}</span></td><td>{d}</td><td class="n">{v[1]}</td>'
        f'<td class="n {_sign(v[2])}">{v[2]:+.2f}%</td><td class="n">{v[3]}%</td>'
        f'<td class="n"><span class="{_sign(v[5])}">{v[5]:+.2f}%</span>，{v[6]}%</td></tr>'
        for k, v, d in ((k, v, {'A': '两条都满足，优先买', 'B': '满足一条，正常买', 'C': '都不满足，只看不买'}[k])
                        for k, v in GRADES.items()))
    return f'''<div class="rules">
<h3>什么时候出信号</h3>
<ol>
  <li>MACD 柱在零轴下方，连着 3 天往上抬（跌势在变弱）。</li>
  <li>前一天 RSI14 低于 30，当天回升。</li>
  <li>当天开盘就比前一天收盘低。</li>
  <li>这只股票 5 天内第一次出现以上情况。</li>
</ol>
<h3>怎么买卖</h3>
<ul>
  <li>信号第二天收盘前买入（美中 14:50–15:00，提前收盘的日子是 11:50–12:00）。</li>
  <li>拿满 {ctx["hold"]} 个交易日，收盘前卖出。</li>
  <li>大跌后可能一天出十几只：先定好每天最多买几只，单只仓位要小。</li>
  <li>持有期内有财报，或者波动特别大的，买一半。</li>
</ul>
<h3>分档</h3>
<p>看两条：一是开盘跌得深（跌 1% 以上，或跌破前一天最低价）；二是整个市场在恐慌（VIX 25 以上，或当天有 5 只以上同时出信号）。</p>
<div class="scroll"><table>
<thead><tr><th>档</th><th></th><th class="n">每年几笔</th><th class="n">平均每笔</th><th class="n">赚钱比例</th><th class="n">2026 年</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="note">2000–2025 年的历史成分股（含后来被剔除的）回测，2026 年单独检验。收益已减成本。</p>
<h3>整体</h3>
<p>2000–2025 年平均每年约 {BT["per_year"]} 笔，平均每笔 +{BT["net"]:.2f}%，{BT["win"]}% 的交易赚钱，{BT["pos_years"]}% 的年份是赚的。
2026 年到现在 {BT["n26"]} 笔，平均每笔 +{BT["net26"]:.2f}%，{BT["win26"]}% 赚钱。
作为对比，随便挑一只成分股拿 10 天，平均每笔 +{BT["random"]:.2f}%。</p>
</div>'''


def render_report(ctx):
    results = ctx['results']
    main = [r for r in results if r.get('new_rule')]
    vix = ctx['vix_price']
    sig = ctx['signal_day']; ent = ctx['entry_day']; ext = ctx['exit_day']
    if main:
        ab = sum((r.get('main_grade') or 'C') != 'C' for r in main)
        verdict = f'<div class="verdict buy">买 {ab} 只</div>' if ab else '<div class="verdict">只有 C 档</div>'
        intro = (f'<p>{_day(sig, False)}收盘后，有 {len(main)} 只股票符合条件'
                 + (f'，其中 {ab} 只是 A、B 档。' if ab else '，都是 C 档，历史上不赚钱，只看不买。') + '</p>')
        table = _pick_cards(main, ctx)
    else:
        verdict = '<div class="verdict">不买</div>'
        intro = (f'<p>{_day(sig, False)}收盘后，没有股票符合条件。</p>'
                 '<p>这类机会多半跟在大跌后面，一出就是好几只；市场平稳的时候，几周没有也正常。</p>')
        table = ''
    when = ''
    if main:
        when = (f'<dl class="when"><dt>买入</dt><dd><b>{_day(ent)} {_buy_time(ent)}</b>（美中，收盘前）</dd>'
                f'<dt>卖出</dt><dd><b>{_day(ext)}</b> 收盘前，拿 {ctx["hold"]} 个交易日</dd></dl>')
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>标普500 抄底信号</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="mast">
  <h1>标普500 抄底信号</h1>
  <div class="mk">标普 <b class="num">{ctx["spy_price"]:,.0f}</b>，VIX <b class="num">{vix:.1f}</b></div>
</header>

<section class="today">
  <div class="date">{_day(sig)}收盘</div>
  {verdict}
  {intro}
  {when}
</section>
{table}

<h2>最近 20 笔</h2>
{_last20(ctx.get("replay") or [], ctx["hold"])}

<details>
  <summary>规则和历史回测</summary>
  {_rules_v3(ctx)}
</details>

<footer>{ctx["date_str"]} 更新，共看了 {ctx["funnel"]["universe"]} 只成分股。仅供自己研究，不是投资建议；红色是涨，绿色是跌。</footer>
</div>
</body>
</html>'''
