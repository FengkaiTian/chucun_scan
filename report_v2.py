"""新版报告页面（2026-11-19）：以「主推」规则为中心。旧版页面仍在 report.py（daily_scan.py --old-page）。

页面结构：今日主推 → 持仓中（主推，未满持有期）→ 主推实盘记录 → 规则与回测 → 其他信号（旧星级，折叠）→ 全部选股记录（折叠）。
注意：信号行的第一格保持 `<td ...><b>TICKER</b> $价格` 形状，daily_scan.git_history_picks 靠它从 git 历史反查。
"""
from html import escape

# 主推规则回测（analysis/private/relax_grid.py；T1 收盘买、持有 10 个交易日、扣 0.10% 来回成本）
BT = {'per_year': 120, 'net': 1.21, 't': 4.5, 'win': 57, 'pos_years': 81, 'worst5': 0.59, 'excess': 0.25,
      'n26': 156, 'net26': 2.43, 'win26': 65, 'random': 0.40}

CSS = '''
:root{
  color-scheme:light dark;
  --bg:#f6f7f9; --card:#ffffff; --ink:#111827; --ink2:#4b5563; --ink3:#9ca3af; --line:#e5e7eb; --soft:#f3f4f6;
  --brand:#2563eb; --brand-soft:#eff4ff; --brand-ink:#1d4ed8;
  --up:#dc2626; --up-soft:#fef2f2; --down:#059669; --down-soft:#ecfdf5;   /* 红涨绿跌 */
  --warn:#b45309; --warn-soft:#fffbeb;
  --shadow:0 1px 2px rgba(16,24,40,.04),0 1px 3px rgba(16,24,40,.06);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0b0f17; --card:#131a24; --ink:#e5e9f0; --ink2:#a1abba; --ink3:#6b7686; --line:#232c39; --soft:#1a222e;
  --brand:#6b9bff; --brand-soft:#17233d; --brand-ink:#9dbbff;
  --up:#ff6b6b; --up-soft:#2e1719; --down:#34d399; --down-soft:#11291f;
  --warn:#f5b454; --warn-soft:#2b2112; --shadow:none;
}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:28px 16px 48px}
.num,td.n,.n{font-variant-numeric:tabular-nums}
.up{color:var(--up)} .down{color:var(--down)} .muted{color:var(--ink3)}

.top{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-bottom:18px}
.top h1{margin:0;font-size:24px;letter-spacing:-.01em}
.top .sub{margin:4px 0 0;color:var(--ink2);font-size:13.5px}
.mkt{display:flex;gap:8px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:6px;padding:6px 11px;border-radius:999px;background:var(--card);border:1px solid var(--line);font-size:13px;color:var(--ink2);white-space:nowrap}
.pill b{color:var(--ink)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--ink3)}
.dot.g{background:var(--down)} .dot.r{background:var(--up)} .dot.y{background:var(--warn)}

.card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);margin-bottom:16px;overflow:hidden}
.card>header{padding:16px 18px 10px;display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.card h2{margin:0;font-size:17px}
.card header p{margin:0;color:var(--ink2);font-size:13px}
.body{padding:0 18px 18px}

.hero{border-color:color-mix(in srgb,var(--brand) 35%,var(--line))}
.hero>header{background:var(--brand-soft)}
.hero h2{color:var(--brand-ink)}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;background:var(--brand);color:#fff;font-size:12px;font-weight:700;vertical-align:2px;margin-left:6px}
.plan{display:flex;gap:10px;flex-wrap:wrap;padding:12px 18px;border-bottom:1px solid var(--line);font-size:13.5px;color:var(--ink2)}
.plan span{background:var(--soft);padding:4px 10px;border-radius:8px}
.plan b{color:var(--ink)}

.picks{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;padding:16px 18px 18px}
.pick{border:1px solid var(--line);border-radius:12px;padding:14px;background:var(--card)}
.pick .tk{font-size:22px;font-weight:800;letter-spacing:.01em}
.pick .co{color:var(--ink2);font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pick .px{margin-top:8px;font-size:15px}
.pick dl{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;margin:10px 0 0;font-size:12.5px}
.pick dt{color:var(--ink3)} .pick dd{margin:0;text-align:right}
.warnchip{display:inline-block;margin-top:8px;padding:2px 8px;border-radius:6px;background:var(--warn-soft);color:var(--warn);font-size:12px;font-weight:600}
.empty{padding:28px 18px 30px;text-align:center;color:var(--ink2);font-size:14px}
.empty b{display:block;color:var(--ink);font-size:17px;margin-bottom:4px}

.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{font-size:12px;font-weight:600;color:var(--ink3);text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:0}
th:first-child,td:first-child{padding-left:18px} th:last-child,td:last-child{padding-right:18px}
th.n,td.n{text-align:right}
.chip{display:inline-block;padding:1px 7px;margin:1px 2px 1px 0;border-radius:5px;background:var(--soft);color:var(--ink2);font-size:12px}
.tier{display:inline-block;min-width:34px;text-align:center;padding:1px 7px;border-radius:5px;font-size:12px;font-weight:700;background:var(--soft);color:var(--ink2)}
.tier.main{background:var(--brand);color:#fff}
.prog{display:inline-block;width:70px;height:6px;border-radius:3px;background:var(--soft);vertical-align:middle;margin-right:6px;overflow:hidden}
.prog i{display:block;height:100%;background:var(--brand)}

.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border-top:1px solid var(--line)}
.kpi{background:var(--card);padding:14px 18px}
.kpi .k{font-size:12px;color:var(--ink3)} .kpi .v{font-size:22px;font-weight:700;margin-top:2px} .kpi .d{font-size:12px;color:var(--ink2)}

.rules{display:grid;grid-template-columns:1fr 1fr;gap:18px 28px;font-size:13.5px;color:var(--ink2)}
.rules h3{margin:0 0 6px;font-size:14px;color:var(--ink)}
.rules ol,.rules ul{margin:0;padding-left:20px} .rules li{margin:3px 0}
.rules table td,.rules table th{padding:6px 8px} .rules table td:first-child,.rules table th:first-child{padding-left:0}

details.card>summary{list-style:none;cursor:pointer;padding:15px 18px;display:flex;justify-content:space-between;align-items:center;gap:10px}
details.card>summary::-webkit-details-marker{display:none}
details.card>summary h2{font-size:16px}
details.card>summary .muted{font-size:13px}
details.card>summary::after{content:"展开";font-size:12.5px;color:var(--brand)}
details.card[open]>summary::after{content:"收起"}
details.card[open]>summary{border-bottom:1px solid var(--line)}
.tall{max-height:520px;overflow:auto}
.tall thead th{position:sticky;top:0;background:var(--card)}

footer{color:var(--ink3);font-size:12px;text-align:center;margin-top:20px;line-height:1.8}
@media (max-width:720px){
  .wrap{padding:18px 12px 36px}
  .top h1{font-size:20px}
  .kpis{grid-template-columns:repeat(2,1fr)}
  .rules{grid-template-columns:1fr}
}
'''


def _sign(v, eps=0.0):
    return 'up' if v is not None and v > eps else 'down' if v is not None and v < -eps else ''


def _pct(v, digits=2):
    return '<span class="muted">—</span>' if v is None else f'<span class="{_sign(v)}">{v:+.{digits}f}%</span>'


def _pick_cards(rows, ctx):
    if not rows:
        return (f'<div class="empty"><b>今天没有主推</b>'
                f'这条规则平均每年约 {BT["per_year"]} 只，常常集中在大跌后的几天出现，空仓日很正常。</div>')
    cards = ''
    for r in rows:
        earn = f'<span class="warnchip">财报 {escape(r["earnings_warn"])}，持有期内</span>' if r.get('earnings_warn') else ''
        atr = r.get('atr_pct', 0)
        cards += f'''<div class="pick">
  <div class="tk">{escape(r["ticker"])}</div>
  <div class="co" title="{escape(r["name"])} · {escape(r["sector"])}">{escape(r["name"])} · {escape(r["sector"])}</div>
  <div class="px num">${r["price"]} <span class="{_sign(r["chg"])}">{r["chg"]:+.2f}%</span></div>
  <dl>
    <dt>买入</dt><dd>{ctx["entry_day"]} 收盘</dd>
    <dt>卖出</dt><dd>约 {ctx["exit_day"]} 收盘</dd>
    <dt>RSI</dt><dd class="num">{r["rsi"]}</dd>
    <dt>波动 ATR%</dt><dd class="num {"up" if atr >= 4 else ""}">{atr:.1f}%{"（偏大）" if atr >= 4 else ""}</dd>
    <dt>距 1 年高点</dt><dd class="num">{r.get("dist_high252", 0):.0f}%</dd>
  </dl>{earn}
</div>'''
    return f'<div class="picks">{cards}</div>'


def _holding(picks, hold):
    rows = [p for p in picks if p.get('new_rule') and (p.get('held') or 0) < hold]
    if not rows:
        return '<div class="empty" style="padding:18px">目前没有持有中的主推</div>'
    body = ''
    for p in sorted(rows, key=lambda p: p['scan_date']):
        held = p.get('held') or 0
        ex = (p['rnow'] - 0.10 - p['bnow']) if p.get('rnow') is not None and p.get('bnow') is not None else None
        body += (f'<tr><td><b>{escape(p["ticker"])}</b></td><td class="num">{p["scan_date"]}</td>'
                 f'<td><span class="prog"><i style="width:{held / hold * 100:.0f}%"></i></span><span class="num">{held}/{hold}</span></td>'
                 f'<td class="n">{_pct(p.get("rnow"))}</td><td class="n">{_pct(p.get("bnow"))}</td><td class="n">{_pct(ex)}</td></tr>')
    return f'''<div class="scroll"><table>
<thead><tr><th>股票</th><th>买入日</th><th>已持有（交易日）</th><th class="n">至今涨跌</th><th class="n">同期标普500</th><th class="n">扣费后超额</th></tr></thead>
<tbody>{body}</tbody></table></div>'''


def _live_record(picks, hold):
    done = [p for p in picks if p.get('new_rule') and p.get(f'r{hold}') is not None]
    n_all = sum(bool(p.get('new_rule')) for p in picks)
    if not done:
        return (f'<div class="kpis">'
                f'<div class="kpi"><div class="k">已出结果</div><div class="v num">0</div><div class="d">共 {n_all} 只主推，2026-11-19 开始记录</div></div>'
                f'<div class="kpi"><div class="k">回测每笔净收益</div><div class="v num up">+{BT["net"]:.2f}%</div><div class="d">2000–2025，t {BT["t"]}</div></div>'
                f'<div class="kpi"><div class="k">回测胜率</div><div class="v num">{BT["win"]}%</div><div class="d">2026 年 {BT["win26"]}%</div></div>'
                f'<div class="kpi"><div class="k">随便买一只</div><div class="v num">+{BT["random"]:.2f}%</div><div class="d">同样持有 10 天</div></div></div>')
    vals = [p[f'r{hold}'] - 0.10 for p in done]
    bench = [p[f'b{hold}'] for p in done if p.get(f'b{hold}') is not None]
    avg = sum(vals) / len(vals); win = sum(v > 0 for v in vals) / len(vals) * 100
    ex = avg - sum(bench) / len(bench) if bench else None
    return (f'<div class="kpis">'
            f'<div class="kpi"><div class="k">已出结果</div><div class="v num">{len(done)}</div><div class="d">共 {n_all} 只主推</div></div>'
            f'<div class="kpi"><div class="k">实盘每笔净收益</div><div class="v num {_sign(avg)}">{avg:+.2f}%</div><div class="d">回测 +{BT["net"]:.2f}%</div></div>'
            f'<div class="kpi"><div class="k">实盘胜率</div><div class="v num">{win:.0f}%</div><div class="d">回测 {BT["win"]}%</div></div>'
            f'<div class="kpi"><div class="k">比标普500多赚</div><div class="v num">{_pct(ex)}</div><div class="d">回测 +{BT["excess"]:.2f}%</div></div></div>')


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
    <h3>回测（T1 收盘买、持有 10 天、扣 0.10% 来回成本）</h3>
    <table>
      <thead><tr><th></th><th class="n">2000–2025</th><th class="n">2026</th></tr></thead>
      <tbody>
        <tr><td>笔数</td><td class="n">约 {BT["per_year"]} / 年</td><td class="n">{BT["n26"]}</td></tr>
        <tr><td>每笔净收益</td><td class="n up">+{BT["net"]:.2f}%</td><td class="n up">+{BT["net26"]:.2f}%</td></tr>
        <tr><td>胜率</td><td class="n">{BT["win"]}%</td><td class="n">{BT["win26"]}%</td></tr>
        <tr><td>赚钱的年份</td><td class="n">{BT["pos_years"]}%</td><td class="n">—</td></tr>
        <tr><td>最差连续 5 年每笔</td><td class="n up">+{BT["worst5"]:.2f}%</td><td class="n">—</td></tr>
        <tr><td>比标普500多赚</td><td class="n">+{BT["excess"]:.2f}%</td><td class="n">—</td></tr>
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
        tag = '<span class="tier main">主推</span>' if p.get('new_rule') else f'<span class="tier">{escape(p.get("tier") or p.get("grade") or "—")}</span>'
        body += (f'<tr><td><span style="font-weight:700">{escape(p["ticker"])}</span></td><td>{tag}</td>'
                 f'<td class="num muted">{p["scan_date"]}</td>'
                 + ''.join(f'<td class="n">{_pct(p.get(f"r{n}"))}</td>' for n in (1, 3, 5, 10)) + '</tr>')
    return f'''<div class="scroll tall"><table>
<thead><tr><th>股票</th><th>类别</th><th>买入日</th><th class="n">T+1</th><th class="n">T+3</th><th class="n">T+5</th><th class="n">T+10</th></tr></thead>
<tbody>{body}</tbody></table></div>'''


def render_report(ctx):
    results = ctx['results']
    main = [r for r in results if r.get('new_rule')]
    others = [r for r in results if not r.get('new_rule')]
    recent = ctx.get('recent') or []
    vix = ctx['vix_price']
    vdot = 'r' if vix >= 30 else 'y' if vix >= 20 else 'g'
    env_dot = 'g' if ctx['is_bull'] else 'r'
    f = ctx['funnel']
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>标普500 主推信号</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<div class="top">
  <div>
    <h1>标普500 主推信号</h1>
    <p class="sub">信号日 {ctx["signal_day"]} 收盘 · 扫描于 {ctx["date_str"]} · 扫描 {f["universe"]} 只，基础信号 {f["base"]} 只</p>
  </div>
  <div class="mkt">
    <span class="pill"><i class="dot {env_dot}"></i>标普500 <b class="num">{ctx["spy_price"]:,.0f}</b> <span class="num">{ctx["spy_pct"]:+.1f}%</span> vs 200 日线</span>
    <span class="pill"><i class="dot {vdot}"></i>VIX <b class="num">{vix:.1f}</b></span>
  </div>
</div>

<section class="card hero">
  <header><h2>今日主推<span class="badge">{len(main)} 只</span></h2><p>明天收盘买，持有 {ctx["hold"]} 个交易日</p></header>
  <div class="plan"><span>买入日 <b>{ctx["entry_day"]}</b> 收盘</span><span>卖出日 约 <b>{ctx["exit_day"]}</b> 收盘</span><span>回测每笔 <b class="up">+{BT["net"]:.2f}%</b> · 胜率 <b>{BT["win"]}%</b></span></div>
  {_pick_cards(main, ctx)}
</section>

<section class="card">
  <header><h2>持有中</h2><p>还没满 {ctx["hold"]} 个交易日的主推</p></header>
  {_holding(recent, ctx["hold"])}
</section>

<section class="card">
  <header><h2>主推实盘记录</h2><p>持有满 {ctx["hold"]} 天、扣 0.10% 成本后的结果，和回测对照</p></header>
  {_live_record(recent, ctx["hold"])}
</section>

<section class="card">
  <header><h2>规则与回测</h2><p>2026-11-19 起使用</p></header>
  {_rules(ctx)}
</section>

<details class="card">
  <summary><h2>其他信号 <span class="muted">旧星级，仅参考 · {len(others)} 只</span></h2></summary>
  {_others(others)}
</details>

<details class="card">
  <summary><h2>全部选股记录 <span class="muted">近 180 天 · {len(recent)} 条</span></h2></summary>
  {_all_picks(recent)}
</details>

<footer>仅供研究参考，不构成投资建议。历史回测不代表未来表现。<br>红色上涨 · 绿色下跌</footer>
</div>
</body>
</html>'''
