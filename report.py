"""
报告页面渲染 —— 把 daily_scan.py 算出的数据变成 index.html。

注意：今日信号表里每行的第一格必须保持 `<td ...><b>TICKER</b> $价格` 这个形状，
daily_scan.py 靠它从 git 历史的 index.html 里反查过去推荐过的股票（去重 + 近期表现）。
近期表现表里的股票代码故意不用 <b>，避免被当成当日信号再读回去。
"""
from collections import defaultdict
from html import escape

GRADE_ORDER = ['SSS', 'S', 'A', 'B']
GRADE_DESC = {
    'SSS': ('高收益区', '5日中位收益 >2%，样本较少'),
    'S':   ('最强',     '5日胜率 ≥80%'),
    'A':   ('强',       '5日胜率 70–80%'),
    'B':   ('有效',     '5日胜率 65–70%'),
}
BASE_SIGNALS = ('MACD收窄', 'RSI<30反弹')

CSS = '''
:root{
  color-scheme:light dark;
  --bg:#f4f5f7; --card:#fff; --ink:#18212f; --ink2:#5b6676; --ink3:#8a94a3;
  --line:#e6e8ec; --soft:#f3f4f7; --accent:#2f5bd8; --accent-soft:#e9eefc;
  --up:#d92d20; --up-soft:#fdecea; --down:#0a8f55; --down-soft:#e5f6ee;   /* 红涨绿跌 */
  --warn:#b54708; --warn-soft:#fff4e5; --danger:#b42318; --danger-soft:#fdecea;
  --g-sss:#b45309; --g-s:#6d28d9; --g-a:#0e7490; --g-b:#526072;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1319; --card:#181d26; --ink:#e8ecf2; --ink2:#a3adbb; --ink3:#727c8b;
  --line:#2a313d; --soft:#1f2530; --accent:#8aa6ff; --accent-soft:#222c47;
  --up:#ff7a6e; --up-soft:#3a1f1d; --down:#4fd093; --down-soft:#17332a;
  --warn:#fdb022; --warn-soft:#33270f; --danger:#ff7a6e; --danger-soft:#3a1f1d;
  --g-sss:#f5a524; --g-s:#b79bff; --g-a:#5cc8de; --g-b:#9aa6b6;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1320px;margin:0 auto;padding:28px 20px 48px}
.num,td.n,th.n{font-variant-numeric:tabular-nums}
.up{color:var(--up)} .down{color:var(--down)} .muted{color:var(--ink3)}

.top{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:8px 24px;margin-bottom:18px}
h1{margin:0;font-size:24px;letter-spacing:-.01em;font-weight:700}
.sub{margin:4px 0 0;color:var(--ink2);font-size:13px}
.stamp{color:var(--ink2);font-size:13px;text-align:right}
.stamp b{display:block;color:var(--ink);font-size:15px;font-weight:600}

.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--ink2)}
.tile .v{font-size:26px;font-weight:700;line-height:1.25;margin-top:2px}
.tile .d{font-size:12px;color:var(--ink2);margin-top:2px}

.alert{border-radius:10px;padding:12px 16px;margin-bottom:14px;font-size:13px;border:1px solid transparent}
.alert b{display:block;font-size:14px;margin-bottom:2px}
.alert.warn{background:var(--warn-soft);color:var(--ink)} .alert.warn b{color:var(--warn)}
.alert.danger{background:var(--danger-soft);color:var(--ink)} .alert.danger b{color:var(--danger)}

.card{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:14px;overflow:hidden}
.card>header{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;padding:16px 18px 10px}
h2{margin:0;font-size:16px;font-weight:700}
.card>header p{margin:0;color:var(--ink2);font-size:12.5px}
.body{padding:0 18px 16px}
.scroll{overflow-x:auto}

table{width:100%;border-collapse:collapse}
th{font-size:12px;font-weight:600;color:var(--ink2);text-align:left;white-space:nowrap;
  padding:8px 10px;background:var(--soft);border-bottom:1px solid var(--line)}
td{padding:11px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--soft)}
th.n,td.n{text-align:right;white-space:nowrap}
th.grp{text-align:center;border-left:1px solid var(--line);border-right:1px solid var(--line)}
th[data-sort]{cursor:pointer;user-select:none}
th[data-sort]:hover{color:var(--ink)}
th[aria-sort=descending]::after{content:" ↓"} th[aria-sort=ascending]::after{content:" ↑"}
th:first-child,td:first-child{padding-left:18px} th:last-child,td:last-child{padding-right:18px}

.stock b{font-size:15px;letter-spacing:.01em}
.stock .px{font-variant-numeric:tabular-nums;color:var(--ink2);margin-left:2px}
.stock .chg{font-size:12px;font-weight:600;margin-left:6px;font-variant-numeric:tabular-nums}
.stock .co{display:block;font-size:12px;color:var(--ink2);max-width:210px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.earn{display:inline-block;margin-top:4px;padding:1px 7px;border-radius:5px;font-size:11px;font-weight:600;
  background:var(--warn-soft);color:var(--warn)}

.grade{display:inline-block;min-width:38px;text-align:center;padding:3px 8px;border-radius:6px;
  font-size:12px;font-weight:700;letter-spacing:.03em;color:#fff}
@media (prefers-color-scheme:dark){.grade{color:#10141b}}
.g-SSS{background:var(--g-sss)} .g-S{background:var(--g-s)} .g-A{background:var(--g-a)} .g-B{background:var(--g-b)}

.chips{display:flex;flex-wrap:wrap;gap:4px;width:230px}
.chip{padding:2px 8px;border-radius:999px;font-size:11.5px;white-space:nowrap;background:var(--accent-soft);color:var(--accent);font-weight:600}
.chip.base{background:var(--soft);color:var(--ink2);font-weight:500}
.sample{display:block;margin-top:4px;font-size:11px;color:var(--ink3)}

td.wr{min-width:62px}
td.wr span{display:block;font-weight:600}
td.wr i{display:block;height:3px;margin-top:4px;border-radius:2px;background:var(--line);position:relative;overflow:hidden}
td.wr i::after{content:"";position:absolute;inset:0 auto 0 0;width:var(--w);background:var(--accent);border-radius:2px}
td.wr.main span{font-size:15px;color:var(--accent)}
th.main{color:var(--ink)}
td.hot{color:var(--warn);font-weight:700}
.reading{margin:0 18px 12px;padding:9px 12px;border-radius:8px;background:var(--soft);color:var(--ink2);font-size:12.5px}
.reading b{color:var(--ink)}

.empty{padding:36px 18px 40px;text-align:center;color:var(--ink2)}
.empty b{display:block;font-size:16px;color:var(--ink);margin-bottom:6px}

.bars{display:grid;grid-template-columns:max-content 1fr max-content max-content;gap:9px 14px;align-items:center;font-size:13px}
.bars .h{font-size:12px;color:var(--ink2)}
.bar{height:10px;border-radius:5px;background:var(--soft);overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);border-radius:5px}

td.ret{text-align:center;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}
td.ret span{display:inline-block;min-width:68px;padding:3px 8px;border-radius:6px}
td.ret .up{background:var(--up-soft)} td.ret .down{background:var(--down-soft)}
th.c{text-align:center}
.scroll.tall{max-height:560px;overflow-y:auto;border-top:1px solid var(--line)}
.scroll.tall thead th{position:sticky;top:0;z-index:1}
.tag{margin-left:8px;padding:1px 6px;border-radius:4px;font-size:11px;background:var(--soft);color:var(--ink3)}
table.stats{margin-bottom:6px} table.stats td{padding:9px 8px}
table.stats th:first-child,table.stats td:first-child{padding-left:18px}

.summary{display:flex;flex-wrap:wrap;gap:6px 22px;padding:0 18px 12px;font-size:13px;color:var(--ink2)}
.summary b{color:var(--ink);font-variant-numeric:tabular-nums}
.summary b.up{color:var(--up)} .summary b.down{color:var(--down)}

details{padding:0 18px 16px}
summary{cursor:pointer;font-weight:700;font-size:16px;padding:16px 0 6px;list-style-position:inside}
.guide{display:grid;grid-template-columns:1fr 1fr;gap:18px 32px;margin-top:8px;font-size:13px;color:var(--ink2)}
.guide h3{margin:0 0 6px;font-size:13px;color:var(--ink)}
.guide p{margin:0 0 6px}
.guide table td,.guide table th{padding:6px 8px;font-size:12.5px}
.guide table td:first-child,.guide table th:first-child{padding-left:8px}

footer{color:var(--ink3);font-size:12px;text-align:center;margin-top:22px;line-height:1.8}

@media (max-width:820px){
  .wrap{padding:18px 12px 36px}
  .tiles{grid-template-columns:repeat(2,1fr)}
  .tile .v{font-size:22px}
  .guide{grid-template-columns:1fr}
  .stamp{text-align:left}
}
'''

SORT_JS = '''
document.querySelectorAll('table[data-sortable] th[data-sort]').forEach(function(th){
  th.addEventListener('click',function(){
    var tb=th.closest('table').tBodies[0], key=th.dataset.sort;
    var asc=th.getAttribute('aria-sort')==='descending';
    th.closest('tr').parentNode.querySelectorAll('th').forEach(function(x){x.removeAttribute('aria-sort')});
    th.setAttribute('aria-sort',asc?'ascending':'descending');
    Array.from(tb.rows).sort(function(a,b){
      var x=a.querySelector('[data-k="'+key+'"]').dataset.v, y=b.querySelector('[data-k="'+key+'"]').dataset.v;
      var nx=parseFloat(x), ny=parseFloat(y), d=isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny;
      return asc?d:-d;
    }).forEach(function(r){tb.appendChild(r)});
  });
});
'''


def _sign_cls(v, eps=0.0):
    return 'up' if v > eps else 'down' if v < -eps else ''


def _signals_table(rows, bl):
    if not rows:
        return ('<div class="empty"><b>今天没有符合条件的股票</b>'
                '基础信号（MACD 柱零轴下连续收窄 + RSI 从 30 以下回升）本身就少见，'
                '再叠加回测验证过的组合后，空仓日是常态。</div>')

    def wr(r, key, main=False):
        v = r.get(key, 0)
        w = max(0, min(100, (v - 50) * 2))
        return (f'<td class="n wr{" main" if main else ""}" data-k="{key}" data-v="{v}">'
                f'<span>{v:.1f}%</span><i style="--w:{w:.0f}%"></i></td>')

    body = ''
    for r in rows:
        extra = [s for s in r['matched_signals'] if s not in BASE_SIGNALS]
        chips = ''.join(f'<span class="chip base">{escape(s)}</span>' for s in BASE_SIGNALS)
        chips += ''.join(f'<span class="chip">{escape(s)}</span>' for s in extra)
        earn = (f'<span class="earn">财报 {escape(r["earnings_warn"])}，持仓期内注意</span>'
                if r.get('earnings_warn') else '')
        g = r['grade']
        body += f'''<tr>
  <td class="stock" data-k="ticker" data-v="{r["ticker"]}"><b>{r["ticker"]}</b> $<span class="px">{r["price"]}</span><span class="chg {_sign_cls(r["chg"])}">{r["chg"]:+.2f}%</span>
    <span class="co" title="{escape(r["name"])} · {escape(r["sector"])}">{escape(r["name"])} · {escape(r["sector"])}</span>{earn}</td>
  <td data-k="grade" data-v="{3 - GRADE_ORDER.index(g)}"><span class="grade g-{g}" title="{GRADE_DESC[g][0]} · {GRADE_DESC[g][1]}">{g}</span></td>
  <td><div class="chips">{chips}</div><span class="sample">回测样本 {r["sample"]} 次</span></td>
  {wr(r, "wr1")}{wr(r, "wr3")}{wr(r, "wr5", True)}{wr(r, "wr10")}
  <td class="n" data-k="rsi" data-v="{r["rsi"]}">{r["rsi"]}</td>
  <td class="n" data-k="cci" data-v="{r["cci"]}">{r["cci"]}</td>
  <td class="n" data-k="cmf" data-v="{r["cmf"]}">{r["cmf"]:+.3f}</td>
  <td class="n" data-k="wr" data-v="{r["wr"]}">{r["wr"]}</td>
  <td class="n" data-k="d52" data-v="{r["dist_52w"]}">+{r["dist_52w"]:.1f}%</td>
  <td class="n">{escape(r["volume_status"])}</td>
  <td class="n {"hot" if r.get("atr_pct", 0) >= 4 else ""}" data-k="atr" data-v="{r.get("atr_pct", 0)}">{r.get("atr_pct", 0):.1f}%</td>
  <td class="n" data-k="h252" data-v="{r.get("dist_high252", 0)}">{r.get("dist_high252", 0):.0f}%</td>
  <td class="n">${r["stop_loss"]}</td>
</tr>'''

    return f'''<div class="scroll"><table data-sortable>
<thead>
<tr>
  <th rowspan="2" data-sort="ticker">股票</th>
  <th rowspan="2" data-sort="grade">评级</th>
  <th rowspan="2">匹配的信号组合</th>
  <th colspan="4" class="grp">该组合的历史回测胜率</th>
  <th rowspan="2" class="n" data-sort="rsi">RSI</th>
  <th rowspan="2" class="n" data-sort="cci">CCI</th>
  <th rowspan="2" class="n" data-sort="cmf">CMF</th>
  <th rowspan="2" class="n" data-sort="wr">威廉%R</th>
  <th rowspan="2" class="n" data-sort="d52">距52周低</th>
  <th rowspan="2" class="n">成交量/5日均</th>
  <th rowspan="2" class="n" data-sort="atr" title="14日平均真实波幅占股价比例，衡量单日波动">ATR%</th>
  <th rowspan="2" class="n" data-sort="h252" title="现价距过去一年最高收盘价">距1年高点</th>
  <th rowspan="2" class="n">止损价</th>
</tr>
<tr>
  <th class="n" data-sort="wr1" title="随机基准 {bl["ret1"]}%">T+1</th>
  <th class="n" data-sort="wr3" title="随机基准 {bl["ret3"]}%">T+3</th>
  <th class="n main" data-sort="wr5" aria-sort="descending" title="随机基准 {bl["ret5"]}%">T+5</th>
  <th class="n" data-sort="wr10" title="随机基准 {bl["ret10"]}%">T+10</th>
</tr>
</thead>
<tbody>{body}</tbody></table></div>'''


def _sector_card(rows):
    if not rows:
        return ''
    stats = defaultdict(lambda: [0, 0.0])
    for r in rows:
        stats[r['sector']][0] += 1
        stats[r['sector']][1] += r.get('wr5', 0)
    ordered = sorted(stats.items(), key=lambda x: -x[1][0])
    top = ordered[0][1][0]
    cells = '<span class="h">板块</span><span class="h"></span><span class="h">信号数</span><span class="h">平均 T+5 胜率</span>'
    for sec, (n, s) in ordered:
        cells += (f'<span>{escape(sec)}</span><span class="bar"><i style="width:{n / top * 100:.0f}%"></i></span>'
                  f'<b class="num">{n}</b><span class="num">{s / n:.1f}%</span>')
    return f'''<section class="card">
  <header><h2>板块分布</h2><p>今日信号集中在哪些行业</p></header>
  <div class="body"><div class="bars">{cells}</div></div>
</section>'''


def _recent_card(picks):
    if not picks:
        return ''

    def cell(v):
        if v is None:
            return '<td class="ret muted">—</td>'
        return f'<td class="ret"><span class="{_sign_cls(v, 0.2)}">{v:+.2f}%</span></td>'

    body = ''
    for p in picks:
        g = p.get('grade')
        grade = f'<span class="grade g-{g}">{g}</span>' if g else '<span class="muted">—</span>'
        tag = '<span class="tag">补算</span>' if p.get('source') == 'backfill' else ''
        body += (f'<tr><td><span style="font-weight:700">{p["ticker"]}</span>{tag}</td>'
                 f'<td>{grade}</td><td class="num muted">{p["scan_date"]}</td>'
                 f'{cell(p["r1"])}{cell(p["r3"])}{cell(p["r5"])}{cell(p["r10"])}</tr>')

    # 汇总：每个持有期的上涨比例、平均收益，以及同期标普500的平均收益
    stat_rows = ''
    for n in (1, 3, 5, 10):
        done = [p for p in picks if p[f'r{n}'] is not None]
        if not done:
            continue
        vals = [p[f'r{n}'] for p in done]
        bench = [p[f'b{n}'] for p in done if p.get(f'b{n}') is not None]
        avg = sum(vals) / len(vals)
        med = sorted(vals)[len(vals) // 2]
        wins = sum(v > 0 for v in vals)
        b_avg = sum(bench) / len(bench) if bench else None
        excess = avg - b_avg if b_avg is not None else None
        stat_rows += (
            f'<tr><td><b>T+{n}</b></td><td class="n">{len(vals)}</td>'
            f'<td class="n"><b>{wins / len(vals) * 100:.1f}%</b></td>'
            f'<td class="n {_sign_cls(avg)}"><b>{avg:+.2f}%</b></td>'
            f'<td class="n {_sign_cls(med)}">{med:+.2f}%</td>'
            + (f'<td class="n {_sign_cls(b_avg)}">{b_avg:+.2f}%</td>'
               f'<td class="n {_sign_cls(excess)}"><b>{excess:+.2f}%</b></td>' if b_avg is not None
               else '<td class="n muted">—</td><td class="n muted">—</td>')
            + '</tr>')
    stats = f'''<div class="scroll"><table class="stats">
    <thead><tr><th>持有期</th><th class="n">已出结果</th><th class="n">上涨比例</th><th class="n">平均收益</th>
      <th class="n">中位收益</th><th class="n">同期标普500</th><th class="n">超额</th></tr></thead>
    <tbody>{stat_rows}</tbody></table></div>''' if stat_rows else ''

    n_back = sum(p.get('source') == 'backfill' for p in picks)
    first, last = picks[-1]['scan_date'], picks[0]['scan_date']
    note = (f' · 其中 {n_back} 条是事后补算（扫描停跑期间，按当时能看到的数据逐日重放）' if n_back else '')
    return f'''<section class="card">
  <header><h2>选股的实际表现</h2><p>{first} 至 {last} 共 {len(picks)} 条，从入场日收盘价起算的真实涨跌{note}</p></header>
  {stats}
  <header><h2 style="font-size:14px">逐条明细</h2><p>— 表示交易日还没到</p></header>
  <div class="scroll tall"><table>
    <thead><tr><th>股票</th><th>评级</th><th>入场日</th><th class="c">T+1</th><th class="c">T+3</th><th class="c">T+5</th><th class="c">T+10</th></tr></thead>
    <tbody>{body}</tbody>
  </table></div>
</section>'''


def _reading(results, f):
    """回测复盘得出的几条参考（只是倾向，统计上不显著，不是过滤规则）"""
    if not results:
        return ''
    n = f['base']
    if n >= 5:
        env = f'今天全市场有 <b>{n}</b> 只股票触发基础信号（≥5 只，说明是大盘整体超卖）：回测中这类日子反弹更好、大跌更少。'
    elif n == 1:
        env = '今天全市场只有 <b>1</b> 只触发基础信号（孤立信号）：回测中这类日子表现偏弱，更像个股自身问题而非大盘超卖。'
    else:
        env = f'今天全市场有 <b>{n}</b> 只股票触发基础信号。'
    return (f'<div class="reading">{env}<br>'
            '<b>财报</b>落在持仓期内的，回测中 T+5 大跌（≤-5%）比例约 28% 对 10%，建议回避或减仓；'
            '<b>ATR% ≥ 4</b>（橙色）既容易大涨也容易大跌，建议仓位减半。</div>')


def render_report(ctx):
    """ctx: date_str, n_combos, grade_counts, bl, is_bull, spy_price, spy_pct,
    vix_price, vix_high, vix_extreme, results, funnel, recent"""
    results = ctx['results']
    f = ctx['funnel']
    bl = ctx['bl']
    vix = ctx['vix_price']

    env_cls = 'up' if ctx['is_bull'] else 'down'
    if ctx['vix_extreme']:
        vix_word, vix_style = '极度恐慌', 'color:var(--danger)'
    elif ctx['vix_high']:
        vix_word, vix_style = '偏高', 'color:var(--warn)'
    else:
        vix_word, vix_style = '平静', ''

    alert = ''
    if ctx['vix_extreme']:
        alert = ('<div class="alert danger"><b>VIX ≥ 30，信号暂停</b>'
                 '回测显示极端恐慌期的超卖信号失败率很高，等 VIX 回落到 30 以下再恢复出信号。</div>')
    elif ctx['vix_high']:
        alert = ('<div class="alert warn"><b>VIX 高于 20，信号可信度更高</b>'
                 '回测显示 VIX > 20 时超卖反弹信号的胜率大约高 5–10 个百分点。</div>')

    funnel_bits = [f'扫描 {f["universe"]} 只', f'基础信号 {f["base"]} 只']
    if f['deduped']:
        funnel_bits.append(f'近期已推荐 {f["deduped"]} 只已剔除')
    grade_rows = ''.join(
        f'<tr><td><span class="grade g-{g}">{g}</span></td><td>{GRADE_DESC[g][0]}</td>'
        f'<td>{GRADE_DESC[g][1]}</td><td class="n">{ctx["grade_counts"].get(g, 0)} 种</td></tr>'
        for g in GRADE_ORDER)

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>标普500 信号扫描 · {ctx["date_str"]}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<div class="top">
  <div>
    <h1>标普500 每日信号扫描</h1>
    <p class="sub">在 MACD 金叉之前入场 · {ctx["n_combos"]} 种经 6 年回测验证的信号组合</p>
  </div>
  <div class="stamp">扫描时间<b class="num">{ctx["date_str"]}</b></div>
</div>

<div class="tiles">
  <div class="tile"><div class="k">今日信号</div><div class="v num">{len(results)} <span style="font-size:14px;font-weight:500;color:var(--ink2)">只</span></div><div class="d">{" · ".join(funnel_bits)}</div></div>
  <div class="tile"><div class="k">市场环境</div><div class="v {env_cls}">{"牛市" if ctx["is_bull"] else "熊市"}</div><div class="d">标普500 在 200 日均线{"上方" if ctx["is_bull"] else "下方"}</div></div>
  <div class="tile"><div class="k">标普500</div><div class="v num">{ctx["spy_price"]:,.0f}</div><div class="d">较 200 日均线 <span class="num {env_cls}">{ctx["spy_pct"]:+.1f}%</span></div></div>
  <div class="tile"><div class="k">VIX 恐慌指数</div><div class="v num" style="{vix_style}">{vix:.1f}</div><div class="d">{vix_word}（20 以上偏高，30 以上暂停）</div></div>
</div>

{alert}

<section class="card">
  <header><h2>今日信号</h2><p>按 T+5 胜率排序，点表头可换排序 · 建议持仓 5 个交易日</p></header>
  {_reading(results, f)}
  {_signals_table(results, bl)}
</section>

{_sector_card(results)}

{_recent_card(ctx["recent"])}

<section class="card">
  <details>
    <summary>怎么读这份报告</summary>
    <div class="guide">
      <div>
        <h3>选股逻辑</h3>
        <p>先要求基础信号：MACD 柱状图在零轴下方连续 3 根收窄（空头减弱），同时 RSI 从 30 以下开始回升。</p>
        <p>再看这只股票还满足哪些附加条件，只有命中「回测中 T+1 / T+3 / T+5 / T+10 全部跑赢随机基准」的组合才会出现在表里。</p>
        <h3 style="margin-top:14px">表里的数字</h3>
        <p><b>回测胜率</b>是这个信号组合在 2020–2025 年历史上的上涨概率，不是对这只股票的预测。随机买入的基准是 T+1 {bl["ret1"]}%、T+3 {bl["ret3"]}%、T+5 {bl["ret5"]}%、T+10 {bl["ret10"]}%，进度条从 50% 起算。</p>
        <p><b>止损价</b> = 20 日均线 × 0.97。<b>成交量</b>是当日相对 5 日均量的倍数。</p>
        <p><b>回测胜率偏高，请打折看：</b>这些组合是从 1626 种里挑出来的，存在选择偏差。同一批信号在回测期内的实际 T+5 上涨比例约 56–59%，而不是表里的 65–80%；其中约 70% 的收益来自大盘本身的涨跌。评级只有 SSS 一档还显示出弱的排序价值。</p>
        <p>涨跌颜色：<span class="up">红色上涨</span>，<span class="down">绿色下跌</span>。</p>
      </div>
      <div>
        <h3>评级</h3>
        <table>
          <thead><tr><th>等级</th><th>含义</th><th>标准</th><th class="n">组合数</th></tr></thead>
          <tbody>{grade_rows}</tbody>
        </table>
      </div>
    </div>
  </details>
</section>

<footer>仅供研究参考，不构成投资建议。历史回测不代表未来表现。</footer>

</div>
<script>{SORT_JS}</script>
</body>
</html>'''
