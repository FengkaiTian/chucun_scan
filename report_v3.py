"""第三版报告页面（2026-11-20）：信息面板式。旧版仍可用：daily_scan.py --v2-page / --old-page。

结构：市场概况（5 个小卡片带走势线）→ 今天买不买 + 观察名单 → 大盘与主推信号图（可悬停）
→ 最近 20 笔（带每笔走势线）→ 分档成绩 + 规则。
"""
import json
from html import escape
from report_v2 import BT, GRADES, COND_GRADES, IN_GRADE, MARKET_CONDS, cond_tags, _day, _close_ct, _buy_time, _sign
CG_SUMMARY = (f'{len(COND_GRADES)} 个条件入选（' + '、'.join(f'{g} {sum(v[0] == g for v in COND_GRADES.values())}' for g in ('SSS', 'A', 'B')) + '）')
GL = {'A': 'SSS', 'B': 'A', 'C': 'B'}   # 内部仍用 A/B/C 排序和存档，页面显示 SSS/A/B

CSS = '''
:root{
  color-scheme:light;
  --bg:#f3f3f1; --panel:#ffffff; --line:#e4e3de; --line2:#cfcec8;
  --ink:#0b0b0b; --ink2:#52514e; --ink3:#86857f;
  --up:#d03b2f; --down:#0f8a5f;
  --gA:#6250d6; --gB:#2a78d6; --gC:#a3a29b;
  --spx:#3d3c39; --hl:#fbf3e0;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  color-scheme:dark;
  --bg:#121211; --panel:#1a1a19; --line:#2c2c2a; --line2:#3d3d3a;
  --ink:#f0efec; --ink2:#c3c2b7; --ink3:#8b8a83;
  --up:#ef6b5f; --down:#3cc48f;
  --gA:#9085e9; --gB:#3987e5; --gC:#6b6a64;
  --spx:#d8d7d0; --hl:#2a2619;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#121211; --panel:#1a1a19; --line:#2c2c2a; --line2:#3d3d3a;
  --ink:#f0efec; --ink2:#c3c2b7; --ink3:#8b8a83;
  --up:#ef6b5f; --down:#3cc48f;
  --gA:#9085e9; --gB:#3987e5; --gC:#6b6a64;
  --spx:#d8d7d0; --hl:#2a2619;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:22px 16px 48px}
.n,.num{font-variant-numeric:tabular-nums}
.up{color:var(--up)} .down{color:var(--down)} .m{color:var(--ink3)}

.top{display:flex;justify-content:space-between;align-items:flex-end;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.top h1{margin:0;font-size:20px;font-weight:700}
.top p{margin:0;color:var(--ink3);font-size:13px}

.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:10px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:11px 13px 9px;min-width:0}
.tile .k{font-size:12.5px;color:var(--ink2)}
.tile .v{font-size:22px;font-weight:650;line-height:1.25;margin-top:2px}
.tile .v small{font-size:13px;font-weight:500;margin-left:6px}
.tile .s{font-size:12px;color:var(--ink3);margin-top:1px;line-height:1.45}
.tile svg{display:block;width:100%;height:26px;margin-top:6px}

.grid2{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:10px;margin-bottom:10px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:10px;min-width:0}
.grid2 .panel{margin-bottom:0}
.ph{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;padding:13px 16px 9px}
.ph h2{margin:0;font-size:15px;font-weight:650}
.ph .r{font-size:12.5px;color:var(--ink3)}
.pb{padding:0 16px 14px}

.verdict{font-size:34px;font-weight:750;line-height:1.15;margin:2px 0 8px}
.verdict.buy{color:var(--up)}
.today p{margin:0 0 6px;color:var(--ink2)}
.when{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:12px 0 4px;padding:10px 12px;background:var(--hl);border-radius:6px}
.when dt{color:var(--ink3)} .when dd{margin:0} .when b{font-weight:650}

.tw{overflow-x:auto} .tw.long{max-height:640px;overflow-y:auto} .tw.long thead th{position:sticky;top:0;background:var(--panel);z-index:1}
table{width:100%;border-collapse:collapse}
th{font-size:12px;font-weight:500;color:var(--ink3);text-align:left;padding:7px 10px 7px 0;border-bottom:1px solid var(--line2);white-space:nowrap}
td{padding:8px 10px 8px 0;border-bottom:1px solid var(--line);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
th:last-child,td:last-child{padding-right:0}
th.n,td.n{text-align:right}
td .nm{display:block;font-size:12px;color:var(--ink3);max-width:170px;overflow:hidden;text-overflow:ellipsis}
td.wr{white-space:normal}
.wl{display:block;font-size:12px;color:var(--up)}
.g{display:inline-block;min-width:20px;padding:0 4px;box-sizing:border-box;height:20px;line-height:20px;text-align:center;border-radius:4px;font-weight:700;font-size:12px;color:#fff}
.gA{background:var(--gA)} .gB{background:var(--gB)} .gC{background:var(--gC)}
.sp{display:block;width:96px;height:24px}
.bar{display:inline-block;width:40px;height:4px;border-radius:2px;background:var(--line);vertical-align:middle;margin-right:5px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ink3)}

.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:0 0 6px}
.kpi{padding:9px 16px}
.kpi+.kpi{border-left:1px solid var(--line)}
.kpi .k{font-size:12px;color:var(--ink3)} .kpi .v{font-size:19px;font-weight:650}

.chart{position:relative;padding:0 56px 10px 16px}
.chart svg{display:block;width:100%}
.chart .ax{display:flex;justify-content:space-between;font-size:11.5px;color:var(--ink3);margin-top:3px}
.chart .yl{position:absolute;right:8px;width:44px;font-size:11.5px;color:var(--ink3);background:var(--panel);padding:0 0 0 4px}
.lg{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2)}
.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.lg i.ln{height:2px;width:14px;border-radius:1px;vertical-align:3px}
.cap{font-size:12px;color:var(--ink3);margin:0 0 4px}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line2);border-radius:6px;padding:7px 10px;font-size:12.5px;line-height:1.6;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;white-space:nowrap;z-index:2}
.tip b{font-weight:650}
.hair{position:absolute;top:0;width:1px;background:var(--ink3);opacity:.5;display:none;pointer-events:none}

.rules{color:var(--ink2);font-size:13.5px}
.rules ol,.rules ul{margin:0 0 10px;padding-left:20px} .rules li{margin:2px 0}
.rules h3{font-size:13.5px;color:var(--ink);margin:4px 0 4px}
.note{font-size:12px;color:var(--ink3);margin:8px 0 0}
.cts{display:block;margin-top:3px;line-height:1.9}
.cd{margin-top:3px} .cd summary{list-style:none;cursor:pointer;margin:0;font-size:inherit;color:inherit;line-height:1.9} .cd summary::-webkit-details-marker{display:none} .cd summary::after{content:' 展开';font-size:11px;color:var(--ink3)} .cd[open] summary::after{content:' 收起'} .cd .cts{max-width:420px}
.ct{display:inline-block;font-size:11px;line-height:17px;padding:0 5px;border-radius:3px;margin-right:4px;border:1px solid;white-space:nowrap}
.ct.cSSS{color:var(--gA);border-color:var(--gA);font-weight:650} .ct.cA{color:var(--gB);border-color:var(--gB)} .ct.cB{color:var(--ink3);border-color:var(--line2)}
details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin-top:6px}
footer{color:var(--ink3);font-size:12px;margin-top:16px}

@media (max-width:900px){
  .stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .stats .tile:first-child{grid-column:1/-1}
  .grid2{grid-template-columns:minmax(0,1fr)}
  .grid2 .panel{margin-bottom:0}
}
@media (max-width:600px){
  .wrap{padding:16px 12px 40px}
  .verdict{font-size:28px}
  .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
  .kpi:nth-child(3){border-left:0}
  .kpi:nth-child(n+3){border-top:1px solid var(--line)}
  .hs{display:none}
  td .nm{display:none}
}
'''


# ---------- 小工具 ----------
def _pct(v, d=2):
    return '<span class="m">—</span>' if v is None else f'<span class="{_sign(v)}">{v:+.{d}f}%</span>'


def _spark(vals, w=120, h=26, color='var(--ink3)', zero=False, cls=''):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return ''
    lo, hi = min(v + ([0] if zero else [])), max(v + ([0] if zero else []))
    rng = (hi - lo) or 1
    pts = []
    for i, x in enumerate(vals):
        if x is None:
            continue
        pts.append(f'{i / (len(vals) - 1) * w:.1f},{h - 2 - (x - lo) / rng * (h - 4):.1f}')
    z = ''
    if zero:
        zy = h - 2 - (0 - lo) / rng * (h - 4)
        z = f'<line x1="0" x2="{w}" y1="{zy:.1f}" y2="{zy:.1f}" stroke="var(--line2)" stroke-dasharray="2 2" vector-effect="non-scaling-stroke"/>'
    c = f' class="{cls}"' if cls else ''
    return (f'<svg{c} viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true">{z}'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5" '
            f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>')


def _vix_word(v):
    return '平静' if v < 15 else '正常' if v < 20 else '偏紧张' if v < 25 else '紧张' if v < 30 else '恐慌'


# ---------- 各块 ----------
def _stats(ctx):
    m = ctx.get('mkt') or {}
    spy = m.get('spy') or []; vix = m.get('vix') or []; br = m.get('breadth') or []
    s2 = [x for x in spy if x is not None]
    chg = (s2[-1] / s2[-2] - 1) * 100 if len(s2) >= 2 else ctx.get('spy_chg', 0)
    ret120 = (s2[-1] / s2[0] - 1) * 100 if len(s2) >= 2 else None
    v = ctx['vix_price']
    b_now = br[-1] if br and br[-1] is not None else None
    b_5 = br[-6] if len(br) >= 6 and br[-6] is not None else None
    n_watch = len(ctx.get('watch') or [])
    n_base = ctx['funnel']['base']
    tiles = [
        ('标普500', f'{ctx["spy_price"]:,.0f}<small class="{_sign(chg)}">{chg:+.2f}%</small>',
         f'比 200 日均线 {ctx["spy_pct"]:+.1f}%' + (f'，今年 {ret120:+.1f}%' if ret120 is not None else ''),
         _spark(spy, color='var(--spx)')),
        ('VIX 恐慌指数', f'{v:.1f}<small class="m">{_vix_word(v)}</small>',
         '25 以上算恐慌，评级加分', _spark(vix, color='var(--up)' if v >= 25 else 'var(--ink3)')),
        ('站上 50 日均线', f'{b_now:.0f}%' if b_now is not None else '—',
         (f'5 天前 {b_5:.0f}%，' if b_5 is not None else '') + '成分股比例', _spark(br)),
        ('今天超卖', f'{n_base}<small class="m">只</small>', 'MACD 和 RSI 两条都满足的股票', ''),
        ('观察名单', f'{n_watch}<small class="m">只</small>', '明天低开、收涨就会出信号', ''),
    ]
    return '<section class="stats">' + ''.join(
        f'<div class="tile"><div class="k">{k}</div><div class="v n">{val}</div><div class="s">{s}</div>{sp}</div>'
        for k, val, s, sp in tiles) + '</section>'


def _tags(r):
    """命中的条件：先给 SSS / A / B 各几个，点开看全部"""
    t = r.get('cond_tags') or cond_tags(r.get('conds') or [])
    if not t:
        return ''
    cnt = {g: sum(x == g for x, _ in t) for g in ('SSS', 'A', 'B')}
    summ = ''.join(f'<span class="ct c{g}">{g} × {n}</span>' for g, n in cnt.items() if n)
    full = ''.join(f'<span class="ct c{g}">{escape(n)}</span>' for g, n in t)
    return f'<details class="cd"><summary>{summ}</summary><div class="cts">{full}</div></details>'


def _today(ctx):
    main = [r for r in ctx['results'] if r.get('new_rule')]
    sig, ent, ext = ctx['signal_day'], ctx['entry_day'], ctx['exit_day']
    n_watch = len(ctx.get('watch') or [])
    if not main:
        body = (f'<div class="verdict">今天不买</div>'
                f'<p>{_day(sig, False)}收盘后，没有股票符合主推条件。</p>'
                f'<p>这类机会多半跟在大跌后面，一出就是好几只；市场平稳时几周没有也正常。'
                + (f'观察名单里 {n_watch} 只离触发只差一天，明天收盘后再看。' if n_watch else '') + '</p>')
        return f'<section class="panel today"><div class="ph"><h2>今天买不买</h2><span class="r">{_day(sig)}收盘</span></div><div class="pb">{body}</div></section>'
    ab = sum((r.get('main_grade') or 'C') != 'C' for r in main)
    verdict = f'<div class="verdict buy">买 {ab} 只</div>' if ab else '<div class="verdict">只有 B 级，不买</div>'
    rows = ''
    for r in main:
        g = r.get('main_grade') or 'C'
        warn = []
        if r.get('open_lt_pl'):
            warn.append('开盘跌破前一天最低价')
        if r.get('earnings_warn'):
            warn.append(f'持有期内有财报（{escape(r["earnings_warn"])}），买一半')
        if (r.get('atr_pct') or 0) >= 4:
            warn.append(f'波动大（ATR {r["atr_pct"]:.1f}%），买一半')
        rows += (f'<tr><td class="wr"><b>{escape(r["ticker"])}</b> <span class="m">{escape(r["name"])}</span>'
                 + ''.join(f'<span class="wl">{w}</span>' for w in warn)
                 + _tags(r) + '</td>'
                 f'<td><span class="g g{g}">{GL[g]}</span></td><td class="n">{r.get("gap", 0):.1f}%</td>'
                 f'<td class="n hs">{r["rsi"]}</td><td class="n">${r["price"]}</td></tr>')
    body = (verdict
            + f'<p>{len(main)} 只符合条件，其中 SSS、A 级 {ab} 只；B 级历史上不赚钱，只看不买。同一级别里按命中的 SSS 条件多少排，越靠前越优先。</p>'
            + f'<dl class="when"><dt>买入</dt><dd><b>{_day(ent)} {_buy_time(ent)}</b> 美中，收盘前</dd>'
            + f'<dt>卖出</dt><dd><b>{_day(ext)}</b> 收盘前，拿满 {ctx["hold"]} 个交易日</dd></dl>'
            + f'<div class="tw"><table><thead><tr><th>股票</th><th>级别</th><th class="n">开盘跌幅</th>'
              f'<th class="n hs">RSI</th><th class="n">收盘价</th></tr></thead><tbody>{rows}</tbody></table></div>')
    return f'<section class="panel today"><div class="ph"><h2>今天买不买</h2><span class="r">{_day(sig)}收盘</span></div><div class="pb">{body}</div></section>'


def _watch(ctx):
    w = ctx.get('watch') or []
    head = '<div class="ph"><h2>观察名单：明天可能触发</h2><span class="r">按离触发价远近排序</span></div>'
    if not w:
        return f'<section class="panel">{head}<div class="pb"><p class="m">没有接近触发的股票。</p></div></section>'
    rows = ''
    for r in w[:12]:
        rows += (f'<tr><td><b>{escape(r["ticker"])}</b><span class="nm">{escape(r["name"])}</span></td>'
                 f'<td class="n">{r["rsi"]:.0f}</td><td class="n">${r["close"]:.2f}</td>'
                 f'<td class="n">${r["need"]:.2f} <span class="m">({r["need_pct"]:+.1f}%)</span></td></tr>')
    more = f'<p class="note">另有 {len(w) - 12} 只没列出。</p>' if len(w) > 12 else ''
    return (f'<section class="panel">{head}<div class="pb"><div class="tw"><table><thead><tr><th>股票</th>'
            f'<th class="n">RSI</th><th class="n">今天收盘</th><th class="n">明天收盘至少</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            f'<p class="note">明天要同时满足：开盘低于今天收盘，收盘不低于右列价格。满足就进主推，后天收盘买。</p>{more}</div></section>')


def _market_chart(ctx):
    m = ctx.get('mkt') or {}
    dates = m.get('dates') or []
    if len(dates) < 10:
        return ''
    spy = m['spy']; vix = m['vix']; br = m['breadth']
    pos = {d: i for i, d in enumerate(dates)}
    cnt = [{'A': 0, 'B': 0, 'C': 0} for _ in dates]
    for r in ctx.get('replay') or []:
        d = r['signal_day'].strftime('%Y-%m-%d')
        if d in pos:
            cnt[pos[d]][r['grade']] += 1
    n = len(dates); W = 1000
    sv = [x for x in spy if x is not None]; lo, hi = min(sv), max(sv); pad = (hi - lo) * 0.08 or 1
    lo -= pad; hi += pad
    H1 = 170
    pts = ' '.join(f'{(i + .5) / n * W:.1f},{H1 - (x - lo) / (hi - lo) * H1:.1f}' for i, x in enumerate(spy) if x is not None)
    grid = ''
    step = 10 ** len(str(int((hi - lo) / 4))) / 10 or 1
    step = next(s for s in (step, step * 2, step * 2.5, step * 5, step * 10) if (hi - lo) / s <= 5)
    ylabels = ''
    y = (int(lo / step) + 1) * step
    while y < hi:
        yy = H1 - (y - lo) / (hi - lo) * H1
        grid += f'<line x1="0" x2="{W}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--line)" vector-effect="non-scaling-stroke"/>'
        ylabels += f'<span class="yl n" style="top:{yy / H1 * 170 - 9:.0f}px">{y:,.0f}</span>'
        y += step
    mx = max(1, max(c['A'] + c['B'] + c['C'] for c in cnt))
    H2 = 70; bw = W / n
    bars = ''
    for i, c in enumerate(cnt):
        base = H2
        for g in 'CBA':
            if c[g]:
                h = c[g] / mx * (H2 - 4)
                bars += (f'<rect x="{i * bw + bw * .15:.1f}" y="{base - h:.1f}" width="{bw * .7:.1f}" height="{max(h - 1, 1):.1f}" '
                         f'fill="var(--g{g})"/>')
                base -= h
    marks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    ax = ''.join(f'<span>{int(dates[i][5:7])}月{int(dates[i][8:])}日</span>' for i in marks)
    data = json.dumps({'d': dates, 's': spy, 'v': vix, 'b': br, 'c': cnt}, separators=(',', ':'))
    total = sum(c['A'] + c['B'] + c['C'] for c in cnt)
    return f'''<section class="panel">
<div class="ph"><h2>大盘和主推信号</h2><span class="r">2026 年以来，共 {total} 个主推</span></div>
<div class="pb" style="padding-bottom:6px"><div class="lg"><span><i class="ln" style="background:var(--spx)"></i>标普500</span><span><i style="background:var(--gA)"></i>SSS 级</span><span><i style="background:var(--gB)"></i>A 级</span><span><i style="background:var(--gC)"></i>B 级</span></div></div>
<div class="chart" id="mc">
  {ylabels}
  <svg viewBox="0 0 {W} {H1}" preserveAspectRatio="none" style="height:170px">{grid}
    <polyline points="{pts}" fill="none" stroke="var(--spx)" stroke-width="1.75" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>
  <p class="cap" style="margin-top:8px">每天出现的主推个数（最多 {mx} 个）</p>
  <svg viewBox="0 0 {W} {H2}" preserveAspectRatio="none" style="height:70px">
    <line x1="0" x2="{W}" y1="{H2}" y2="{H2}" stroke="var(--line2)" vector-effect="non-scaling-stroke"/>{bars}</svg>
  <div class="ax n">{ax}</div>
  <div class="hair"></div><div class="tip"></div>
</div>
<script>
(function(){{
  var D={data}, el=document.getElementById('mc'), tip=el.querySelector('.tip'), hair=el.querySelector('.hair'), n=D.d.length;
  function fmt(x,d){{return x==null?'—':x.toLocaleString('en-US',{{minimumFractionDigits:d,maximumFractionDigits:d}})}}
  function show(e){{
    var r=el.getBoundingClientRect(), pad=16, w=r.width-pad-56, x=e.clientX-r.left-pad;
    if(x<0||x>w){{hide();return}}
    var i=Math.min(n-1,Math.max(0,Math.floor(x/w*n))), c=D.c[i], t=c.A+c.B+c.C, d=D.d[i];
    hair.style.display='block'; hair.style.left=(pad+(i+.5)/n*w)+'px'; hair.style.height=(el.offsetHeight-22)+'px';
    tip.innerHTML='<b>'+(+d.slice(5,7))+'月'+(+d.slice(8))+'日</b><br>标普 '+fmt(D.s[i],0)+'　VIX '+fmt(D.v[i],1)+
      '<br>站上 50 日线 '+fmt(D.b[i],0)+'%<br>主推 '+(t?t+' 个（SSS '+c.A+'，A '+c.B+'，B '+c.C+'）':'无');
    tip.style.display='block';
    var tx=pad+(i+.5)/n*w+12; if(tx+tip.offsetWidth>r.width) tx=pad+(i+.5)/n*w-12-tip.offsetWidth;
    tip.style.left=tx+'px'; tip.style.top='8px';
  }}
  function hide(){{tip.style.display='none';hair.style.display='none'}}
  el.addEventListener('mousemove',show); el.addEventListener('mouseleave',hide);
  el.addEventListener('touchstart',function(e){{show(e.touches[0])}},{{passive:true}});
  el.addEventListener('touchmove',function(e){{show(e.touches[0])}},{{passive:true}});
}})();
</script>
</section>'''


def _last20(ctx):
    hold = ctx['hold']
    rows = [r for r in (ctx.get('replay') or []) if r.get('status') in ('完成', '持有中')]
    head = f'<div class="ph"><h2>2026 年全部主推</h2><span class="r">共 {len(rows)} 笔，收益已减 0.10% 成本</span></div>'
    if not rows:
        return f'<section class="panel">{head}<div class="pb"><p class="m">最近没有主推。</p></div></section>'
    done = [r for r in rows if r['status'] == '完成']
    v = [r['net'] for r in done]
    avg = sum(v) / len(v) if v else None
    win = sum(x > 0 for x in v) / len(v) * 100 if v else None
    best = max(done, key=lambda r: r['net']) if done else None
    kp = [('已卖出', f'{len(done)} 笔', ''), ('赚钱比例', f'{win:.0f}%' if win is not None else '—', ''),
          ('平均每笔', f'{avg:+.2f}%' if avg is not None else '—', _sign(avg)),
          ('持有中', f'{len(rows) - len(done)} 笔', '')]
    kpis = '<div class="kpis">' + ''.join(f'<div class="kpi"><div class="k">{k}</div><div class="v n {c}">{val}</div></div>' for k, val, c in kp) + '</div>'
    body = ''
    for r in rows:
        g = r['grade']
        if r['status'] == '完成':
            st = f'<span class="{"up" if r["net"] > 0 else "down"}">{"赚" if r["net"] > 0 else "亏"}</span>'
        else:
            st = f'<span class="bar hs"><i style="width:{r["held"] / hold * 100:.0f}%"></i></span><span class="m n">{r["held"]}/{hold}天</span>'
        path = r.get('path') or []
        col = 'var(--up)' if (r.get('net') or 0) > 0 else 'var(--down)'
        body += (f'<tr><td class="wr"><b>{escape(r["ticker"])}</b><span class="nm">{escape(r["name"])}</span>'
                 + _tags(r) + '</td>'
                 f'<td><span class="g g{g}">{GL[g]}</span></td>'
                 f'<td class="n hs m">{r["signal_day"]:%m-%d}</td>'
                 f'<td class="n hs">{r["gap"]:.1f}%</td>'
                 f'<td class="n">{r["entry_day"]:%m-%d} {_close_ct(r["entry_day"])}</td>'
                 f'<td class="hs">{_spark(path + [None] * (hold + 1 - len(path)), 96, 24, col, zero=True, cls='sp')}</td>'
                 f'<td class="n">{_pct(r["net"])}</td><td>{st}</td></tr>')
    return (f'<section class="panel">{head}{kpis}<div class="pb"><div class="tw long"><table><thead><tr>'
            f'<th>股票</th><th>级别</th><th class="n hs">信号日</th><th class="n hs">开盘跌幅</th><th class="n">买入（美中）</th>'
            f'<th class="hs">10 天走势</th><th class="n">收益</th><th></th></tr></thead><tbody>{body}</tbody></table></div>'
            f'<p class="note">开盘跌幅：信号当天开盘价比前一天收盘低多少。买入按收盘价，时间写收盘时刻。走势线从买入当天开始，虚线是成本线。'
            + (f'最好一笔 {escape(best["ticker"])} {best["net"]:+.1f}%。' if best else '') + '</p></div></section>')


def _grades(ctx):
    done = [r for r in (ctx.get('replay') or []) if r.get('status') == '完成']
    rows = ''
    for g, v in GRADES.items():
        x = [r['net'] for r in done if r['grade'] == g]
        recent = (f'<td class="n">{len(x)}</td><td class="n">{sum(t > 0 for t in x) / len(x) * 100:.0f}%</td>'
                  f'<td class="n">{_pct(sum(x) / len(x))}</td>') if x else '<td class="n">0</td><td class="n m">—</td><td class="n m">—</td>'
        rows += (f'<tr><td><span class="g g{g}">{GL[g]}</span></td>{recent}'
                 f'<td class="n">{_pct(v[2])}</td><td class="n">{v[3]}%</td></tr>')
    return f'''<section class="panel"><div class="ph"><h2>分级成绩</h2><span class="r">2026 年 对比 2000–2025 年回测</span></div><div class="pb">
<div class="tw"><table><thead><tr><th>级别</th><th class="n">2026 笔数</th><th class="n">赚钱</th><th class="n">平均</th><th class="n">回测平均</th><th class="n">回测赚钱</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="note">SSS：开盘跌得深，同时市场恐慌。A：只占一条。B：都没有。开盘跌得深是指跌 1% 以上或跌破前一天最低价；市场恐慌是指 VIX 25 以上或当天有 5 个以上主推。回测用 2000–2025 年当时的成分股（含后来被剔除的）。</p>
</div></section>'''


def _conds(ctx):
    o = {'SSS': 0, 'A': 1, 'B': 2}
    items = sorted(COND_GRADES.items(), key=lambda kv: (o[kv[1][0]], -(kv[1][2] + kv[1][3])))
    tr = ''
    for k, (g, name, d1, d2) in items:
        note = '<span class="m">（已算进股票级别）</span>' if k in IN_GRADE else ''
        kind = '大盘' if k in MARKET_CONDS else '个股'
        tr += (f'<tr><td><span class="ct c{g}">{g}</span></td><td>{escape(name)}{note}</td><td class="m">{kind}</td>'
               f'<td class="n {_sign(d1)}">{d1:+.2f}%</td><td class="n {_sign(d2)}">{d2:+.2f}%</td></tr>')
    return f'''<section class="panel"><div class="ph"><h2>条件分级</h2><span class="r">主推里命中这个条件的，比没命中的每笔多赚多少</span></div><div class="pb">
<div class="tw long"><table><thead><tr><th>级别</th><th>条件</th><th>类型</th><th class="n">2000–2025</th><th class="n">2026</th></tr></thead>
<tbody>{tr}</tbody></table></div>
<p class="note">共 {CG_SUMMARY}。一共测过约 240 个条件，包括 RSI2、ConnorsRSI、IBS、KDJ、乖离率、布林带、K 线形态、量能、大盘宽度和 VIX 等网上常用的指标。
SSS：两段都多赚 0.5% 以上。A：两段都多赚，但幅度小。B：一段多赚，另一段不算差，只作参考。2026 年明显变差、笔数太少或以往和今年都不赚的没列出。日历类（星期、月份）不算条件。
RSI2、IBS 这类条件和主推很少同时出现（主推当天 RSI 在回升），所以不在表里。</p>
</div></section>'''


def _rules(ctx):
    return f'''<section class="panel"><div class="ph"><h2>规则</h2></div><div class="pb rules">
<h3>出信号（四条都要满足）</h3>
<ol><li>MACD 柱在零轴下方，连续 3 天抬高</li><li>前一天 RSI14 低于 30，当天回升</li>
<li>当天开盘低于前一天收盘</li><li>这只股票 5 天内第一次出信号</li></ol>
<h3>买卖</h3>
<ul><li>信号第二天收盘前买（美中 14:50–15:00；提前收盘日 11:50–12:00）</li>
<li>拿满 {ctx["hold"]} 个交易日，收盘前卖</li><li>一天出很多只时，先买 SSS，再买 A；单只仓位要小</li>
<li>持有期内有财报或波动特别大的，买一半</li></ul>
<details><summary>长期回测数字</summary>
<p class="note" style="font-size:12.5px">2000–2025 年平均每年约 {BT["per_year"]} 笔，平均每笔 +{BT["net"]:.2f}%，{BT["win"]}% 的交易赚钱，{BT["pos_years"]}% 的年份赚钱。
2026 年 {BT["n26"]} 笔，平均每笔 +{BT["net26"]:.2f}%，{BT["win26"]}% 赚钱。对比：随便买一只成分股拿 10 天，平均 +{BT["random"]:.2f}%。</p></details>
</div></section>'''


def render_report(ctx):
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
<header class="top"><h1>标普500 抄底信号</h1><p>数据截至 {_day(ctx["signal_day"])}收盘，{ctx["date_str"]} 更新</p></header>
{_stats(ctx)}
<div class="grid2">{_today(ctx)}{_watch(ctx)}</div>
{_market_chart(ctx)}
{_last20(ctx)}
<div class="grid2">{_grades(ctx)}{_rules(ctx)}</div>
{_conds(ctx)}
<footer>仅供自己研究，不是投资建议。红色是涨，绿色是跌。记录和图表按每天收盘数据、用同一套规则算出，按当前成分股。</footer>
</div>
</body>
</html>'''
