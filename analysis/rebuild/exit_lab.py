"""卖出规则实验："买卖同源"（用入场时看的那些指标走坏来卖）、破底止损、金叉止盈、网格加减仓。

信号：2000–2026 标普500 基础信号（在指数内、VIX<30），同一股票 14 天内去重（和每日扫描一致）。
入场：信号次日收盘。卖出：触发当天收盘（A），或触发次日收盘（B，更保守）。最长持有 20 个交易日。
规则只在 2000–2019 年挑，2020–2026 年检验。分星级报告。
输出：out_exit/results.csv、out_exit/summary.txt
"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data2000')
OUT = os.path.join(HERE, 'out_exit'); os.makedirs(OUT, exist_ok=True)
MAXH = 20
lines = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)

# ── 信号 ──
s = pd.read_parquet(os.path.join(D, 'signals_20y.parquet'))
d = s[s.in_index & (s.vix < 30)].sort_values('signal_day').reset_index(drop=True)
keep, last = [], {}
for t, day in zip(d.ticker, d.signal_day):
    ok = t not in last or (day - last[t]).days > 14
    if ok: last[t] = day
    keep.append(ok)
d = d[keep].reset_index(drop=True)
vc = json.load(open(os.path.join(HERE, '..', '..', 'validated_combos.json'), encoding='utf-8'))
sss = np.zeros(len(d), bool)
for c in vc:
    if c['grade'] == 'SSS':
        sss |= np.logical_and.reduce([d[k].fillna(False).astype(bool).values for k in c['conditions']])
G = d.gap_down_close.fillna(False).values.astype(bool); Dj = d.doji_bottom.fillna(False).values.astype(bool)
d['tier'] = np.where(G, '★★★', np.where(Dj, '★★', np.where(sss, '★', '参考')))

# ── 行情和指标 ──
tick = sorted(d.ticker.unique())
px = {k: pd.read_parquet(os.path.join(D, f'ohlcv_{k}.parquet'))[tick] for k in ('close', 'high', 'low', 'volume')}
C, H, L, V = px['close'], px['high'], px['low'], px['volume']
ema12 = C.ewm(span=12, adjust=False).mean(); ema26 = C.ewm(span=26, adjust=False).mean()
dif = ema12 - ema26; hist = dif - dif.ewm(span=9, adjust=False).mean()
dl = C.diff()
rsi = 100 - 100 / (1 + dl.clip(lower=0).ewm(com=13, min_periods=14).mean() / (-dl.clip(upper=0)).ewm(com=13, min_periods=14).mean())
ma5 = C.rolling(5).mean(); ma20 = C.rolling(20).mean()
clv = ((C - L) - (H - C)) / (H - L).replace(0, np.nan)
cmf = (clv * V).rolling(20).sum() / V.rolling(20).sum()
dates = C.index
A = {k: v.values for k, v in dict(c=C, l=L, hist=hist, rsi=rsi, ma5=ma5, ma20=ma20, cmf=cmf).items()}
col = {t: i for i, t in enumerate(tick)}
pos = {dt: i for i, dt in enumerate(dates)}

# 每个信号的路径：第 0 天 = 入场日
paths = []
for r in d.itertuples():
    i0 = pos.get(r.signal_day)
    if i0 is None or i0 + 1 + MAXH + 1 >= len(dates):
        paths.append(None); continue
    j = col[r.ticker]; sl = slice(i0, i0 + MAXH + 3)   # 含信号日（第 -1 天）
    p = {k: A[k][sl, j] for k in A}
    if np.isnan(p['c'][:MAXH + 3]).any():
        paths.append(None); continue
    paths.append(p)
ok = np.array([p is not None for p in paths])
d = d[ok].reset_index(drop=True); paths = [p for p in paths if p is not None]
yr = d.signal_day.dt.year.values
say(f'信号 {len(d)} 条（14 天去重后）；星级分布 ' + ' '.join(f'{t}:{(d.tier == t).sum()}' for t in ('★★★', '★★', '★', '参考')))


# ── 卖出规则：给定路径，返回 (卖出日 k, 原因)；k 从 1 起（入场后第 k 天收盘判断） ──
def bad_count(p, k):
    """第 k 天（相对入场日，数组下标 k+1）有几个指标走坏"""
    i = k + 1
    return int(p['hist'][i] < p['hist'][i - 1]) + int(p['rsi'][i] < p['rsi'][i - 1]) + \
        int(p['c'][i] < p['ma5'][i]) + int(p['cmf'][i] < p['cmf'][i - 1]) + int(p['c'][i] < p['c'][i - 1])


def make_rule(fixed=None, macd_down=False, n_bad=None, n_bad_days=1, gold=False, break_low=False, ma20=False,
              min_hold=1, maxh=MAXH, tp=None, sl=None):
    def rule(p):
        entry = p['c'][1]; low0 = p['l'][0]; streak = 0
        for k in range(1, maxh + 1):
            i = k + 1
            c = p['c'][i]
            if fixed and k >= fixed: return k
            if sl is not None and c <= entry * (1 + sl / 100): return k
            if tp is not None and c >= entry * (1 + tp / 100): return k
            if break_low and c < low0: return k
            if k < min_hold: continue
            if gold and p['hist'][i] > 0: return k
            if ma20 and c > p['ma20'][i]: return k
            if macd_down and p['hist'][i] < p['hist'][i - 1]: return k
            if n_bad is not None:
                streak = streak + 1 if bad_count(p, k) >= n_bad else 0
                if streak >= n_bad_days: return k
        return maxh
    return rule


RULES = {
    '固定 5 天': make_rule(fixed=5),
    '固定 10 天': make_rule(fixed=10),
    '固定 20 天': make_rule(fixed=20),
    '站上 MA20 就卖（≤10 天）': make_rule(ma20=True, maxh=10),
    '同源：MACD 柱重新变差就卖': make_rule(macd_down=True),
    '同源：MACD 柱变差（至少持 2 天）': make_rule(macd_down=True, min_hold=2),
    '同源：MACD 金叉就卖': make_rule(gold=True),
    '同源：金叉 或 柱变差（至少 2 天）': make_rule(gold=True, macd_down=True, min_hold=2),
    '破信号日最低价就卖': make_rule(break_low=True),
    '金叉止盈 + 破底止损': make_rule(gold=True, break_low=True),
    '金叉止盈 + 破底止损（≤10 天）': make_rule(gold=True, break_low=True, maxh=10),
}
for n in (2, 3, 4, 5):
    RULES[f'5 个指标走坏≥{n} 个就卖'] = make_rule(n_bad=n)
    RULES[f'走坏≥{n} 个、连续 2 天才卖'] = make_rule(n_bad=n, n_bad_days=2)
    RULES[f'走坏≥{n} 个 + 金叉止盈'] = make_rule(n_bad=n, gold=True)
    RULES[f'走坏≥{n} 个 + 金叉止盈 + 破底止损'] = make_rule(n_bad=n, gold=True, break_low=True)


def run_rule(rule, lag):
    """lag=0 触发当天收盘卖；lag=1 次日收盘卖"""
    ret = np.empty(len(paths)); days = np.empty(len(paths))
    for n, p in enumerate(paths):
        k = min(rule(p) + lag, MAXH + 1)
        ret[n] = (p['c'][k + 1] / p['c'][1] - 1) * 100; days[n] = k
    return ret, days


# ── 网格：跌了加仓，涨了分批卖 ──
def grid(p, add_at=(-3, -6), tp_steps=(3, 6), maxh=10, stop=None):
    """按金额记账：每份 1 元。初始买 1 份；收盘价跌到 入场价×(1+a%) 再买 1 份；
    收盘价涨到 持仓均价×(1+t%) 时，卖掉当前持仓的 1/剩余批数；止损/到期全部卖出。
    返回 (盈亏占"预留资金"的 %, 持有天数, 盈亏占"实际最多投入"的 %)。预留资金 = 1 + 加仓次数上限。"""
    entry = p['c'][1]; shares = 1 / entry; cost = 1.0; pnl = 0.0; added = set(); steps_done = 0
    max_cost = 1.0; reserve = 1 + len(add_at)
    def out(k):
        return pnl / reserve * 100, k, pnl / max_cost * 100
    for k in range(1, maxh + 1):
        c = p['c'][k + 1]
        if shares > 0:
            if stop is not None and c <= entry * (1 + stop / 100):
                pnl += shares * c - cost; shares = 0; cost = 0; return out(k)
            for a in add_at:
                if a not in added and c <= entry * (1 + a / 100):
                    shares += 1 / c; cost += 1; added.add(a); max_cost = max(max_cost, cost)
            while steps_done < len(tp_steps) and shares > 0 and c >= cost / shares * (1 + tp_steps[steps_done] / 100):
                f = 1 / (len(tp_steps) - steps_done)
                pnl += f * (shares * c - cost); shares *= (1 - f); cost *= (1 - f); steps_done += 1
            if steps_done >= len(tp_steps):
                return out(k)
    c = p['c'][maxh + 1]
    pnl += shares * c - cost
    return out(maxh)


GRIDS = {
    '网格：跌 3%/6% 各加 1 份，涨 3%/6% 分两批卖（≤10 天）': dict(),
    '网格：跌 4%/8% 加仓，涨 4%/8% 卖（≤10 天）': dict(add_at=(-4, -8), tp_steps=(4, 8)),
    '网格：跌 3%/6% 加仓，涨 3%/6% 卖，跌 10% 全止损': dict(stop=-10),
    '网格：跌 3%/6% 加仓，涨 3%/6% 卖（≤20 天）': dict(maxh=20),
    '只加仓不分批：跌 3%/6% 加，涨 5% 全卖（≤10 天）': dict(tp_steps=(5,)),
    '只分批不加仓：涨 3%/6% 分两批卖（≤10 天）': dict(add_at=()),
    '不加仓：涨 5% 全卖（≤10 天）': dict(add_at=(), tp_steps=(5,)),
    '不加仓：涨 3% 全卖（≤10 天）': dict(add_at=(), tp_steps=(3,)),
}

# ── 汇总 ──
periods = [('2000–19 挑规则', yr <= 2019), ('2020–26 检验', yr >= 2020)]
tiers = [('全部', np.ones(len(d), bool)), ('★★★', (d.tier == '★★★').values), ('★★', (d.tier == '★★').values),
         ('★', (d.tier == '★').values), ('带星合计', (d.tier != '参考').values)]
rows = []


def add_rows(name, ret, days, extra=None):
    for pl, pm in periods:
        for tl, tm in tiers:
            m = pm & tm
            if m.sum() < 10: continue
            x, dd = ret[m], days[m]
            rows.append(dict(rule=name, period=pl, tier=tl, n=m.sum(), avg=x.mean(), win=(x > 0).mean() * 100,
                             days=dd.mean(), per_day=x.mean() / dd.mean(), p5=np.percentile(x, 5), worst=x.min(),
                             **(extra or {})))


for name, rule in RULES.items():
    for lag, ln in ((0, '当天收盘卖'), (1, '次日收盘卖')):
        if lag == 1 and name.startswith('固定'): continue
        ret, days = run_rule(rule, lag)
        add_rows(name if lag == 0 else name + '（次日卖）', ret, days)
for name, kw in GRIDS.items():
    res = np.array([grid(p, **kw) for p in paths])
    add_rows(name, res[:, 0], res[:, 1])                       # 按预留资金
    if kw.get('add_at', (-3,)):
        add_rows(name + '［按实际投入算］', res[:, 2], res[:, 1])
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, 'results.csv'), index=False, encoding='utf-8-sig')

# 2000–19 上按"每持有日收益"选，2020–26 看
def table(tier, sort='per_day', top=12):
    tr = R[(R.period == '2000–19 挑规则') & (R.tier == tier)].set_index('rule')
    te = R[(R.period == '2020–26 检验') & (R.tier == tier)].set_index('rule')
    j = tr.join(te, lsuffix='_tr', rsuffix='_te').sort_values(f'{sort}_tr', ascending=False)
    say(f'\n## {tier}：按 2000–19 的「{"每持有日收益" if sort == "per_day" else "每笔平均收益"}」排序（前 {top} + 固定 5 天对照）')
    say(f'  {"规则":<34}| 00–19 每笔 胜率 天数 每日 最差5% | 20–26 每笔 胜率 天数 每日 最差5%')
    show = list(j.index[:top])
    for ref in ('固定 5 天', '固定 10 天'):
        if ref not in show: show.append(ref)
    for rule in show:
        r = j.loc[rule]
        say(f'  {rule[:34]:<34}| {r.avg_tr:+5.2f}% {r.win_tr:4.0f}% {r.days_tr:4.1f} {r.per_day_tr:+.3f} {r.p5_tr:+6.1f}% '
            f'| {r.avg_te:+5.2f}% {r.win_te:4.0f}% {r.days_te:4.1f} {r.per_day_te:+.3f} {r.p5_te:+6.1f}%')

for tier in ('全部', '带星合计', '★★★', '★★', '★'):
    table(tier, 'per_day')
for tier in ('全部', '带星合计'):
    table(tier, 'avg')
for tier in ('全部', '带星合计'):
    say(f'\n## 网格类（{tier}；收益按预留资金算，［按实际投入算］的是只算真正花出去的钱）')
    for g in sorted(R.rule.unique()):
        if not (g.startswith('网格') or g.startswith('只') or g.startswith('不加仓') or g.startswith('固定')):
            continue
        a = R[(R.rule == g) & (R.tier == tier)].set_index('period')
        say(f'  {g[:52]:<52} 00–19 每笔 {a.loc["2000–19 挑规则","avg"]:+.2f}% 胜率 {a.loc["2000–19 挑规则","win"]:.0f}% 最差5% {a.loc["2000–19 挑规则","p5"]:+.1f}% '
            f'| 20–26 每笔 {a.loc["2020–26 检验","avg"]:+.2f}% 胜率 {a.loc["2020–26 检验","win"]:.0f}% 最差5% {a.loc["2020–26 检验","p5"]:+.1f}%')
open(os.path.join(OUT, 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
