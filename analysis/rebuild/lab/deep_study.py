"""锁定的两条规则复核（2026-11-18，按专家建议）。

规则（冻结，不再调参）：
  R1 纯深跌：收盘 < MA200 × 0.75
  R2 深跌+缩量：R1 且 当日成交量 < 此前 20 日（不含当日）均量 × 0.70
  信号日收盘决定，次日收盘买入，持有 20 个交易日，同一股票持仓不重叠。
数据：修正后的时点成分股（membership_v2，按入指日期截断），基准为含分红的 SPY 总回报，每笔来回成本 0.10%（超额也扣）。
时期：2000–2019、2020–2024；2025–2026 已经看过，只作诊断。

输出 out/deep_study.txt：
  1. 两条规则逐期成绩，以及"旧数据/旧口径"对照（看修正带来多少变化）
  2. 缩量的增量：每笔 R2 交易配一笔"同日、同行业、深跌幅度相近（±5 点）、事前 60 日贝塔相近（±0.3）、但没缩量"的对照，
     同进同出，报告匹配覆盖率、配对收益差、按月分块 bootstrap 的 95% 区间
  3. 两条日净值：固定仓位（每笔 5% 初始资金，最多 20 笔）买候选股 vs 同日同金额买 SPY；扣费后净收益、最大回撤、资金占用，
     差值按连续 60 个交易日分块 bootstrap 求 95% 区间
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_common import Lab, _trades, _cluster_t, COST, DATA

HOLD, LAG = 20, 1
SIDE = COST / 2 / 100    # 每边 0.05%（COST 的单位是 %，这里换成小数）
rng = np.random.default_rng(20261118)
lines = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)

PER = [('2000–2019', 2000, 2019), ('2020–2024', 2020, 2024), ('2025–2026（已看过，仅诊断）', 2025, 2026)]


def rules(lab):
    C, V = lab.C, lab.V
    ma200 = lab.ma(200)
    dist = C / ma200 - 1
    vprior = pd.DataFrame(V).shift(1).rolling(20).mean().values
    deep = np.isfinite(ma200) & (C < 0.75 * ma200)
    shrink = (V > 0) & np.isfinite(vprior) & (V < 0.70 * vprior)
    return deep, deep & shrink, shrink, dist


def trade_table(lab, sig):
    s = np.ascontiguousarray(sig & lab.member)
    t, j, r = _trades(s, lab.C, HOLD, LAG)
    sp = (lab.spx[t + LAG + HOLD] / lab.spx[t + LAG] - 1) * 100
    return pd.DataFrame({'t': t, 'j': j, 'r': r, 'sp': sp, 'year': lab.year[t]})


def period_stats(T, label):
    say(f'  {label}')
    for name, lo, hi in PER:
        x = T[(T.year >= lo) & (T.year <= hi)]
        if len(x) < 5:
            say(f'    {name}: {len(x)} 笔'); continue
        net = x.r - COST; ex = x.r - COST - x.sp
        say(f'    {name}: {len(x)} 笔（{x.t.nunique()} 个信号日）| 胜率 {(net > 0).mean()*100:.1f}% | 每笔净收益 {net.mean():+.2f}% '
            f'(t={_cluster_t(net.values, x.t.values):.1f}) | 每笔净超额 {ex.mean():+.2f}% (t={_cluster_t(ex.values, x.t.values):.1f})')


def block_ci_by_month(vals, months, reps=2000):
    df = pd.DataFrame({'v': vals, 'm': months})
    g = df.groupby('m').v.agg(['sum', 'count'])
    S, N = g['sum'].values, g['count'].values
    k = len(S)
    if k < 5:
        return np.nan, np.nan
    idx = rng.integers(0, k, size=(reps, k))
    boot = S[idx].sum(1) / N[idx].sum(1)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    say('# 锁定两条规则的复核\n')
    say('规则：R1 收盘<MA200×0.75；R2 再加 当日量<此前20日均量×0.70。次日收盘买，持有 20 个交易日，不重叠，每笔来回 0.10%。\n')

    # ── 1. 成绩：新数据新口径 vs 旧 ──
    say('## 1. 两条规则逐期成绩')
    old = Lab()                                          # 旧名单 + 标普价格指数
    new = Lab('membership_v2.csv.gz', bench='spy_tr')    # 修正名单 + 含分红 SPY
    for tag, lab in (('旧（原名单、^GSPC 价格指数）', old), ('新（修正名单、SPY 含分红）', new)):
        d1, d2, _, _ = rules(lab)
        say(f'\n### {tag}')
        period_stats(trade_table(lab, d1), 'R1 纯深跌')
        period_stats(trade_table(lab, d2), 'R2 深跌+缩量')
    lab = new
    deep, deepshrink, shrink, dist = rules(lab)
    T1, T2 = trade_table(lab, deep), trade_table(lab, deepshrink)
    say(f'\n成分股修正的影响：成分"股票-日"从 {old.member.sum():,} 降到 {new.member.sum():,}（−{(1 - new.member.sum()/old.member.sum())*100:.1f}%）')

    # ── 2. 缩量的增量：同日匹配对照 ──
    say('\n## 2. 缩量到底有没有用：同日匹配对照')
    sec = pd.read_csv(os.path.join(DATA, 'sectors_current.csv'), index_col=0).iloc[:, 0].to_dict()
    sector = np.array([sec.get(t, '未知') for t in lab.tickers])
    rs = pd.DataFrame(lab.C).pct_change().values
    rm = pd.Series(lab.spx).pct_change().values[:, None]
    W = 60
    mxy = pd.DataFrame(rs * rm).rolling(W, min_periods=50).mean().values
    mx = pd.DataFrame(rs).rolling(W, min_periods=50).mean().values
    my = pd.Series(rm[:, 0]).rolling(W, min_periods=50).mean().values[:, None]
    vy = pd.Series(rm[:, 0]).rolling(W, min_periods=50).var(ddof=0).values[:, None]
    beta = (mxy - mx * my) / vy
    C = lab.C
    pool = lab.member & deep & ~shrink

    def match(strict_sector):
        rows = []
        for tr in T2.itertuples():
            t, j = tr.t, tr.j
            e, x = t + LAG, t + LAG + HOLD
            cand = np.where(pool[t] & np.isfinite(C[e]) & np.isfinite(C[x]) & np.isfinite(beta[t]))[0]
            cand = cand[cand != j]
            if strict_sector:
                cand = cand[sector[cand] == sector[j]]
            if not np.isfinite(beta[t, j]):
                rows.append(None); continue
            dd = np.abs(dist[t, cand] - dist[t, j]); db = np.abs(beta[t, cand] - beta[t, j])
            ok = (dd <= 0.05) & (db <= 0.30)
            cand, dd, db = cand[ok], dd[ok], db[ok]
            if len(cand) == 0:
                rows.append(None); continue
            k = cand[np.argmin((dd / 0.05) ** 2 + (db / 0.30) ** 2)]
            rc = (C[x, k] / C[e, k] - 1) * 100
            rows.append((tr.r, rc, tr.year, lab.dates[t].strftime('%Y-%m')))
        return rows

    for strict, lab_s in ((True, '同日 + 同行业 + 深跌幅度±5点 + 事前贝塔±0.3'), (False, '同日 + 深跌幅度±5点 + 事前贝塔±0.3（不限行业）')):
        rows = match(strict)
        got = [r for r in rows if r is not None]
        say(f'\n### 对照条件：{lab_s}')
        say(f'  匹配覆盖率：{len(got)}/{len(rows)} = {len(got)/len(rows)*100:.0f}%')
        P = pd.DataFrame(got, columns=['r', 'rc', 'year', 'month'])
        P['diff'] = P.r - P.rc
        for name, lo, hi in PER:
            x = P[(P.year >= lo) & (P.year <= hi)]
            tot = sum(1 for r in rows if r is not None or True)
            if len(x) < 10:
                say(f'  {name}: {len(x)} 对'); continue
            lo_ci, hi_ci = block_ci_by_month(x['diff'].values, x.month.values)
            say(f'  {name}: {len(x)} 对 | 缩量组每笔 {x.r.mean():+.2f}% vs 对照 {x.rc.mean():+.2f}% | 差 {x["diff"].mean():+.2f} 点 '
                f'（按月分块 95% 区间 {lo_ci:+.2f} ~ {hi_ci:+.2f}）| 缩量组胜过对照的比例 {(x["diff"] > 0).mean()*100:.0f}%')
    say('  说明：行业用的是当前 GICS 行业（已退市股票标"未知"，只能和同样"未知"的配对）；成本两边相同，已相互抵消。')

    # ── 3. 日净值 ──
    say('\n## 3. 日净值：固定仓位买候选股 vs 同日同金额买 SPY')
    say('设定：初始资金 1；每笔投入 5% 初始资金（不复利），最多同时 20 笔；同一天候选太多时，先买跌得最深的；'
        '股票每边扣 0.05%，SPY 不扣费（对策略更严格）；持仓期间停牌/退市取最后有效价。')
    years_idx = lab.year
    T = len(lab.dates)
    spx = lab.spx

    def nav(sig):
        s = sig & lab.member
        pnl_s = np.zeros(T); pnl_b = np.zeros(T); used = np.zeros(T)
        open_until = {}            # j -> exit idx
        n_open = np.zeros(T, int)
        for t in range(T - LAG - HOLD):
            if years_idx[t] < 2000:
                continue
            js = np.where(s[t])[0]
            if len(js) == 0:
                continue
            js = js[np.argsort(dist[t, js])]
            e, x = t + LAG, t + LAG + HOLD
            for j in js:
                if n_open[e] >= 20:
                    break
                if open_until.get(j, -1) >= e or not np.isfinite(C[e, j]):
                    continue
                open_until[j] = x
                n_open[e:x] += 1
                a = 0.05
                path = pd.Series(C[e:x + 1, j]).ffill().values
                v = a * path / path[0]
                pnl_s[e + 1:x + 1] += np.diff(v)
                pnl_s[e] -= a * SIDE; pnl_s[x] -= v[-1] * SIDE
                b = a * spx[e:x + 1] / spx[e]
                pnl_b[e + 1:x + 1] += np.diff(b)
                used[e:x] += a
        return pnl_s, pnl_b, used

    def mdd(p):
        nav_ = 1 + np.cumsum(p)
        return (nav_ / np.maximum.accumulate(nav_) - 1).min() * 100

    def boot_block(dif, L=60, reps=2000):
        n = len(dif); k = int(np.ceil(n / L))
        starts = rng.integers(0, n - L, size=(reps, k))
        idx = (starts[:, :, None] + np.arange(L)).reshape(reps, -1)[:, :n]
        m = dif[idx].mean(1) * 252 * 100
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    for rname, sig in (('R1 纯深跌', deep), ('R2 深跌+缩量', deepshrink)):
        ps, pb, used = nav(sig)
        say(f'\n### {rname}')
        say('  | 时期 | 策略 年化净收益 | 同日SPY 年化 | 差（95% 区间） | 策略最大回撤 | 同日SPY最大回撤 | 平均资金占用 |')
        say('  |---|---|---|---|---|---|---|')
        for name, lo, hi in PER + [('2000–2026 全期', 2000, 2026)]:
            m = (years_idx >= lo) & (years_idx <= hi)
            a, b = ps[m], pb[m]
            yrs = m.sum() / 252
            ci = boot_block(a - b)
            say(f'  | {name} | {a.sum()/yrs*100:+.1f}% | {b.sum()/yrs*100:+.1f}% | {(a.sum()-b.sum())/yrs*100:+.1f}%（{ci[0]:+.1f} ~ {ci[1]:+.1f}） '
                f'| {mdd(a):.1f}% | {mdd(b):.1f}% | {used[m].mean()*100:.0f}% |')
    m_all = years_idx >= 2000
    spy_bh = (spx[m_all][-1] / spx[m_all][0]) ** (252 / m_all.sum()) - 1
    say(f'\n参考：2000–2026 全仓持有 SPY（含分红）年化 {spy_bh*100:.1f}%。上表"年化"是相对初始资金的简单年化（不复利）。')

    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'deep_study.txt'), 'w', encoding='utf-8').write('\n'.join(lines))


if __name__ == '__main__':
    main()
