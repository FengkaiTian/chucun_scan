"""Shrinkage / few-parameter scoring vs SSS hard grading. Walk-forward Y=2008..2026, cluster by signal_day."""
import json, numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from sklearn.linear_model import Ridge
OUT = 'C:/Users/ft7b6/Desktop/美股/chucun_scan/analysis/rebuild/agents/stats/'
d = pd.read_parquet('C:/Users/ft7b6/Desktop/美股/chucun_scan/analysis/rebuild/data2000/signals_20y.parquet')
d = d[d.in_index & (d.vix < 30) & d.r5.notna()].reset_index(drop=True)
cols = list(d.columns); COND = cols[cols.index('base') + 1: cols.index('consec_down3') + 1]
combos = json.load(open('C:/Users/ft7b6/Desktop/美股/chucun_scan/validated_combos.json', encoding='utf-8'))
grades = np.array([c['grade'] for c in combos])
X = d[COND].fillna(False).astype(bool).values
H = np.column_stack([X[:, [COND.index(k) for k in c['conditions']]].all(1) for c in combos])
y = d.x5.values.astype(float); yr = d.year.values
day = pd.factorize(d.signal_day)[0]
rng = np.random.default_rng(0)
print(f'N={len(d)} days={day.max()+1} conds={len(COND)} combos={len(combos)} any-hit={H.any(1).mean():.2%}')
lines = []
def P(s=''):
    print(s); lines.append(s)

def cl_mean_se(yv, g):
    """mean and cluster-robust se for vector yv grouped by g"""
    n = len(yv)
    if n < 2: return np.nan, np.nan, n, 0
    m = yv.mean(); s = pd.Series(yv - m).groupby(g).sum().values; G = len(s)
    if G < 2: return m, np.nan, n, G
    return m, np.sqrt((s ** 2).sum() / n ** 2 * G / (G - 1)), n, G

def multi_cl(Hm, yv, g):
    """vectorised cluster mean/se for many boolean columns Hm (n x K)"""
    G = g.max() + 1; D = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))
    Hf = Hm.astype(float); S = D @ (Hf * yv[:, None]); C = D @ Hf
    n = C.sum(0); m = S.sum(0) / np.maximum(n, 1)
    R = S - C * m; Gc = (C > 0).sum(0)
    se2 = (R ** 2).sum(0) / np.maximum(n, 1) ** 2 * Gc / np.maximum(Gc - 1, 1)
    return m, se2, n, Gc

def eb(m, se2, w=None):
    ok = np.isfinite(m) & np.isfinite(se2) & (se2 > 0)
    mu = np.average(m[ok], weights=1 / se2[ok])
    eb.ratio = np.var(m[ok] - mu) / np.mean(se2[ok])
    tau2 = max(np.var(m[ok] - mu) - np.mean(se2[ok]), 1e-6)
    B = np.where(ok, tau2 / (tau2 + se2), 0.0)
    return mu, tau2, np.where(ok, mu + B * (m - mu), mu), B

# ---------------- scoring methods (fit on train mask) ----------------
MKT = ['vix', 'spx_dist_ma200', 'spx_ret5d', 'spx_ret20d', 'spx_rsi', 'vix_chg5', 'n_base_day_idx']
def fit_score(tr, te):
    out = {}
    yt = y[tr]; gt = pd.factorize(day[tr])[0]; base = yt.mean()
    # 1 EB combos
    Ht = H[tr]; m, se2, n, Gc = multi_cl(Ht, yt, gt); m = np.where(n >= 10, m, np.nan)
    mu, tau2, shr, B = eb(m, np.where(n >= 10, se2, np.nan))
    eff = shr - base  # shrunk excess vs base signal
    He = H[te].astype(float)
    E = np.where(H[te], eff[None, :], -np.inf)
    out['EB组合_max'] = np.where(H[te].any(1), E.max(1), 0.0)
    out['EB组合_均值'] = np.where(H[te].any(1), (He * eff).sum(1) / np.maximum(He.sum(1), 1), 0.0)
    raw = np.where(np.isfinite(m), m - base, -99)
    Er = np.where(H[te], raw[None, :], -np.inf)
    out['未收缩组合_max'] = np.where(H[te].any(1), Er.max(1), 0.0)
    # 2 additive per-condition EB
    Xt = X[tr]; mc, s2c, nc, _ = multi_cl(Xt, yt, gt); m0, s20, _, _ = multi_cl(~Xt, yt, gt)
    dlt = mc - m0; dse2 = s2c + s20; dlt = np.where(nc >= 20, dlt, np.nan)
    _, tau2c, shc, Bc = eb(dlt, np.where(nc >= 20, dse2, np.nan))
    shc = np.where(np.isfinite(dlt), shc, 0.0)
    out['单条件EB加法'] = X[te].astype(float) @ shc
    out['单条件未收缩加法'] = X[te].astype(float) @ np.nan_to_num(dlt)
    # 2b ridge: 37 bool + market vars standardized, heavy penalty, alpha by leave-year-out within train
    Z = np.column_stack([X.astype(float), d[MKT].values.astype(float)])
    Zt = Z[tr]; mu_z = np.nanmean(Zt, 0); sd_z = np.nanstd(Zt, 0) + 1e-9
    Zs = np.nan_to_num((Z - mu_z) / sd_z)
    yrs_t = np.unique(yr[tr]); best = None
    for a in [1e2, 1e3, 1e4, 1e5]:
        ic = []
        for v in yrs_t[-6:]:
            a_tr = tr & (yr != v) & (yr < v); a_te = tr & (yr == v)
            if a_tr.sum() < 500 or a_te.sum() < 30: continue
            p = Ridge(alpha=a).fit(Zs[a_tr], y[a_tr]).predict(Zs[a_te])
            ic.append(stats.spearmanr(p, y[a_te])[0])
        sc = np.nanmean(ic) if ic else -1
        if best is None or sc > best[0]: best = (sc, a)
    out['Ridge(37+大盘)'] = Ridge(alpha=best[1]).fit(Zs[tr], yt).predict(Zs[te])
    # 3 fixed market rule (pre-specified, no fit)
    out['规则:n_base_day_idx'] = d.n_base_day_idx.values[te].astype(float)
    out['SSS硬分级'] = H[te][:, grades == 'SSS'].any(1).astype(float)
    out['SSS+S硬分级'] = H[te][:, np.isin(grades, ['SSS', 'S'])].any(1).astype(float)
    return out, dict(tau2_combo=tau2, meanB_combo=np.nanmean(B[B > 0]), tau2_cond=tau2c, alpha=best[1],
                     top_cond=[COND[i] for i in np.argsort(-shc)[:3]])

def top_sel(score, frac=0.2):
    if np.unique(score).size <= 2:  # binary rule
        return score > 0 if score.max() > 0 else np.zeros(len(score), bool)
    r = pd.Series(score + rng.uniform(0, 1e-9, len(score))).rank(pct=True).values
    return r > 1 - frac

# ---------------- walk-forward ----------------
rows, meta = [], {}
for Y in range(2008, 2027):
    tr = yr < Y; te = yr == Y
    sc, mt = fit_score(tr, te); meta[Y] = mt
    yte = y[te]; ym = yte.mean(); gte = day[te]
    for k, s in sc.items():
        sel = top_sel(s)
        ic = stats.spearmanr(s, yte)[0] if np.unique(s).size > 2 else np.nan
        rows.append(dict(Y=Y, method=k, n_sel=sel.sum(), lift=(yte[sel].mean() - ym) if sel.sum() else np.nan, ic=ic,
                         dev=list(yte[sel] - ym), g=list(gte[sel]), wl=100*((d.r5.values[te][sel]>0).mean()-(d.r5.values[te]>0).mean()) if sel.sum() else np.nan))
R = pd.DataFrame(rows)

def summarize(R, years, label):
    P(f'\n== {label}: 选中信号 x5 相对当年全体均值的提升(点), 聚类t, 正年份比例, 符号检验p, 平均IC ==')
    P(f'{"方法":<22}{"n选中":>7}{"提升":>7}{"t":>6}{"正年":>7}{"p符号":>7}{"IC":>7}{"胜率差(年均)":>10}')
    res = []
    for k, g in R[R.Y.isin(years)].groupby('method', sort=False):
        dev = np.concatenate([np.array(v) for v in g.dev]); gg = np.concatenate([np.array(v) for v in g.g])
        m, se, n, _ = cl_mean_se(dev, gg) if len(dev) > 1 else (np.nan, np.nan, 0, 0)
        L = g.lift.dropna(); pos = (L > 0).sum(); nn = len(L)
        p = stats.binomtest(pos, nn, 0.5).pvalue if nn else np.nan
        res.append((k, n, m, m / se if se else np.nan, f'{pos}/{nn}', p, g.ic.mean()))
        P(f'{k:<22}{n:>7}{m:>7.2f}{m/se if se else np.nan:>6.2f}{pos:>4}/{nn:<2}{p:>7.3f}{g.ic.mean():>7.3f}{g.wl.mean():>10.1f}')
    return res
summarize(R, range(2008, 2027), 'Walk-forward 2008–2026 全部')
summarize(R, range(2008, 2020), '2008–2019 (SSS挑选时未见)')
summarize(R, [2025, 2026], '2025–2026')
P('\n每年提升(点) 按方法:')
piv = R.pivot(index='Y', columns='method', values='lift').round(1)
P(piv.to_string())
P('\nEB诊断(部分年份): ' + '; '.join(f'{Y}: tau_combo={np.sqrt(meta[Y]["tau2_combo"]):.2f} 平均B={meta[Y]["meanB_combo"]:.2f} tau_cond={np.sqrt(meta[Y]["tau2_cond"]):.2f} ridge_a={meta[Y]["alpha"]:.0e} top={meta[Y]["top_cond"]}' for Y in [2008, 2015, 2020, 2025]))

# ---------------- 3. market tiers by year ----------------
P('\n== 3. 大盘少参数分档：每年 高档-低档 的 x5 均值差 与 r5胜率差 ==')
nb = d.n_base_day_idx.values; s20 = d.spx_ret20d.values; s20c = -3 if np.nanmax(np.abs(s20)) > 1 else -0.03; vx = d.vix.values
tiers = {
    'n_base_day_idx≥5 vs <5': (nb >= 5, nb < 5),
    'n_base_day_idx≥3 vs 1-2': (nb >= 3, nb < 3),
    'n_base_day_idx≥10 vs <3': (nb >= 10, nb < 3),
    'spx_ret20d<-3% vs ≥-3%': (s20 < s20c, s20 >= s20c),
    'spx_ret20d<0 vs ≥0': (s20 < 0, s20 >= 0),
    'vix≥20 vs <20': (vx >= 20, vx < 20),
    'vix≥25 vs <18': (vx >= 25, vx < 18),
}
win = (d.r5.values > 0).astype(float)
tier_rows = []
for name, (hb, lb) in tiers.items():
    diffs, wdiffs = [], []
    for Y in range(2000, 2027):
        a = (yr == Y) & hb; b = (yr == Y) & lb
        if a.sum() < 10 or b.sum() < 10 or len(np.unique(day[a])) < 2: diffs.append(np.nan); wdiffs.append(np.nan); continue
        diffs.append(y[a].mean() - y[b].mean()); wdiffs.append(100 * (win[a].mean() - win[b].mean()))
    D = np.array(diffs); W = np.array(wdiffs); ok = np.isfinite(D)
    pos = (D[ok] > 0).sum(); wpos = (W[ok] > 0).sum(); nn = ok.sum()
    # pooled clustered difference
    m1, s1, n1, _ = cl_mean_se(y[hb], day[hb]); m0, s0, n0, _ = cl_mean_se(y[lb], day[lb])
    pre = (yr <= 2024); m1p, s1p, *_ = cl_mean_se(y[hb & pre], day[hb & pre]); m0p, s0p, *_ = cl_mean_se(y[lb & pre], day[lb & pre])
    post = yr >= 2025; m1q, s1q, *_ = cl_mean_se(y[hb & post], day[hb & post]); m0q, s0q, *_ = cl_mean_se(y[lb & post], day[lb & post])
    tier_rows.append(dict(rule=name, n_hi=int(hb.sum()), days_hi=len(np.unique(day[hb])), diff_all=m1 - m0, t_all=(m1 - m0) / np.hypot(s1, s0),
                          diff_00_24=m1p - m0p, t_00_24=(m1p - m0p) / np.hypot(s1p, s0p), diff_25_26=m1q - m0q,
                          yrs_pos_x5=f'{pos}/{nn}', p_sign_x5=stats.binomtest(int(pos), int(nn), .5).pvalue,
                          yrs_pos_win=f'{wpos}/{nn}', p_sign_win=stats.binomtest(int(wpos), int(nn), .5).pvalue,
                          median_year_diff=np.nanmedian(D)))
    P(f'{name}: 每年x5差 ' + ' '.join(f'{Y%100:02d}:{v:+.1f}' for Y, v in zip(range(2000, 2027), D) if np.isfinite(v)))
    P(f'   每年胜率差 ' + ' '.join(f'{Y%100:02d}:{v:+.0f}' for Y, v in zip(range(2000, 2027), W) if np.isfinite(v)))
T = pd.DataFrame(tier_rows); P(T.round(3).to_string(index=False)); T.to_csv(OUT + 'market_tiers.csv', index=False, encoding='utf-8-sig')

# ---------------- 4. fit 2000–2024 -> 2025–26 ----------------
P('\n== 4. 拟合 2000–2024 → 评估 2025–26 ==')
tr = yr <= 2024; te = yr >= 2025
sc, _ = fit_score(tr, te); yte = y[te]; gte = day[te]
for k, s in sc.items():
    sel = top_sel(s); m, se, n, _ = cl_mean_se(yte[sel] - yte.mean(), gte[sel])
    P(f'{k:<22} n={n:>4} 提升={m:+.2f} t={m/se:.2f}')
# overfit demo: best raw combos by 2000–2024 mean (n>=30)
m, se2, n, _ = multi_cl(H[tr], y[tr], pd.factorize(day[tr])[0]); mt, se2t, nt, _ = multi_cl(H[te], y[te], pd.factorize(day[te])[0])
ok = (n >= 30) & (nt >= 10)
mu, tau2, shr, B = eb(np.where(n >= 10, m, np.nan), np.where(n >= 10, se2, np.nan))
rho_raw = stats.spearmanr(m[ok], mt[ok])[0]; rho_t = stats.spearmanr((m / np.sqrt(se2))[ok], mt[ok])[0]
P(f'组合排名 训练(2000–24)均值 vs 2025–26均值 Spearman={rho_raw:.3f}; 用t值排={rho_t:.3f}; n组合={ok.sum()}')
P(f'组合间观测方差/平均抽样方差 = {eb.ratio:.2f} (<1 即组合差异完全可由噪音解释; 组合高度重叠使该比更偏低)')
P(f'收缩系数 B 中位数={np.nanmedian(B[n>=10]):.2f} (=保留多少原始超额), tau={np.sqrt(tau2):.2f}点')
for q in [10, 30]:
    i_raw = np.argsort(-np.where(ok, m, -99))[:q]; i_shr = np.argsort(-np.where(ok, shr, -99))[:q]
    P(f'Top{q}组合: 训练原始均值={np.mean(m[i_raw]):.2f} 收缩后预测={np.mean(shr[i_shr]):.2f} 实际2025–26={np.mean(mt[i_raw]):.2f}(原始排) / {np.mean(mt[i_shr]):.2f}(收缩排)  基准2025–26={y[te].mean():.2f}')

# ---------------- 5. power ----------------
P('\n== 5. 功效：按信号日聚类的最小可检测效应(MDE, 80%功效) ==')
sd_i = y.std(); m, se2, n, Gc = multi_cl(H, y, day); okk = n >= 30
deff = se2[okk] / (sd_i ** 2 / n[okk]); P(f'x5 个体sd={sd_i:.2f}点; 组合的设计效应 deff 中位数={np.median(deff):.2f} (IQR {np.percentile(deff,25):.2f}-{np.percentile(deff,75):.2f}); 每组合 n/信号日 中位数={np.median(n[okk]/Gc[okk]):.2f}')
dm = np.median(deff)
P(f'{"n信号":>7}{"n_eff":>7}{"SE":>6}{"MDE单检验":>10}{"MDE(334次Bonf)":>15}{"MDE 2年(÷3.2)":>15}')
for nn in [30, 50, 100, 200, 500, 1000, 3000]:
    se = sd_i * np.sqrt(dm / nn); P(f'{nn:>7}{nn/dm:>7.0f}{se:>6.2f}{2.80*se:>10.2f}{(stats.norm.ppf(1-0.025/334)+0.84)*se:>15.2f}{"":>15}')
# a 2-year OOS window: what n does a combo have in 2025–26?
P(f'2025–26 组合命中数中位数={np.median(nt[nt>0]):.0f}，SSS 合计命中={H[te][:, grades=="SSS"].any(1).sum()}')
# direction-consistency power: 年度符号检验
P('若真效应=+1点且每年组合仅~30个信号，每年正号概率≈Φ(1/(sd*sqrt(deff/30)))=' + f'{stats.norm.cdf(1/(sd_i*np.sqrt(dm/30))):.2f}')
R.drop(columns=['dev', 'g']).to_csv(OUT + 'walkforward_by_year.csv', index=False, encoding='utf-8-sig')
open(OUT + 'report.txt', 'w', encoding='utf-8').write('\n'.join(lines))
