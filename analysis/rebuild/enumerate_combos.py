"""20 年组合穷举：37 个原条件 + 8 个市场/个股环境条件，1–4 重组合。
训练 2006–2020（只在这里挑组合），验证 2021–2026（只看一次）。
同时：用 2006–2019 检验现有 validated_combos.json（那 334 个组合是用 2020–2025 挑的，2006–2019 它从没见过）。
输出 out/：combos_all.parquet、summary.txt
"""
import os, sys, json, time, itertools
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_data import DATA, COND_NAMES

OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
TRAIN_END = pd.Timestamp('2020-12-31')
MIN_N_TRAIN = 50          # 训练期至少 50 条信号才算一个组合
MIN_N_EXPAND = 40         # 三重组合至少 40 条才继续扩到四重
HOLD = (1, 3, 5, 10)
lines = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)

s = pd.read_parquet(os.path.join(DATA, 'signals_20y.parquet'))
d = s[s.in_index & (s.vix < 30) & s.r10.notna() & s.b10.notna()].reset_index(drop=True)   # 和每日扫描一致：VIX≥30 不出信号

# 额外的环境条件（信号日收盘时都能知道）
EXTRA = {
    'vix_gt20': d.vix > 20, 'vix_gt25': d.vix > 25,
    'spx_below_ma200': d.spx_dist_ma200 < 0, 'spx_ret5d_lt_m2': d.spx_ret5d < -2,
    'n_base_ge5': d.n_base_day_idx >= 5, 'atr_ge4': d.atr_pct >= 4,
    'dist_high_lt_m30': d.dist_high252 < -30, 'ret20d_lt_m10': d.ret20d < -10,
}
NAMES = COND_NAMES + list(EXTRA)
C = np.column_stack([d[n].values.astype(bool) for n in COND_NAMES] + [v.values for v in EXTRA.values()])
K = len(NAMES)

tr = (d.signal_day <= TRAIN_END).values
te = ~tr
day_id = pd.factorize(d.signal_day)[0]
R = {k: d[f'r{k}'].values for k in HOLD}
X5 = d.x5.values
UP = {k: (R[k] > 0).astype(float) for k in HOLD}
base_up = {p: {k: UP[k][m].mean() for k in HOLD} for p, m in (('tr', tr), ('te', te))}
say(f'样本：{len(d)} 条基础信号（在指数内、VIX<30），训练 {tr.sum()} 条（2006–2020），验证 {te.sum()} 条（2021–2026）')
say(f'条件 {K} 个（原 37 + 环境 8）')
say('基础信号上涨比例  训练 ' + ' '.join(f'T+{k} {base_up["tr"][k]*100:.1f}%' for k in HOLD)
    + '  |  验证 ' + ' '.join(f'T+{k} {base_up["te"][k]*100:.1f}%' for k in HOLD))


def cluster_t(y, days):
    """按信号日聚类的均值 t 统计量"""
    n = len(y)
    if n < 2: return np.nan, 0
    S = np.bincount(days, y); cnt = np.bincount(days)
    g = cnt > 0; G = int(g.sum())
    if G < 2: return np.nan, G
    mu = S.sum() / n
    res = S[g] - cnt[g] * mu
    se = np.sqrt((res ** 2).sum() * G / (G - 1)) / n
    return (mu / se if se > 0 else np.nan), G


def stats(mask, part, up=UP, x5=X5, pb=None):
    m = mask & part
    n = int(m.sum())
    if n == 0: return None
    out = {'n': n}
    for k in HOLD:
        out[f'wr{k}'] = up[k][m].mean() * 100
    out['r5'] = R[5][m].mean(); out['x5'] = x5[m].mean()
    out['t_x5'], out['days'] = cluster_t(x5[m], day_id[m])
    out['t_up5'], _ = cluster_t(up[5][m] - pb, day_id[m])
    return out


def run(up, x5, label=''):
    """在训练期枚举全部组合；返回每个组合的训练期统计。"""
    pb_tr = up[5][tr].mean()
    res = []
    keep3 = []
    t0 = time.time()
    for size in (1, 2, 3, 4):
        if size < 4:
            it = itertools.combinations(range(K), size)
        else:   # 四重组合只从样本够的三重组合往上扩
            it = (c3 + (j,) for c3 in keep3 for j in range(c3[-1] + 1, K))
        for combo in it:
            m = C[:, combo].all(axis=1)
            ntr = int((m & tr).sum())
            if size == 3 and ntr >= MIN_N_EXPAND: keep3.append(combo)
            if ntr < MIN_N_TRAIN: continue
            st = stats(m, tr, up, x5, pb_tr)
            st['combo'] = combo; st['size'] = size
            res.append(st)
        if label == '':
            say(f'  {size} 重组合完成，累计 {len(res)} 个样本够的组合，{time.time() - t0:.0f}s')
    return pd.DataFrame(res)


def union_eval(combos, part):
    """一组组合在某个时期的「并集」表现：每条信号只算一次（和每日扫描一样，命中任一组合就推荐）"""
    if len(combos) == 0: return None
    m = np.zeros(len(d), bool)
    for c in combos: m |= C[:, list(c)].all(axis=1)
    pb = UP[5][part].mean()
    return stats(m, part, pb=pb)


def fmt(st, base):
    if st is None: return 'n=0'
    return (f"n={st['n']:>5} 天数={st['days']:>4}  上涨 T+5 {st['wr5']:.1f}%（基础 {base*100:.1f}%）"
            f"  T+10 {st['wr10']:.1f}%  超额 x5 {st['x5']:+.2f}%  t={st['t_x5']:.2f}")


# ── 1. 枚举 ─────────────────────────────────────────────
say('\n## 1. 穷举（训练期 2006–2020）')
A = run(UP, X5)
say(f'共试了 {len(A)} 个样本够（n≥{MIN_N_TRAIN}）的组合')

# 每个组合补上验证期统计
pb_te = UP[5][te].mean()
te_rows = []
for combo in A.combo:
    st = stats(C[:, list(combo)].all(axis=1), te, pb=pb_te) or {}
    te_rows.append({f'te_{k}': v for k, v in st.items()})
A = pd.concat([A, pd.DataFrame(te_rows)], axis=1)
A['name'] = A.combo.map(lambda c: ' + '.join(NAMES[i] for i in c))

# ── 2. 复刻原来的挑法：T+1/3/5/10 都跑赢基础信号，再按 T+5 胜率分级 ──
say('\n## 2. 复刻原来的挑法（训练期挑，验证期看）')
beat = np.all([A[f'wr{k}'] > base_up['tr'][k] * 100 for k in HOLD], axis=0)
V = A[beat & (A.wr5 >= 65)].copy()
say(f'训练期「四个周期都跑赢 + T+5 胜率≥65%」的组合：{len(V)} 个（占 {len(V)/len(A)*100:.1f}%）')
say(f'  这些组合训练期平均 T+5 胜率 {V.wr5.mean():.1f}% → 验证期 {V.te_wr5.mean():.1f}%（验证期基础信号 {pb_te*100:.1f}%）')
say('  训练期 并集：' + fmt(union_eval(V.combo, tr), UP[5][tr].mean()))
say('  验证期 并集：' + fmt(union_eval(V.combo, te), pb_te))
say('  验证期 基础：' + fmt(stats(np.ones(len(d), bool), te, pb=pb_te), pb_te))
for g, lo, hi in (('B', 65, 70), ('A', 70, 80), ('S', 80, 101)):
    sub = V[(V.wr5 >= lo) & (V.wr5 < hi)]
    if len(sub):
        say(f'  {g} 级（训练 {lo}–{hi}%）{len(sub)} 个 → 验证期并集 ' + fmt(union_eval(sub.combo, te), pb_te))

# ── 3. 严格挑法：按训练期聚类 t 排序 ──
say('\n## 3. 严格挑法：训练期按超额的聚类 t 值排序（天数≥30）')
S = A[A.days >= 30].sort_values('t_x5', ascending=False)
for top in (10, 50, 200, 1000):
    sub = S.head(top)
    say(f'  前 {top:>4} 个：训练 t≥{sub.t_x5.min():.2f}，验证期并集 ' + fmt(union_eval(sub.combo, te), pb_te))
ok = A[A.te_n >= 30]
rho_x = ok[['x5', 'te_x5']].corr(method='spearman').iloc[0, 1]
rho_w = ok[['wr5', 'te_wr5']].corr(method='spearman').iloc[0, 1]
say(f'  训练期排名和验证期排名的相关性（验证 n≥30 的 {len(ok)} 个组合）：超额 ρ={rho_x:.3f}，T+5 胜率 ρ={rho_w:.3f}')
say(f'  训练期显著（t≥2）的 {int((ok.t_x5 >= 2).sum())} 个里，验证期超额仍为正的 {(ok[ok.t_x5 >= 2].te_x5 > 0).mean()*100:.0f}%'
    f'，验证期也 t≥2 的 {(ok[ok.t_x5 >= 2].te_t_x5 >= 2).mean()*100:.0f}%')

# ── 4. 打乱标签：纯噪音下会「挑出」多少 ──
say('\n## 4. 噪音对照：训练期按信号日整块打乱收益，再用同样的挑法')
rng = np.random.default_rng(1)
tr_days = np.unique(day_id[tr])
max_t, n_pass = [], []
for p in range(3):
    perm = dict(zip(tr_days, rng.permutation(tr_days)))
    # 每个训练日的信号整块换成另一天的收益（保留同日相关性）；行数不同时按位置循环取
    idx = np.arange(len(d))
    by_day = {k: np.where(day_id == k)[0] for k in tr_days}
    new_idx = idx.copy()
    for k in tr_days:
        src, dst = by_day[perm[k]], by_day[k]
        new_idx[dst] = src[np.arange(len(dst)) % len(src)]
    upP = {k: UP[k][new_idx] for k in HOLD}
    x5P = X5[new_idx]
    P = run(upP, x5P, label='perm')
    bt = np.all([P[f'wr{k}'] > upP[k][tr].mean() * 100 for k in HOLD], axis=0)
    n_pass.append(int((bt & (P.wr5 >= 65)).sum()))
    max_t.append(P[P.days >= 30].t_x5.max())
    say(f'  打乱第 {p+1} 次：「四周期都跑赢 + ≥65%」的组合 {n_pass[-1]} 个；最大 t={max_t[-1]:.2f}')
say(f'  真实数据：{len(V)} 个；最大 t={S.t_x5.max():.2f}；t 超过噪音最大值的组合 {int((S.t_x5 > max(max_t)).sum())} 个')
surv = S[S.t_x5 > max(max_t)]
if len(surv):
    say('  这些组合的验证期并集：' + fmt(union_eval(surv.combo, te), pb_te))
    for r in surv.head(15).itertuples():
        say(f'    {r.name:<55} 训练 n={r.n} x5={r.x5:+.2f} t={r.t_x5:.2f} | 验证 n={r.te_n:.0f} x5={r.te_x5:+.2f} 胜率={r.te_wr5:.1f}%')

# ── 5. 现有 334 个组合在它从没见过的 2006–2019 ──
say('\n## 5. 现有 validated_combos.json（2020–2025 挑出）在 2006–2019 的表现')
combos = json.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'validated_combos.json'), encoding='utf-8'))
pre = (d.signal_day < '2020-01-01').values
mid = ((d.signal_day >= '2020-01-01') & (d.signal_day < '2026-01-01')).values
y26 = (d.signal_day >= '2026-01-01').values
for lab, part in (('2006–2019（没见过）', pre), ('2020–2025（挑选期）', mid), ('2026（之后）', y26)):
    pb = UP[5][part].mean()
    say(f'  {lab} 基础信号：' + fmt(stats(np.ones(len(d), bool), part, pb=pb), pb))
    for g in ('SSS', 'S', 'A', 'B', None):
        sub = [c for c in combos if g is None or c['grade'] == g]
        idxs = [tuple(NAMES.index(x) for x in c['conditions']) for c in sub]
        say(f'    {g or "全部":>4}（{len(sub)} 个，声称 T+5 {np.mean([c["wr5"] for c in sub]):.0f}%）：' + fmt(union_eval(idxs, part), pb))

A.drop(columns=['combo']).to_parquet(os.path.join(OUT, 'combos_all.parquet'), index=False)
open(os.path.join(OUT, 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
say('\n完成：out/combos_all.parquet、out/summary.txt')
