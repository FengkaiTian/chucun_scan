"""穷举：所有单条规则 + 两两组合，对「5日超额收益」和「惨败率」做簇稳健检验，
FWER 用打乱标签的最大|t|临界值校正，再看样本外(2026)能否复现。"""
import numpy as np, pandas as pd, itertools, sys
rng = np.random.default_rng(7)
d = pd.read_csv('signals_dataset.csv', encoding='utf-8-sig', parse_dates=['signal_day'], keep_default_na=False, na_values=[''])
d['grade'] = d['grade'].fillna('')
d = d[d.r5.notna() & d.b5.notna() & d.r10.notna()].reset_index(drop=True)
d['ex5'] = d.r5 - d.b5
d['crash'] = ((d.r5 <= -5) | (d.r10 <= -8)).astype(float)
d['dow'] = d.signal_day.dt.dayofweek
IS = (d.signal_day < '2026-01-01').values; OOS = ~IS
day_codes = pd.factorize(d.signal_day)[0]

num = ['rsi','cci','cmf','wr','hist','dist_52w','deviation','vol_ratio5','ret5d','ret20d','ret60d','dist_high252','dist_ma200',
       'atr_pct','gap_open_pct','spx_vs_ma200','spx_ret5','spx_ret20','vix','vix_chg5','n_base_day','n_graded_day','sector_n_day',
       'days_to_earn','days_since_earn','n_conds','combo_wr5']
rules, names = [], []
def add(name, mask):
    m = np.asarray(mask, dtype=bool)
    if m[IS].sum() >= 60 and m[OOS].sum() >= 15:   # 样本内至少60条，样本外至少15条
        rules.append(m); names.append(name)
for f in num:
    v = d[f]; qs = np.nanquantile(v[IS], [.2, .4, .6, .8])
    for q, th in zip((20, 40, 60, 80), qs):
        add(f'{f} >= {th:.2f} (P{q})', (v >= th).values & v.notna().values)
        add(f'{f} <= {th:.2f} (P{q})', (v <= th).values & v.notna().values)
for c in sorted({x for s in d.conds.fillna('') for x in s.split('|') if x}):
    add(f'cond:{c}', d.conds.fillna('').str.contains(rf'(?:^|\|){c}(?:\||$)').values)
add('earn_in_hold=1', (d.earn_in_hold == 1).values)
add('graded', (d.grade != '').values)
for g in ('SSS', 'A', 'B'): add(f'grade={g}', (d.grade == g).values)
for s in d.sector.unique(): add(f'sector={s}', (d.sector == s).values)
for k in range(5): add(f'weekday={k}', (d.dow == k).values)
for k in (1, 2, 3, 4): add(f'n_base_day=={k}', (d.n_base_day == k).values)
R = np.array(rules); print('单条规则', len(names), flush=True)
# 两两组合
pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        m = R[i] & R[j]
        if m[IS].sum() >= 60 and m[OOS].sum() >= 15 and m[IS].sum() < IS.sum() * 0.9:
            pairs.append((i, j))
print('两两组合', len(pairs), flush=True)
allr = [(names[i], R[i]) for i in range(len(names))] + [(names[i] + '  &  ' + names[j], R[i] & R[j]) for i, j in pairs]
M = np.array([m for _, m in allr]); N = [n for n, _ in allr]
print('总规则数', len(N), flush=True)

def tstats(y, mask_rows):
    """每条规则 vs 其余：y = a + b*I 的 b 与簇稳健 t（按 signal_day 聚类）"""
    idx = np.where(mask_rows)[0]
    yy = y[idx]; dc = pd.factorize(day_codes[idx])[0]; nd = dc.max() + 1
    I = M[:, idx].astype(float)
    Ib = I.mean(1, keepdims=True); Ic = I - Ib
    ybar = yy.mean(); var = (Ic ** 2).sum(1)
    b = (Ic * (yy - ybar)).sum(1) / np.where(var > 0, var, np.nan)
    e = (yy - ybar)[None, :] - b[:, None] * Ic
    sc = Ic * e
    S = np.zeros((M.shape[0], nd))
    for k in range(nd):
        S[:, k] = sc[:, dc == k].sum(1)
    se = np.sqrt((S ** 2).sum(1)) / np.where(var > 0, var, np.nan)
    return b, b / se, I.sum(1)

out = []
for tgt in ('ex5', 'crash'):
    y = d[tgt].values.astype(float)
    b, t, n = tstats(y, IS)
    # 打乱标签的 FWER 临界值（|t| 最大值的 95 分位）
    mx = []
    for _ in range(60):
        yp = y.copy(); yp[IS] = rng.permutation(y[IS])
        _, tp, _ = tstats(yp, IS); mx.append(np.nanmax(np.abs(tp)))
    crit = float(np.quantile(mx, .95))
    bo, to, no = tstats(y, OOS)
    df = pd.DataFrame(dict(rule=N, n_is=n.astype(int), eff_is=b, t_is=t, n_oos=no.astype(int), eff_oos=bo, t_oos=to))
    sig = df[np.abs(df.t_is) >= crit]
    thr_bh = 2.0
    cand = df[np.abs(df.t_is) >= thr_bh]          # 宽松：|t|>=2（≈未校正 p<0.05）
    rep = cand[np.sign(cand.eff_is) == np.sign(cand.eff_oos)]
    print(f'\n=== 目标 {tgt}：规则 {len(df)} 条 ===', flush=True)
    print(f'打乱标签 FWER 5% 临界 |t| = {crit:.2f}；样本内通过 FWER 的规则数 = {len(sig)}', flush=True)
    print(f'样本内 |t|>=2 的规则 {len(cand)} 条；样本外方向一致 {len(rep)} 条 ({len(rep)/max(len(cand),1)*100:.0f}%，纯噪音约 50%)；'
          f'样本外同向且 |t|>=2：{int(((np.sign(cand.eff_is)==np.sign(cand.eff_oos))&(np.abs(cand.t_oos)>=2)).sum())} 条', flush=True)
    show = (sig if len(sig) else cand).copy()
    show['score'] = np.abs(show.t_is)
    show = show.sort_values('score', ascending=False).head(12)
    print(show[['rule', 'n_is', 'eff_is', 't_is', 'n_oos', 'eff_oos', 't_oos']].round(2).to_string(index=False), flush=True)
    df['target'] = tgt; out.append(df)
pd.concat(out).to_csv('explore_rules_all.csv', index=False, encoding='utf-8-sig')
