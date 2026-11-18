"""Walk-forward ML vs simple rules on base signals. Output in this folder."""
import json, time, warnings, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error
warnings.filterwarnings("ignore")
T0 = time.time()
OUT = "C:/Users/ft7b6/Desktop/美股/chucun_scan/analysis/rebuild/agents/ml/"
d = pd.read_parquet("C:/Users/ft7b6/Desktop/美股/chucun_scan/analysis/rebuild/data2000/signals_20y.parquet")
d = d[d.in_index & (d.vix < 30) & d.r5.notna() & d.x5.notna()].copy()
d["signal_day"] = pd.to_datetime(d.signal_day)
d = d.sort_values("signal_day").reset_index(drop=True)
cols = list(d.columns)
BOOL = cols[cols.index("base") + 1: cols.index("consec_down3") + 1]
NUM = """rsi cci cmf wr hist atr_pct dist_52w dist_high252 dist_ma200 dist_ma50 ret5d ret20d ret60d vol_ratio20 rsi_prev rsi_min5 mfi stoch_k stoch_d adx hist_norm hist_slope hist_min20_norm ret1d ret3d ret10d ret120d ret250d vol_ratio5 bb_pctb bb_width_ratio atr_ratio body_pct lower_shadow upper_shadow range_pct gap_today dist_ma20 ma50_vs_ma200 ma200_slope20 days_since_low252 down_streak vol60 cmf_chg5 obv_slope20 log_price log_dvol20""".split()
MKT = "vix spx_dist_ma200 spx_ret5d spx_ret20d spx_rsi vix_chg5 rel_ret20d weekday month n_base_day n_base_day_idx".split()
FEATS = BOOL + NUM + MKT
GROUP = {f: "bool" for f in BOOL} | {f: "stock_num" for f in NUM} | {f: "market" for f in MKT}
GROUP["rel_ret20d"] = "stock_num"
X = d[FEATS].astype(float).replace([np.inf, -np.inf], np.nan).values
y_cls = (d.x5 > 0).astype(int).values
y_reg = d.x5.clip(d.x5.quantile(.01), d.x5.quantile(.99)).values
print("rows", len(d), "feats", len(FEATS))

# rule baselines
combos = [c["conditions"] for c in json.load(open("C:/Users/ft7b6/Desktop/美股/chucun_scan/validated_combos.json", encoding="utf-8")) if c["grade"] == "SSS"]
miss = {c for cc in combos for c in cc if c not in d.columns}
assert not miss, miss
sss = np.zeros(len(d), bool)
for cc in combos:
    sss |= d[cc].astype(bool).all(axis=1).values
d["SSS"] = sss
d["NB5"] = d.n_base_day_idx >= 5
d["SSS_or_NB5"] = d.SSS | d.NB5

# ---------------- models ----------------
def prep_lin(Xtr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
    lo, hi = np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0)
    f = lambda A: np.clip(np.where(np.isnan(A), med, A), lo, hi)
    A, B = f(Xtr), f(Xte)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    return (A - mu) / sd, (B - mu) / sd

def fit_pred(kind, target, hp, Xtr, ytr, Xte):
    if kind == "lin":
        A, B = prep_lin(Xtr, Xte)
        if target == "cls":
            m = LogisticRegression(C=hp, max_iter=2000).fit(A, ytr); return m.predict_proba(B)[:, 1], m
        m = Ridge(alpha=hp).fit(A, ytr); return m.predict(B), m
    # GBM with time-split early stopping (last 15% of training, by time)
    nv = int(len(Xtr) * 0.15)
    Cls = HistGradientBoostingClassifier if target == "cls" else HistGradientBoostingRegressor
    kw = dict(max_depth=3, learning_rate=0.03, min_samples_leaf=hp["leaf"], l2_regularization=hp["l2"],
              max_features=0.5, max_bins=64, early_stopping=False, random_state=0)
    m = Cls(max_iter=10, warm_start=True, **kw)
    best, bi = np.inf, 10
    for it in range(10, 401, 10):
        m.set_params(max_iter=it); m.fit(Xtr[:-nv], ytr[:-nv])
        p = m.predict_proba(Xtr[-nv:])[:, 1] if target == "cls" else m.predict(Xtr[-nv:])
        l = log_loss(ytr[-nv:], p) if target == "cls" else mean_squared_error(ytr[-nv:], p)
        if l < best - 1e-6: best, bi = l, it
        elif it - bi >= 60: break
    m = Cls(max_iter=bi, **kw).fit(Xtr, ytr)
    return (m.predict_proba(Xte)[:, 1] if target == "cls" else m.predict(Xte)), m

YEARS = list(range(2008, 2027))
def walk_forward(kind, target, hp, keep_models=False):
    score = np.full(len(d), np.nan); models = {}
    yv = y_cls if target == "cls" else y_reg
    for Y in YEARS:
        first = d.signal_day[d.year == Y].min()
        cut = first - pd.tseries.offsets.BDay(20)
        tr = (d.signal_day < cut).values; te = (d.year == Y).values
        p, m = fit_pred(kind, target, hp, X[tr], yv[tr], X[te])
        score[te] = p
        if keep_models: models[Y] = m
    return score, models

def yearly_spearman(score, mask_years):
    r = []
    for Y in mask_years:
        m = (d.year == Y).values
        r.append(spearmanr(score[m], d.x5.values[m]).correlation)
    return np.nanmean(r)

GRID = {("lin", "cls"): [0.001, 0.01, 0.1], ("lin", "reg"): [100.0, 1000.0, 10000.0],
        ("gbm", "cls"): [{"leaf": 100, "l2": 10.0}, {"leaf": 300, "l2": 10.0}],
        ("gbm", "reg"): [{"leaf": 100, "l2": 10.0}, {"leaf": 300, "l2": 10.0}]}
TUNE = list(range(2008, 2025))
scores, chosen, tune_log = {}, {}, []
for (kind, target), hps in GRID.items():
    best = None
    for hp in hps:
        s, _ = walk_forward(kind, target, hp)
        rho = yearly_spearman(s, TUNE)
        tune_log.append(dict(model=f"{kind}_{target}", hp=str(hp), rho_2008_2024=rho))
        print(f"{kind}_{target} {hp} rho08-24={rho:.4f}  t={time.time()-T0:.0f}s", flush=True)
        if best is None or rho > best[0]: best = (rho, hp, s)
    name = f"{kind}_{target}"; scores[name] = best[2]; chosen[name] = best[1]
pd.DataFrame(tune_log).to_csv(OUT + "tuning_2008_2024.csv", index=False)

# ---------------- evaluation ----------------
def stats(mask, label, period):
    sub = d[mask]
    n = len(sub)
    if n < 5: return dict(period=period, rule=label, n=n)
    e = sub.x5 - sub.x5.mean()
    g = e.groupby(sub.signal_day).sum()
    se = np.sqrt((g ** 2).sum()) / n
    return dict(period=period, rule=label, n=n, wr_r5=round(100 * (sub.r5 > 0).mean(), 1),
                wr_x5=round(100 * (sub.x5 > 0).mean(), 1), mean_r5=round(sub.r5.mean(), 3),
                mean_x5=round(sub.x5.mean(), 3), t_x5_cluster=round(sub.x5.mean() / se, 2))

def top_mask(score, q):
    m = np.zeros(len(d), bool)
    for Y in YEARS:
        idx = np.where((d.year == Y).values)[0]
        thr = np.quantile(score[idx], 1 - q)
        m[idx[score[idx] >= thr]] = True
    return m

PERIODS = {"2008-2024": (2008, 2024), "2025-2026": (2025, 2026), "2008-2026": (2008, 2026)}
rows = []
for pn, (a, b) in PERIODS.items():
    pm = d.year.between(a, b).values
    for lab in ["ALL", "SSS", "NB5", "SSS_or_NB5"]:
        rows.append(stats(pm & (True if lab == "ALL" else d[lab].values), lab, pn))
    for name, s in scores.items():
        for q in (0.2, 0.33):
            rows.append(stats(pm & top_mask(s, q), f"{name}_top{int(q*100)}", pn))
        rows.append(stats(pm & top_mask(s, 0.33) & d.NB5.values, f"{name}_top33&NB5", pn))
res = pd.DataFrame(rows); res.to_csv(OUT + "wf_results.csv", index=False)
# per-year spearman & per-year top20 vs rule
py = []
for Y in YEARS:
    m = (d.year == Y).values
    r = dict(year=Y, n=m.sum())
    for name, s in scores.items():
        r[f"rho_{name}"] = round(spearmanr(s[m], d.x5.values[m]).correlation, 3)
        r[f"x5_top20_{name}"] = round(d.x5.values[m & top_mask(s, .2)].mean(), 2)
    r["x5_SSS_or_NB5"] = round(d.x5.values[m & d.SSS_or_NB5.values].mean(), 2)
    r["x5_ALL"] = round(d.x5.values[m].mean(), 2)
    py.append(r)
py = pd.DataFrame(py); py.to_csv(OUT + "wf_per_year.csv", index=False)
rho_sum = {n: dict(rho_08_24=yearly_spearman(s, TUNE), rho_25_26=yearly_spearman(s, [2025, 2026]),
                   hp=str(chosen[n])) for n, s in scores.items()}
# rule spearman analog: rank-corr of indicator with x5
for lab in ["SSS", "NB5", "SSS_or_NB5"]:
    rho_sum[lab] = dict(rho_08_24=yearly_spearman(d[lab].values.astype(float), TUNE),
                        rho_25_26=yearly_spearman(d[lab].values.astype(float), [2025, 2026]))
print(pd.DataFrame(rho_sum).T)
pd.DataFrame(rho_sum).T.to_csv(OUT + "spearman_summary.csv")
print(res.to_string()); print(f"t={time.time()-T0:.0f}s", flush=True)

# ---------------- permutation importance on WF test years ----------------
rng = np.random.default_rng(0)
imp_rows = []
for name in ["lin_cls", "gbm_cls"]:
    kind, target = name.split("_")
    _, models = walk_forward(kind, target, chosen[name], keep_models=True)
    base_rho = yearly_spearman(scores[name], YEARS)
    def score_te(Y, Xte):
        tr = (d.signal_day < d.signal_day[d.year == Y].min() - pd.tseries.offsets.BDay(20)).values
        m = models[Y]
        if kind == "lin":
            _, B = prep_lin(X[tr], Xte); return m.predict_proba(B)[:, 1]
        return m.predict_proba(Xte)[:, 1]
    units = [(f, [FEATS.index(f)]) for f in FEATS] + \
            [(f"GROUP:{g}", [FEATS.index(f) for f in FEATS if GROUP[f] == g]) for g in ["bool", "stock_num", "market"]]
    for uname, ix in units:
        drops = []
        for rep in range(3):
            rr = []
            for Y in YEARS:
                te = (d.year == Y).values
                Xte = X[te].copy()
                perm = rng.permutation(len(Xte))
                Xte[:, ix] = Xte[perm][:, ix]
                rr.append(spearmanr(score_te(Y, Xte), d.x5.values[te]).correlation)
            drops.append(base_rho - np.nanmean(rr))
        imp_rows.append(dict(model=name, feature=uname, group=GROUP.get(uname, "GROUP"), rho_drop=np.mean(drops)))
    print(name, "perm done", f"t={time.time()-T0:.0f}s", flush=True)
imp = pd.DataFrame(imp_rows).sort_values(["model", "rho_drop"], ascending=[True, False])
imp.to_csv(OUT + "perm_importance.csv", index=False)
for name in ["lin_cls", "gbm_cls"]:
    print(imp[imp.model == name].head(15).to_string())

# ---------------- full-sample fit (overfit quantification) ----------------
ov = []
for name in scores:
    kind, target = name.split("_")
    yv = y_cls if target == "cls" else y_reg
    if kind == "gbm":  # full fit, fixed iterations = 300 (no OOS stopping, as a naive full fit would)
        Cls = HistGradientBoostingClassifier if target == "cls" else HistGradientBoostingRegressor
        m = Cls(max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=chosen[name]["leaf"],
                l2_regularization=chosen[name]["l2"], max_features=0.5, max_bins=64, early_stopping=False,
                random_state=0).fit(X, yv)
        s = m.predict_proba(X)[:, 1] if target == "cls" else m.predict(X)
    else:
        s, _ = fit_pred(kind, target, chosen[name], X, yv, X)
    for pn, (a, b) in {"IS_2008-2026": (2008, 2026), "IS_2025-2026": (2025, 2026)}.items():
        pm = d.year.between(a, b).values
        st = stats(pm & top_mask(s, .2), f"{name}_FULLFIT_top20", pn)
        st["rho_yearly"] = round(yearly_spearman(s, range(a, b + 1)), 4)
        wf = stats(pm & top_mask(scores[name], .2), f"{name}_WF_top20", pn.replace("IS", "WF"))
        wf["rho_yearly"] = round(yearly_spearman(scores[name], range(a, b + 1)), 4)
        ov += [st, wf]
ov = pd.DataFrame(ov); ov.to_csv(OUT + "fullfit_vs_wf.csv", index=False)
print(ov.to_string())
json.dump({k: str(v) for k, v in chosen.items()}, open(OUT + "chosen_hparams.json", "w"))
print(f"done t={time.time()-T0:.0f}s")
