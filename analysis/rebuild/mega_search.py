"""大范围穷举：几百个条件 × 1–N 重组合，训练 2000–2024，验证 2025–今天（只看一次）。

条件 = 原 37 个 + 每个数值特征按训练期分位数切 4 档（≤10%、≤25%、≥75%、≥90%）+ 板块/星期/月份。
同一特征的两档不会出现在同一组合里；加了条件但信号集合不变的组合不重复计。
训练期每个组合算：n、T+1/3/5/10 上涨比例、T+5 超额均值、按信号日聚类的 t 值。
噪音对照：训练期按信号日整块打乱收益，重跑同样的穷举，记录最大 t 和「原挑法」能挑出的个数。

用法：python mega_search.py [最大重数=4] [打乱次数=5] [最小样本=60]
输出 out2000/：combos_d{N}.parquet（t≥2、原挑法通过的、1% 随机样本）、perm_d{N}.csv、summary_d{N}.txt
"""
import os, sys, time, json
import numpy as np, pandas as pd
from multiprocessing import Pool
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data2000')
OUT = os.path.join(HERE, os.environ.get('MS_OUT', 'out2000'))            # 可用环境变量改输出目录
TRAIN_END = pd.Timestamp(os.environ.get('MS_TRAIN_END', '2024-12-31'))  # 可用环境变量改训练截止日
HOLD = (1, 3, 5, 10)

MAX_DEPTH = min(5, int(sys.argv[1]) if len(sys.argv) > 1 else 4)
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MIN_N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
STORE_T = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0   # 只保存 t≥STORE_T 的（以及原挑法里 T+5≥STORE_WR 的）
STORE_WR = float(sys.argv[5]) if len(sys.argv) > 5 else 65.0
SAMPLE_P = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01

ORIG = ['cci_lt100', 'stoch_cross', 'near_52w', 'rsi_lt25', 'rsi_lt20', 'cmf_pos', 'below_kelt', 'wr_lt80',
        'mfi_lt20', 'mfi_lt30', 'cmf_rising', 'obv_rising', 'stoch_rising', 'hammer', 'above_ma200', 'below_bb',
        'vol_gt12', 'dist_ma50_lt_m10', 'dist_ma200_lt_m15', 'ret5d_lt_m3', 'ret10d_lt_m5', 'cci_rising',
        'mfi_rising', 'wr_cross80', 'vol_lt08', 'bullish_engulf', 'rsi_div', 'gap_down_close', 'ret3d_lt_m5',
        'doji_bottom', 'adx_lt25', 'bb_squeeze', 'vol_spike2x', 'atr_low', 'near_ma50', 'wr_rising3', 'consec_down3']
ORIG_GROUP = {'rsi_lt25': 'rsi_prev', 'rsi_lt20': 'rsi_prev', 'mfi_lt20': 'mfi', 'mfi_lt30': 'mfi',
              'vol_gt12': 'vol_ratio20', 'vol_lt08': 'vol_ratio20', 'vol_spike2x': 'vol_ratio20'}
NUMERIC = ['rsi', 'cci', 'cmf', 'wr', 'hist', 'atr_pct', 'dist_52w', 'dist_high252', 'dist_ma200', 'dist_ma50',
           'ret5d', 'ret20d', 'ret60d', 'vol_ratio20', 'rsi_prev', 'rsi_min5', 'mfi', 'stoch_k', 'stoch_d', 'adx',
           'hist_norm', 'hist_slope', 'hist_min20_norm', 'ret1d', 'ret3d', 'ret10d', 'ret120d', 'ret250d',
           'vol_ratio5', 'bb_pctb', 'bb_width_ratio', 'atr_ratio', 'body_pct', 'lower_shadow', 'upper_shadow',
           'range_pct', 'gap_today', 'dist_ma20', 'ma50_vs_ma200', 'ma200_slope20', 'days_since_low252',
           'down_streak', 'vol60', 'cmf_chg5', 'obv_slope20', 'log_price', 'log_dvol20',
           'vix', 'vix_chg5', 'spx_dist_ma200', 'spx_ret5d', 'spx_ret20d', 'spx_rsi', 'n_base_day_idx', 'rel_ret20d']
QS = ((0.10, '<='), (0.25, '<='), (0.75, '>='), (0.90, '>='))


# ── 数据与条件 ─────────────────────────────────────────────
def build():
    s = pd.read_parquet(os.path.join(DATA, 'signals_20y.parquet'))
    d = s[s.in_index & (s.vix < 30) & s.r10.notna() & s.b10.notna()]
    d = d.sort_values(['signal_day', 'ticker']).reset_index(drop=True)
    tr = (d.signal_day <= TRAIN_END).values
    names, groups, cols = [], [], []
    for c in ORIG:
        names.append(c); groups.append(ORIG_GROUP.get(c, c)); cols.append(d[c].values.astype(bool))
    for f in NUMERIC:
        x = d[f].astype(float)
        seen = set()
        for q, op in QS:
            thr = float(x[tr].quantile(q))
            m = (x <= thr).values if op == '<=' else (x >= thr).values
            key = (op, round(thr, 6))
            if key in seen or m[tr].mean() > 0.6 or m[tr].mean() < 0.02:   # 分位数重合或太宽太窄的跳过
                continue
            seen.add(key)
            names.append(f'{f}{op}{thr:.3g}'); groups.append(f); cols.append(m)
    for cat, vals in (('sector_now', sorted(v for v in d.sector_now.unique() if v)),
                      ('weekday', range(5)), ('month', range(1, 13))):
        for v in vals:
            m = (d[cat] == v).values
            if m[tr].mean() >= 0.02:
                names.append(f'{cat}={v}'); groups.append(cat); cols.append(m)
    C = np.column_stack(cols)
    return d, tr, names, groups, C


def pack(C_rows):
    """bool 矩阵 (n_rows, K) → (K, n_words) uint64 位集"""
    n, K = C_rows.shape
    W = (n + 63) // 64
    pad = np.zeros((K, W * 64), bool); pad[:, :n] = C_rows.T
    b = np.packbits(pad, axis=1, bitorder='little')          # 第 r 行 = 第 r//8 字节的第 r%8 位
    return np.ascontiguousarray(b).view('<u8').astype(np.uint64)   # 第 r 行 = 第 r//64 字的第 r%64 位


# ── numba 内核 ─────────────────────────────────────────────
@njit(cache=True)
def popcnt(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=True)
def node_stats(mask, x5, w1, w3, w5, w10, day):
    """遍历位集里的行：n、四个周期的上涨数、x5 均值和按信号日聚类的 t"""
    n = 0; s = 0.0; c1 = 0; c3 = 0; c5 = 0; c10 = 0
    A = 0.0; B = 0.0; Cn = 0.0; G = 0
    cur = -1; Sg = 0.0; ng = 0
    for wi in range(mask.shape[0]):
        w = mask[wi]
        while w != np.uint64(0):
            low = w & (~w + np.uint64(1))
            r = wi * 64 + int(popcnt(low - np.uint64(1)))
            w ^= low
            dd = day[r]
            if dd != cur:
                if ng > 0:
                    A += Sg * Sg; B += Sg * ng; Cn += ng * ng; G += 1
                cur = dd; Sg = 0.0; ng = 0
            v = x5[r]
            Sg += v; ng += 1
            n += 1; s += v; c1 += w1[r]; c3 += w3[r]; c5 += w5[r]; c10 += w10[r]
    if ng > 0:
        A += Sg * Sg; B += Sg * ng; Cn += ng * ng; G += 1
    mu = s / n if n > 0 else 0.0
    t = 0.0
    if G > 1 and n > 1:
        var = (A - 2 * mu * B + mu * mu * Cn) * G / (G - 1)
        if var > 0:
            t = mu / (np.sqrt(var) / n)
    return n, mu, t, G, c1, c3, c5, c10


@njit(cache=True)
def dfs(first, bits, grp, max_depth, min_n, x5, w1, w3, w5, w10, day, base_wr, store, cap, sample_p, store_t, store_wr):
    """以条件 first 开头的全部组合（下标递增），深度优先。返回记录数、最大 t、原挑法通过数。"""
    K, W = bits.shape
    stack_idx = np.empty(max_depth, np.int64)
    masks = np.empty((max_depth, W), np.uint64)
    parent_n = np.empty(max_depth, np.int64)
    nrec = 0; max_t = -1e9; n_orig = 0
    masks[0] = bits[first]
    stack_idx[0] = first
    depth = 0
    nxt = np.empty(max_depth, np.int64)
    # 先处理根
    tot = 0
    for wi in range(W): tot += np.int64(popcnt(masks[0, wi]))
    if tot < min_n:
        return nrec, max_t, n_orig
    parent_n[0] = -1
    nxt[0] = first + 1
    level_n = np.empty(max_depth, np.int64)
    level_n[0] = tot
    # 记录根
    res = node_stats(masks[0], x5, w1, w3, w5, w10, day)
    nrec, max_t, n_orig = _record(res, stack_idx, 1, store, nrec, cap, max_t, n_orig, base_wr, sample_p, store_t, store_wr)
    while depth >= 0:
        if depth + 1 >= max_depth or nxt[depth] >= K:
            depth -= 1
            continue
        j = nxt[depth]; nxt[depth] += 1
        ok = True
        for q in range(depth + 1):
            if grp[stack_idx[q]] == grp[j]:
                ok = False; break
        if not ok:
            continue
        cnt = 0
        for wi in range(W):
            m = masks[depth, wi] & bits[j, wi]
            masks[depth + 1, wi] = m
            cnt += np.int64(popcnt(m))
        if cnt < min_n or cnt == level_n[depth]:      # 样本不够；或加了条件集合没变（重复）
            continue
        stack_idx[depth + 1] = j
        level_n[depth + 1] = cnt
        res = node_stats(masks[depth + 1], x5, w1, w3, w5, w10, day)
        nrec, max_t, n_orig = _record(res, stack_idx, depth + 2, store, nrec, cap, max_t, n_orig, base_wr, sample_p, store_t, store_wr)
        depth += 1
        nxt[depth] = j + 1
    return nrec, max_t, n_orig


@njit(cache=True)
def _record(res, stack_idx, size, store, nrec, cap, max_t, n_orig, base_wr, sample_p, store_t, store_wr):
    n, mu, t, G, c1, c3, c5, c10 = res
    wr1 = c1 / n * 100; wr3 = c3 / n * 100; wr5 = c5 / n * 100; wr10 = c10 / n * 100
    orig = wr1 > base_wr[0] and wr3 > base_wr[1] and wr5 > base_wr[2] and wr10 > base_wr[3] and wr5 >= 65
    if orig: n_orig += 1
    if G >= 30 and t > max_t: max_t = t
    if store.shape[0] > 0 and nrec < cap and (t >= store_t or (orig and wr5 >= store_wr) or np.random.random() < sample_p):
        for q in range(5):
            store[nrec, q] = stack_idx[q] if q < size else -1
        store[nrec, 5] = n; store[nrec, 6] = mu; store[nrec, 7] = t; store[nrec, 8] = G
        store[nrec, 9] = wr1; store[nrec, 10] = wr3; store[nrec, 11] = wr5; store[nrec, 12] = wr10
        store[nrec, 13] = 1.0 if orig else 0.0
        nrec += 1
    return nrec, max_t, n_orig


# ── 多进程 ─────────────────────────────────────────────────
G_ = {}
def init(payload):
    G_.update(payload)

def work(first):
    p = G_
    cap = 1_500_000 if p['store'] else 0
    store = np.empty((cap, 14), np.float64)
    nrec, mt, no = dfs(first, p['bits'], p['grp'], p['depth'], p['min_n'], p['x5'], p['w1'], p['w3'], p['w5'],
                       p['w10'], p['day'], p['base_wr'], store, cap, SAMPLE_P, STORE_T, STORE_WR)
    if cap and nrec >= cap:
        print(f'    注意：起始条件 {first} 的记录达到上限 {cap}，后面的没存', flush=True)
    return first, store[:nrec].copy(), mt, no


def search(payload, K, label):
    t0 = time.time()
    recs, max_t, n_orig = [], -1e9, 0
    with Pool(max(1, os.cpu_count() - 2), initializer=init, initargs=(payload,)) as pool:
        for k, (first, st, mt, no) in enumerate(pool.imap_unordered(work, range(K))):
            recs.append(st); max_t = max(max_t, mt); n_orig += no
            if label == 'real' and (k + 1) % 20 == 0:
                print(f'    {k + 1}/{K} 个起始条件完成，{time.time() - t0:.0f}s', flush=True)
    return (np.vstack(recs) if recs and payload['store'] else None), max_t, n_orig, time.time() - t0


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)

    d, tr, names, groups, C = build()
    te = ~tr
    K = len(names)
    gid = pd.factorize(pd.Series(groups))[0].astype(np.int64)
    say(f'样本：{len(d)} 条基础信号（在指数内、VIX<30）。训练 {tr.sum()} 条（{d.signal_day[tr].min().date()}–2024），'
        f'验证 {te.sum()} 条（2025–{d.signal_day.max().date()}）')
    say(f'条件 {K} 个（原 37 + 数值分档 + 板块/星期/月份）；最多 {MAX_DEPTH} 重；训练期 n≥{MIN_N}')

    dtr = d[tr]
    x5 = dtr.x5.values.astype(np.float64)
    W = {k: (dtr[f'r{k}'].values > 0).astype(np.int64) for k in HOLD}
    day = pd.factorize(dtr.signal_day)[0].astype(np.int64)
    base_wr = np.array([W[k].mean() * 100 for k in HOLD])
    say('训练期基础信号上涨比例：' + ' '.join(f'T+{k} {b:.1f}%' for k, b in zip(HOLD, base_wr)))
    bits = pack(C[tr])

    payload = dict(bits=bits, grp=gid, depth=MAX_DEPTH, min_n=MIN_N, x5=x5, w1=W[1], w3=W[3], w5=W[5], w10=W[10],
                   day=day, base_wr=base_wr, store=True)
    say('\n真实数据穷举...')
    rec, max_t, n_orig, sec = search(payload, K, 'real')
    say(f'  用时 {sec/60:.1f} 分钟；原挑法（四周期都跑赢 + T+5≥65%）通过 {n_orig} 个；最大 t={max_t:.2f}；保存 {len(rec)} 条')

    # 噪音对照
    rng = np.random.default_rng(7)
    perm_rows = []
    days_u = np.unique(day)
    by_day = [np.where(day == k)[0] for k in days_u]
    for p in range(N_PERM):
        perm = rng.permutation(len(days_u))
        new_idx = np.empty(len(day), np.int64)
        for a, b in enumerate(perm):
            dst, src = by_day[a], by_day[b]
            new_idx[dst] = src[np.arange(len(dst)) % len(src)]
        pl = dict(payload, x5=x5[new_idx], w1=W[1][new_idx], w3=W[3][new_idx], w5=W[5][new_idx],
                  w10=W[10][new_idx], store=False,
                  base_wr=np.array([W[k][new_idx].mean() * 100 for k in HOLD]))
        _, mt, no, sec = search(pl, K, 'perm')
        perm_rows.append({'perm': p + 1, 'max_t': mt, 'n_orig': no})
        say(f'  打乱第 {p+1} 次：最大 t={mt:.2f}，原挑法通过 {no} 个（{sec/60:.1f} 分钟）')
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(os.path.join(OUT, f'perm_d{MAX_DEPTH}_n{MIN_N}.csv'), index=False)

    # 保存训练期结果 + 补验证期统计
    cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'n', 'x5', 't', 'days', 'wr1', 'wr3', 'wr5', 'wr10', 'orig']
    R = pd.DataFrame(rec, columns=cols)
    for c in cols[:5]: R[c] = R[c].astype(int)
    R['name'] = [' + '.join(names[int(i)] for i in row if i >= 0) for row in R[cols[:5]].values]
    dte = d[te]
    bits_te = pack(C[te])
    x5t = dte.x5.values.astype(np.float64)
    Wt = {k: (dte[f'r{k}'].values > 0).astype(np.int64) for k in HOLD}
    dayt = pd.factorize(dte.signal_day)[0].astype(np.int64)
    te_stats = np.array([eval_combo(bits_te, row, x5t, Wt, dayt) for row in R[cols[:5]].values.astype(int)])
    for j, c in enumerate(['te_n', 'te_x5', 'te_t', 'te_days', 'te_wr5', 'te_wr10']):
        R[c] = te_stats[:, j]
    R.to_parquet(os.path.join(OUT, f'combos_d{MAX_DEPTH}_n{MIN_N}.parquet'), index=False)

    # ── 汇总 ──
    base_te = {k: Wt[k].mean() * 100 for k in HOLD}
    say(f'\n验证期基础信号：n={te.sum()}  T+5 上涨 {base_te[5]:.1f}%  T+10 {base_te[10]:.1f}%  超额 x5 {x5t.mean():+.2f}%')
    thr = perm.max_t.max() if len(perm) else np.inf
    q95 = perm.max_t.quantile(0.95) if len(perm) else np.inf
    def union(rows):
        m = np.zeros(len(d), bool)
        for row in rows:
            mm = np.ones(len(d), bool)
            for i in row:
                if i >= 0: mm &= C[:, i]
            m |= mm
        return m
    def show(lab, rows):
        """验证期：命中这些组合的信号 vs 同期没命中的基础信号（按信号日聚类的差值 t）"""
        m = union(rows)
        a = m & te
        if a.sum() == 0:
            say(f'  {lab}: 验证期 0 条'); return
        r5 = d.r5.values[a]; xx = d.x5.values[a]
        rest = te & ~m
        dx, tx = diff_t(m[te], d.x5.values[te], d.signal_day.values[te])
        dw, tw = diff_t(m[te], (d.r5.values[te] > 0) * 100.0, d.signal_day.values[te])
        say(f'  {lab}: 验证期 n={a.sum()}（没命中 {rest.sum()}）  T+5 上涨 {(r5 > 0).mean()*100:.1f}% vs {(d.r5.values[rest] > 0).mean()*100 if rest.sum() else float("nan"):.1f}%'
            f'（差 {dw:+.1f} 点 t={tw:.2f}）  超额 {xx.mean():+.2f}%（差 {dx:+.2f} t={tx:.2f}）')

    O = R[R.orig == 1]
    say(f'\n## 原挑法：训练期通过 {n_orig} 个（打乱后平均 {perm.n_orig.mean():.0f} 个）')
    if len(O):
        say(f'  训练期平均 T+5 胜率 {O.wr5.mean():.1f}% → 验证期 {O.te_wr5.mean():.1f}%')
        show('原挑法全部并集', O[cols[:5]].values.astype(int))
        for lo, hi, g in ((65, 70, 'B'), (70, 80, 'A'), (80, 101, 'S')):
            sub = O[(O.wr5 >= lo) & (O.wr5 < hi)]
            if len(sub): show(f'{g} 级 {len(sub)} 个', sub[cols[:5]].values.astype(int))

    S = R[R.days >= 30].sort_values('t', ascending=False)
    say(f'\n## 按训练期 t 排序（噪音最大 t：均值 {perm.max_t.mean():.2f}，最大 {thr:.2f}）')
    for top in (10, 100, 1000, 10000):
        if len(S) >= top: show(f'前 {top}', S.head(top)[cols[:5]].values.astype(int))
    surv = S[S.t > thr]
    say(f'  超过噪音最大值的组合：{len(surv)} 个')
    if len(surv):
        show('这些组合并集', surv[cols[:5]].values.astype(int))
        for r in surv.head(25).itertuples():
            say(f'    {r.name[:80]:<80} 训练 n={r.n:.0f} x5={r.x5:+.2f} t={r.t:.2f} | 验证 n={r.te_n:.0f} x5={r.te_x5:+.2f} 胜率={r.te_wr5:.1f}%')
    # 稳定性：训练期前后两半（2000–2012 / 2013–2024）都 t≥2
    say('\n## 稳定组合：训练期前半（2000–2012）和后半（2013–2024）都显著（t≥2，各自天数≥20）')
    h1 = (dtr.signal_day <= os.environ.get('MS_H1_END', '2012-12-31')).values
    sub_stats = {}
    for lab, part in (('h1', h1), ('h2', ~h1)):
        bp = pack(C[tr][part])
        dd = dtr[part]
        xw = {k: (dd[f'r{k}'].values > 0).astype(np.int64) for k in HOLD}
        dy = pd.factorize(dd.signal_day)[0].astype(np.int64)
        cand = R[(R.t >= 2) & (R.days >= 40)]
        sub_stats[lab] = pd.DataFrame([eval_combo(bp, row, dd.x5.values.astype(np.float64), xw, dy)
                                       for row in cand[cols[:5]].values.astype(int)],
                                      index=cand.index, columns=['n', 'x5', 't', 'days', 'wr5', 'wr10'])
    H = R.loc[sub_stats['h1'].index].copy()
    for lab in ('h1', 'h2'):
        H[f'{lab}_t'] = sub_stats[lab].t.values; H[f'{lab}_days'] = sub_stats[lab].days.values
        H[f'{lab}_wr5'] = sub_stats[lab].wr5.values
    ST = H[(H.h1_t >= 2) & (H.h2_t >= 2) & (H.h1_days >= 20) & (H.h2_days >= 20)].copy()
    ST['min_t'] = ST[['h1_t', 'h2_t']].min(axis=1)
    ST = ST.sort_values('min_t', ascending=False)
    say(f'  训练期 t≥2 的 {len(H)} 个里，前后两半都显著的 {len(ST)} 个')
    for top in (10, 100, 1000):
        if len(ST) >= 1: show(f'稳定 前 {min(top, len(ST))}', ST.head(top)[cols[:5]].values.astype(int))
    for r in ST.head(20).itertuples():
        say(f'    {r.name[:75]:<75} 前半 t={r.h1_t:.1f} 后半 t={r.h2_t:.1f} 胜率 {r.h1_wr5:.0f}/{r.h2_wr5:.0f}% | 验证 n={r.te_n:.0f} 胜率={r.te_wr5:.1f}% x5={r.te_x5:+.2f}')
    ST.to_parquet(os.path.join(OUT, f'stable_d{MAX_DEPTH}_n{MIN_N}.parquet'), index=False)

    smp = R[(R.te_n >= 30)]
    say(f'  训练/验证排名相关（验证 n≥30 的 {len(smp)} 个）：超额 ρ={smp[["x5","te_x5"]].corr("spearman").iloc[0,1]:.3f}，'
        f'T+5 胜率 ρ={smp[["wr5","te_wr5"]].corr("spearman").iloc[0,1]:.3f}')
    sig = R[(R.t >= 2) & (R.te_n >= 30)]
    say(f'  训练 t≥2 的 {len(sig)} 个：验证期超额为正 {(sig.te_x5 > 0).mean()*100:.0f}%，验证期也 t≥2 {(sig.te_t >= 2).mean()*100:.1f}%')

    open(os.path.join(OUT, f'summary_d{MAX_DEPTH}_n{MIN_N}.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
    say(f'\n完成：out2000/summary_d{MAX_DEPTH}_n{MIN_N}.txt')


def cluster_t_py(y, days):
    df = pd.DataFrame({'y': y, 'd': days})
    n = len(y); mu = y.mean()
    g = df.groupby('d').y.agg(['sum', 'size'])
    G = len(g)
    if G < 2: return np.nan, G
    res = g['sum'] - g['size'] * mu
    se = np.sqrt((res ** 2).sum() * G / (G - 1)) / n
    return mu / se, G


def diff_t(hit, y, days):
    """y 对「是否命中」回归，按信号日聚类：返回 命中−没命中 的差和 t"""
    if hit.sum() == 0 or (~hit).sum() == 0: return np.nan, np.nan
    X = np.column_stack([np.ones(len(y)), hit.astype(float)])
    XtX = np.linalg.inv(X.T @ X)
    beta = XtX @ X.T @ y
    e = y - X @ beta
    g = pd.factorize(days)[0]
    U = np.zeros((g.max() + 1, 2))
    np.add.at(U, g, X * e[:, None])
    V = XtX @ (U.T @ U) @ XtX
    return beta[1], beta[1] / np.sqrt(V[1, 1])


def eval_combo(bits, row, x5, W, day):
    m = None
    for i in row:
        if i < 0: break
        m = bits[i].copy() if m is None else (m & bits[i])
    n, mu, t, G, c1, c3, c5, c10 = node_stats(m, x5, W[1], W[3], W[5], W[10], day)
    if n == 0: return (0, np.nan, np.nan, 0, np.nan, np.nan)
    return (n, mu, t, G, c5 / n * 100, c10 / n * 100)


if __name__ == '__main__':
    main()
