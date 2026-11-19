"""训练期（2000–2025-11）跑赢同期 SPY 且扣费后每笔净收益 > 0.1% 的组合，到 2026 年还能做到的有多少。
条件、信号、成本同 big_conditions.py（前 216 行原样执行）。T1 收盘买，持有 5 天 / 10 天各算一遍。
训练用时间加权平均（半衰期 10 年），≥30 笔；2026 用简单平均。
用法：python pass_filter.py          （全部条件）
      EXCL_DAY=1 python pass_filter.py （去掉大盘/日历类条件）
"""
import os, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'big_conditions.py'), encoding='utf-8').read().split('\n')
exec('\n'.join(src[:216]))          # 到 "可用条件" 为止：得到 X、names、s、w、x5、x10、tr_、te、COST、MINN

res = {}
for N, xx in ((5, x5), (10, x10)):
    net = s[f'r{N}'].values - COST
    trm = (s.signal_day < pd.Timestamp('2025-12-01' if N == 10 else '2025-12-20')).values & ~np.isnan(xx)
    tem = (s.year == 2026).values & ~np.isnan(xx)
    Xa, Xb = X[trm], X[tem]
    wa = w[trm].astype(np.float32)
    A = np.vstack([np.ones_like(wa), wa, wa * xx[trm], wa * net[trm]]).astype(np.float32)
    B = np.vstack([np.ones(tem.sum()), xx[tem], net[tem]]).astype(np.float32)
    keys, sa, sb = [], [], []
    def push(ks, ma, mb): keys.extend(ks); sa.append(A @ ma); sb.append(B @ mb)
    for a in range(NC): push([(a,)], Xa[:, a:a + 1], Xb[:, a:a + 1])
    for a in range(NC - 1):
        push([(a, b) for b in range(a + 1, NC)], Xa[:, a:a + 1] * Xa[:, a + 1:], Xb[:, a:a + 1] * Xb[:, a + 1:])
    for a, b in itertools.combinations(range(NC), 2):
        m2 = Xa[:, a] * Xa[:, b]
        if m2.sum() < MINN or b + 1 >= NC: continue
        push([(a, b, c) for c in range(b + 1, NC)], Xa[:, b + 1:] * m2[:, None], Xb[:, b + 1:] * (Xb[:, a] * Xb[:, b])[:, None])
    sa = np.hstack(sa).astype(np.float64); sb = np.hstack(sb).astype(np.float64)
    n, W, Wx, Wn = sa; n26, X26, N26 = sb
    mx, mn = Wx / np.maximum(W, 1e-12), Wn / np.maximum(W, 1e-12)
    ok_tr = (n >= MINN) & (mx > 0) & (mn > 0.1)
    ex26 = X26 / np.maximum(n26, 1); nt26 = N26 / np.maximum(n26, 1)
    base_x, base_n = np.nanmean(xx[tem]), np.nanmean(net[tem])
    print(f'\n===== 持有 {N} 天 =====')
    print(f'组合总数 {len(keys):,}；训练 ≥30 笔 {int((n >= MINN).sum()):,}；训练跑赢 SPY 且净收益>0.1%：{int(ok_tr.sum()):,}')
    print(f'参照：2026 全部基础信号 超额 {base_x:+.2f}%，净收益 {base_n:+.2f}%')
    for k in (1, 5, 10, 20):
        has = ok_tr & (n26 >= k); both = has & (ex26 > 0) & (nt26 > 0.1)
        beat = has & (ex26 > base_x) & (nt26 > base_n)
        print(f'2026 年 ≥{k} 笔的 {int(has.sum()):,} 个 → 跑赢 SPY 且净收益>0.1%：{int(both.sum()):,}（{both.sum()/max(has.sum(),1)*100:.0f}%）'
              f'；还比全部基础信号好：{int(beat.sum()):,}（{beat.sum()/max(has.sum(),1)*100:.0f}%）')
        bad = (n >= MINN) & ~ok_tr & (n26 >= k)
        print(f'    对照：训练没通过的 {int(bad.sum()):,} 个 → 2026 跑赢 SPY 且净收益>0.1% {((ex26[bad]>0)&(nt26[bad]>0.1)).mean()*100:.0f}%'
              f'；比全部基础信号好 {((ex26[bad]>base_x)&(nt26[bad]>base_n)).mean()*100:.0f}%')
    sel = ok_tr & (n26 >= 10) & (ex26 > 0) & (nt26 > 0.1)
    out = pd.DataFrame({'组合': ['＋'.join(names[j] for j in keys[i]) for i in np.where(sel)[0]],
                        '训练笔数': n[sel].astype(int), '训练加权超额%': mx[sel].round(2), '训练加权净收益%': mn[sel].round(2),
                        '2026笔数': n26[sel].astype(int), '2026超额%': ex26[sel].round(2), '2026净收益%': nt26[sel].round(2)}
                       ).sort_values('2026超额%', ascending=False)
    out.to_csv(os.path.join(HERE, f'pass_{N}d{SUF}.csv'), index=False, encoding='utf-8-sig')
