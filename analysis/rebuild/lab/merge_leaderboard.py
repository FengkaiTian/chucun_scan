"""合并三个 agent 的结果，出总排行，并在保留期 2025–2026 做一次最终检验。

排名分数 score = min(train_t, valid_t)（每笔扣成本后收益的聚类 t），要求 train_n≥50、valid_n≥20。
噪音线 = 随机对照策略里的最高分。
输出：out/leaderboard.csv、out/leaderboard.md
"""
import os, glob
import numpy as np, pandas as pd
from lab_common import unseal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
frames = []
for p in glob.glob(os.path.join(OUT, 'agent*.parquet')):
    df = pd.read_parquet(p)
    if 'agent' not in df.columns:
        df['agent'] = os.path.basename(p).split('.')[0]
    df['source'] = os.path.basename(p)
    frames.append(df)
A = pd.concat(frames, ignore_index=True)
A['hold'] = A['hold'].astype(str)
for c in ('train_n', 'valid_n', 'train_t', 'valid_t', 'train_t_excess', 'valid_t_excess'):
    A[c] = pd.to_numeric(A[c], errors='coerce')
A['is_random'] = A.family.astype(str).str.contains('随机')
ok = (A.train_n >= 50) & (A.valid_n >= 20)
# 主分数：扣同期标普500 后的超额 t（两期取小）。原始收益 t 会被"持有越久越赚大盘的钱"带偏，随机持有 20 天都能进前 10。
A['score'] = np.where(ok, np.minimum(A.train_t_excess, A.valid_t_excess), np.nan)
A['score_raw'] = np.where(ok, np.minimum(A.train_t, A.valid_t), np.nan)
lines = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)

say(f'# 策略大排行\n\n共 {len(A)} 个策略（' + '，'.join(f'{k} {v}' for k, v in A.source.value_counts().items()) +
    f'）；样本够的 {ok.sum()} 个；其中随机对照 {A.is_random.sum()} 个。')
rnd = A[A.is_random & ok]
noise = rnd.score.max() if len(rnd) else np.nan
say(f'随机对照最高分 {noise:.2f}（95 分位 {rnd.score.quantile(.95):.2f}）。高于这条线的真实策略：'
    f'{((A.score > noise) & ~A.is_random).sum()} 个。\n')

R = A[ok & ~A.is_random].sort_values('score', ascending=False).copy()
# 保留期解封（只解前 300 名 + 随机对照）
def ho(row):
    s = row.get('sealed')
    if isinstance(s, str) and s:
        try:
            return unseal(s)
        except Exception:
            return {}
    return {}
top = R.head(300).copy()
H = top.apply(ho, axis=1, result_type='expand')
for c in ('n', 'win', 'avg', 't', 'excess'):
    top[f'ho_{c}'] = H[c] if c in H else np.nan
rH = rnd.apply(ho, axis=1, result_type='expand') if len(rnd) else pd.DataFrame()

say('## 前 50 名（训练 2000–19 / 验证 2020–24 / 保留期 2025–26）\n')
say('| # | 策略 | 方向 | 持有 | 训练 胜率/每笔/超额(t) | 验证 胜率/每笔/超额(t) | 每年次数 | 2025–26 次数/胜率/每笔/超额 |')
say('|---|---|---|---|---|---|---|---|')
for i, r in enumerate(top.head(50).itertuples(), 1):
    hoS = (f'{r.ho_n:.0f} / {r.ho_win:.0f}% / {r.ho_avg:+.2f}% / {r.ho_excess:+.2f}%'
           if pd.notna(r.ho_n) and r.ho_n > 0 else '—')
    say(f'| {i} | {str(r.name)[:60]} | {r.agent} | {r.hold} | {r.train_win:.0f}% / {r.train_avg:+.2f}% / {r.train_excess:+.2f}% ({r.train_t_excess:.1f}) | '
        f'{r.valid_win:.0f}% / {r.valid_avg:+.2f}% / {r.valid_excess:+.2f}% ({r.valid_t_excess:.1f}) | {r.valid_per_year:.0f} | {hoS} |')

say('\n## 保留期检验（2025–26，谁都没看过）')
for k in (10, 50, 100, 300):
    s = top.head(k)
    s = s[s.ho_n.fillna(0) >= 10]
    if len(s):
        say(f'- 前 {k} 名（保留期 ≥10 笔的 {len(s)} 个）：保留期每笔平均中位数 {s.ho_avg.median():+.2f}%，胜率中位数 {s.ho_win.median():.0f}%，'
            f'每笔为正的占 {(s.ho_avg > 0).mean()*100:.0f}%；保留期超额中位数 {s.ho_excess.median():+.2f}%，超额为正的占 {(s.ho_excess > 0).mean()*100:.0f}%')
if len(rH):
    rr = rH[rH.get('n', pd.Series(0, index=rH.index)).fillna(0) >= 10]
    say(f'- 随机对照（{len(rr)} 个）：保留期每笔平均中位数 {rr.avg.median():+.2f}%，胜率中位数 {rr.win.median():.0f}%，超额中位数 {rr.excess.median():+.2f}%')
say(f'- 排名和保留期超额的相关性（前 300）：ρ = {top[["score", "ho_excess"]].corr("spearman").iloc[0,1]:+.2f}')

say('\n## 按方向汇总（各方向前 100 名）')
for ag, g in R.groupby('agent'):
    g = g.head(100)
    say(f'- {ag}：最高分 {g.score.max():.2f}，前 100 名验证期每笔中位数 {g.valid_avg.median():+.2f}%')

top.to_csv(os.path.join(OUT, 'leaderboard_top300.csv'), index=False, encoding='utf-8-sig')
R.drop(columns=['sealed'], errors='ignore').to_csv(os.path.join(OUT, 'leaderboard_all.csv'), index=False, encoding='utf-8-sig')
open(os.path.join(OUT, 'leaderboard.md'), 'w', encoding='utf-8').write('\n'.join(lines))
