"""v2 机械选择（见 PREREGISTRATION_v2.md）：`python3 select_v2.py`

候选 = research.py 的 C01–C25（原样复用 CFG / strategy()，不复制、不改参数）。
训练期 2010-01-01 ~ 2024-12-31：行情与成分股先截断到 <= 2024-12-31 再回测，结构上看不到 2025 年以后的数据。
规则：训练期「日净收益 − SPY」的 Newey-West t 最大者当选；并列取编号小者。
输出 results/selection_v2.md、results/selection_v2.json。
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib, json, subprocess, time
import multiprocessing as mp
import pandas as pd
from harness import stats, load_members
import research as R

START, END = '2010-01-01', '2024-12-31'
MD, JS = 'results/selection_v2.md', 'results/selection_v2.json'
DATA = None


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def load_train():
    """训练数据：一律截断到 END（含）"""
    cut = pd.Timestamp(END)
    close = pd.read_parquet('data/train_close.parquet').loc[:cut]
    dvol = pd.read_parquet('data/train_dvol.parquet').loc[:cut]
    members = {d: s for d, s in load_members().items() if d <= cut}
    assert close.index[-1] <= cut and dvol.index[-1] <= cut and max(members) <= cut
    return close, dvol, members


def summarize(r, b):
    s, x = stats(r, b), r - b
    return {'年化超额': x.mean() * 252, '跟踪误差': s['跟踪误差'], 'IR': s['IR'], 'NW_t': s['NW_t'], '单侧p': s['单侧p'],
            '交易日数': len(r), '首日': f'{r.index[0]:%F}', '末日': f'{r.index[-1]:%F}'}


def job(cid):
    r, b = R.run(R.CFG[cid], (START, END), DATA)
    assert r.index[-1] <= pd.Timestamp(END)
    return cid, summarize(r, b)


def rank(res):
    """NW t 降序；并列按编号升序"""
    return sorted(res, key=lambda c: (-res[c]['NW_t'], c))


if __name__ == '__main__':
    t0 = time.time()
    DATA = load_train()
    with mp.get_context('fork').Pool() as pool:
        res = dict(pool.map(job, list(R.CFG)))
    order = rank(res)
    sel = order[0]
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = {'selected': sel, 'selected_name': R.NAME[sel], 'selected_cfg': repr(R.CFG[sel]),
            'rule': '训练期 2010-01-01~2024-12-31「日净收益 − SPY」Newey-West t 最大者；并列取编号小者',
            'train': [START, END], 'ranking': order, 'train_stats': res,
            'research_sha256': sha('research.py'), 'harness_sha256': sha('harness.py'),
            'git_head': head, 'data_last_day_used': f'{DATA[0].index[-1]:%F}'}
    os.makedirs('results', exist_ok=True)
    json.dump(meta, open(JS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    rows = '\n'.join(f"| {k} | {c} | {R.NAME[c]} | {res[c]['年化超额']:.4f} | {res[c]['跟踪误差']:.4f} | {res[c]['IR']:.3f} | "
                     f"{res[c]['NW_t']:.3f} | {res[c]['单侧p']:.4f} |" for k, c in enumerate(order, 1))
    s = res[sel]
    open(MD, 'w', encoding='utf-8').write(f"""# v2 机械选择结果（训练期 2010–2024）

- 规则（PREREGISTRATION_v2.md）：{meta['rule']}
- 训练区间：{s['首日']} ~ {s['末日']}（{s['交易日数']} 个交易日）；回测前数据已截断到 {END}（实际用到的最后一天 {meta['data_last_day_used']}）
- 引擎：harness.backtest（月末收盘调仓、时点成分股、单边 10bp）；年化超额 = 日均(策略 − SPY)×252
- research.py sha256：`{meta['research_sha256']}`　harness.py sha256：`{meta['harness_sha256']}`　代码提交：`{head}`

**选中：{sel}（{R.NAME[sel]}）** —— 训练期 年化超额 {s['年化超额']:.2%}，跟踪误差 {s['跟踪误差']:.2%}，IR {s['IR']:.3f}，NW t {s['NW_t']:.3f}，单侧 p {s['单侧p']:.4f}

| 排名 | 配置 | 说明 | 年化超额 | 跟踪误差 | IR | NW_t | 单侧p |
|---|---|---|---|---|---|---|---|
{rows}
""")
    print(open(MD, encoding='utf-8').read())
    print(f'耗时 {time.time() - t0:.0f} 秒')
