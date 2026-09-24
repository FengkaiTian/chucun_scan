"""v2 样本外检验（PREREGISTRATION_v2.md）——正式运行只允许一次，结果文件已存在即拒绝重跑。

主判据：results/selection_v2.json 机械选中的配置，2025-01-01 ~ 最后一个完整交易日，
        「日净收益 − SPY」Newey-West 单侧 p < 0.05 且均值 > 0 ⇒ 成功。
次要（仅报告，不作为主判据）：25 个配置单侧 p 值的 Holm 校正（族错误率 5%）。
数据处理（事先规定，与策略结果无关）：
  - 下载截止到运行当天之前（盘中价不是收盘价），去掉 SPY 缺失的日子；
  - Yahoo 批量下载偶尔整天缺一批股票：对有"中间缺口"的股票最多重下 3 轮；
  - 仍有缺口时，剔除「中间缺口占比比前后 11 个交易日中位数高 10 个百分点以上」的日子（收益并入下一交易日），并在报告中列出。
用法：python eval_v2.py         正式运行（需联网，由 .github/workflows/test_v2.yml 执行）
      python eval_v2.py --dry   离线干跑：data/*.parquet 截断到 2024-12-31，测试期换成 2023-01-01~2024-12-31，只写 /tmp/eval_v2_dry/
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib, json, subprocess, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
from harness import stats, load_members
import research as R

DRY = '--dry' in sys.argv
OUT_DIR = '/tmp/eval_v2_dry' if DRY else 'results'
OUT = f'{OUT_DIR}/test_v2.md'
# 测试起点 / 下载起点（保证 600 日回看）/ 股票池取"该月末及以后"的成分股并集
TEST_START, DL_START, UNI_FROM = ('2023-01-01', '2021-01-01', '2022-12-31') if DRY else ('2025-01-01', '2022-01-01', '2024-12-31')
SEL, ALPHA = 'results/selection_v2.json', 0.05
DATA, END = None, None


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def holes(c):
    """前后都有数据的中间缺口"""
    return c.isna() & c.ffill().notna() & c.bfill().notna()


def download():
    import yfinance as yf
    from data_pipeline import membership
    mem = membership(path=None, snapshot='results/wiki_v2')
    members = {pd.Timestamp(d): set(g.ticker) for d, g in mem.groupby('month_end')}
    tickers = sorted(set().union(*(v for d, v in members.items() if d >= pd.Timestamp(UNI_FROM))))
    end = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')          # yfinance 的 end 不含当天
    get = lambda t: yf.download(t, start=DL_START, end=end, auto_adjust=True, progress=False, threads=True)
    raw = get(tickers + ['SPY'])
    C, V = raw['Close'], raw['Volume']
    for k in range(3):
        bad = sorted(C.columns[holes(C).any()])
        print(f'第 {k + 1} 轮补缺：{len(bad)} 只有中间缺口')
        if not bad:
            break
        for i in range(0, len(bad), 100):
            r = get(bad[i:i + 100])
            C = C.combine_first(r['Close'].reindex(columns=C.columns))
            V = V.combine_first(r['Volume'].reindex(columns=V.columns))
    close = C.dropna(axis=1, how='all')
    return close, (C * V)[close.columns], members, tickers


def dry_data():
    cut = pd.Timestamp('2024-12-31')
    members = {d: s for d, s in load_members().items() if d <= cut}
    tickers = sorted(set().union(*(v for d, v in members.items() if d >= pd.Timestamp(UNI_FROM))))
    close = pd.read_parquet('data/train_close.parquet').loc[DL_START:cut]
    close = close[[t for t in tickers if t in close.columns] + ['SPY']].dropna(axis=1, how='all')
    dvol = pd.read_parquet('data/train_dvol.parquet').loc[DL_START:cut, close.columns]
    assert close.index[-1] <= cut and max(members) <= cut
    return close, dvol, members, tickers


def clean(close, dvol):
    close = close[close['SPY'].notna()]
    miss = holes(close.drop(columns='SPY')).mean(axis=1)
    bad = list(miss.index[miss - miss.rolling(11, center=True, min_periods=1).median() > 0.1])
    close = close.drop(bad)
    return close, dvol.loc[close.index], [f'{d:%F}（缺 {miss[d]:.0%}）' for d in bad]


def init(data, end):
    global DATA, END
    DATA, END = data, end


def job(cid):
    t = time.time()
    r, b = R.run(R.CFG[cid], (TEST_START, END), DATA)
    return cid, r, b, time.time() - t


def holm(p):
    """Holm 校正后的 p 值：p_adj(k) = max_{j<=k} min(1, (m-j+1)·p(j))"""
    q = p.sort_values()
    return (q * np.arange(len(q), 0, -1)).cummax().clip(upper=1).reindex(p.index)


if __name__ == '__main__':
    t0 = time.time()
    if not DRY and os.path.exists(OUT):
        sys.exit(f'{OUT} 已存在：v2 测试只允许运行一次。')
    sel = json.load(open(SEL, encoding='utf-8'))
    for f, k in [('research.py', 'research_sha256'), ('harness.py', 'harness_sha256')]:
        if sha(f) != sel[k]:
            sys.exit(f'{f} 自机械选择以来被修改（sha256 与 {SEL} 不符），拒绝运行。')
    S = sel['selected']
    os.makedirs(OUT_DIR, exist_ok=True)
    close, dvol, members, tickers = dry_data() if DRY else download()
    close, dvol, bad = clean(close, dvol)
    DATA, END = (close, dvol, members), close.index[-1]
    t1 = time.time()
    # spawn：下载阶段 yfinance 起过线程，避免在多线程进程里 fork
    with mp.get_context('spawn').Pool(initializer=init, initargs=(DATA, END)) as pool:
        out = pool.map(job, list(R.CFG))
    t2 = time.time()
    rets = pd.DataFrame({c: r for c, r, b, _ in out})
    bench = out[0][2]
    st = pd.DataFrame({c: {**stats(rets[c], bench), '年化超额': (rets[c] - bench).mean() * 252} for c in rets}).T
    st['Holm_p'] = holm(st['单侧p'].astype(float))
    ok = bool(st.loc[S, '单侧p'] < ALPHA and st.loc[S, '年化超额'] > 0)
    rej = [c for c in st.index if st.loc[c, 'Holm_p'] < ALPHA and st.loc[c, '年化超额'] > 0]

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    tr = sel['train_stats'][S]
    main = '\n'.join(f'| {k} | {v:.4f} |' for k, v in st.loc[S].items() if k != 'Holm_p')
    rows = '\n'.join(f"| {'**' + c + '**' if c == S else c} | {R.NAME[c]} | {x['年化超额']:.4f} | {x['跟踪误差']:.4f} | "
                     f"{x['IR']:.3f} | {x['NW_t']:.3f} | {x['单侧p']:.4f} | {x['Holm_p']:.4f} |" for c, x in st.iterrows())
    title = ('# v2 干跑（离线，非正式）：data/*.parquet 截断到 2024-12-31，测试期替换为 2023-01-01 ~ 2024-12-31（属训练期内）'
             if DRY else '# v2 样本外检验结果（训练 2010–2024 / 测试 2025–2026）')
    report = f"""{title}

- 协议：PREREGISTRATION_v2.md；引擎 harness.backtest（月末收盘调仓、时点成分股、单边 10bp）
- 测试区间：{rets.index[0]:%Y-%m-%d} ~ {END:%Y-%m-%d}（{len(rets)} 个交易日）；运行于 {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC
- 被检验策略（机械选择，{SEL}）：**{S}**（{R.NAME[S]}），配置 `{sel['selected_cfg']}`
  训练期 {sel['train'][0]} ~ {sel['train'][1]}：年化超额 {tr['年化超额']:.2%}，IR {tr['IR']:.3f}，NW t {tr['NW_t']:.3f}
- research.py sha256：`{sha('research.py')}`（与选择时一致）　harness.py sha256：`{sha('harness.py')}`
- {SEL} sha256：`{sha(SEL)}`　选择时提交：`{sel['git_head']}`　本次代码提交：`{head}`
- 行情覆盖：{close.shape[1] - 1}/{len(tickers)} 只（{UNI_FROM[:7]} 及以后各月成分股并集），下载起点 {DL_START}
- 数据处理：剔除的数据缺陷日 {'、'.join(bad) if bad else '无'}；不含运行当天
- 耗时：回测 {t2 - t1:.0f} 秒（25 个配置，多进程），全程 {time.time() - t0:.0f} 秒

## 主判据：{S}

| 指标 | 值 |
|---|---|
{main}

**结论：{'✅ 显著跑赢标普 500（单侧 p < 0.05 且均值 > 0）' if ok else '❌ 未能显著跑赢标普 500'}**

## 全部 25 个配置（次要，仅报告；粗体为被检验策略）

| 配置 | 说明 | 年化超额 | 跟踪误差 | IR | NW_t | 单侧p | Holm校正p |
|---|---|---|---|---|---|---|---|
{rows}

## Holm 校正（次要结果，不作为主判据）

族错误率 {ALPHA:.0%}、m = {len(st)}：{'、'.join(rej) + ' 在校正后显著' if rej else '没有任何配置在校正后显著'}。

## 证据力度的限制
见 PREREGISTRATION_v2.md「必须事先披露的污染」：2026 年是第二次使用（v1 已检验 C20）；2025 年曾在 v1 验证期内；
协调者发起 v2 时已知 v1 的 2026 结果。
"""
    open(OUT, 'w', encoding='utf-8').write(report)
    pd.concat([rets, bench.rename('SPY')], axis=1).to_csv(f'{OUT_DIR}/test_v2_daily.csv')
    print(report)
    print('各配置回测耗时（秒）：', {c: round(t, 1) for c, _, _, t in out})
