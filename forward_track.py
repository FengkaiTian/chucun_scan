"""前瞻模拟（规则于 2026-11-18 冻结，见 analysis/rebuild/FROZEN_RULES.md）。

每次运行：
  1. 用当前标普500 成分股和最近 15 个月日线，找出今天收盘满足 R1 / R2 的股票，追加到 forward_log.csv
     （同一股票 20 个交易日内只记一次，和回测一致）。
  2. 给已满 20 个交易日的记录补上结果：次日收盘入场、持有 20 个交易日的收益，以及同期 SPY（含分红）收益。
  3. 打印累计成绩。到复核日期前不要改规则。

用法：python forward_track.py          （建议每个交易日收盘后跑一次，和 daily_scan.py 一起）
"""
import os, sys
from datetime import datetime
import numpy as np, pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from daily_scan import get_sp500_info

LOG = os.path.join(HERE, 'forward_log.csv')
HOLD = 20
FROZEN = '2026-11-18'
COLS = ['signal_date', 'ticker', 'rule', 'close', 'dist_ma200', 'vol_ratio', 'entry_date', 'entry_px',
        'exit_date', 'exit_px', 'ret', 'spy_ret', 'excess_net']


def main():
    log = pd.read_csv(LOG, dtype={'ticker': str}) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
    sector, _ = get_sp500_info()
    tickers = sorted(sector)
    px = yf.download(tickers + ['SPY'], period='15mo', auto_adjust=True, group_by='ticker', threads=True, progress=False)
    C = pd.DataFrame({t: px[t]['Close'] for t in tickers + ['SPY'] if t in px.columns.get_level_values(0)})
    V = pd.DataFrame({t: px[t]['Volume'] for t in tickers if t in px.columns.get_level_values(0)})
    C = C.dropna(how='all')
    # 美股还没收盘时，yfinance 会给出当天的半截日线（成交量偏低，会误判"缩量"），去掉
    import zoneinfo
    ny = datetime.now(zoneinfo.ZoneInfo('America/New_York'))
    if C.index[-1].date() == ny.date() and (ny.hour, ny.minute) < (16, 30):
        C = C.iloc[:-1]
        print(f'美股尚未收盘（纽约 {ny:%H:%M}），只用到 {C.index[-1].date()} 的完整日线')
    V = V.reindex(C.index)
    today = C.index[-1]

    # 1. 今天的新信号
    ma200 = C[V.columns].rolling(200).mean()
    vprior = V.shift(1).rolling(20).mean()
    last_c, last_ma, last_v, last_vp = C[V.columns].iloc[-1], ma200.iloc[-1], V.iloc[-1], vprior.iloc[-1]
    deep = (last_c < 0.75 * last_ma) & last_ma.notna()
    shrink = (last_v > 0) & (last_v < 0.70 * last_vp)
    recent = set()
    if len(log):
        d = pd.to_datetime(log.signal_date)
        cutoff = C.index[max(0, len(C) - HOLD)]
        recent = set(log.loc[d >= cutoff, 'ticker'])
    new = []
    for t in deep[deep].index:
        if t in recent or str(today.date()) in set(log.loc[log.ticker == t, 'signal_date']):
            continue
        new.append({'signal_date': str(today.date()), 'ticker': t, 'rule': 'R2' if shrink.get(t, False) else 'R1',
                    'close': round(float(last_c[t]), 4), 'dist_ma200': round(float(last_c[t] / last_ma[t] - 1) * 100, 1),
                    'vol_ratio': round(float(last_v[t] / last_vp[t]), 2) if last_vp[t] > 0 else np.nan})
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)

    # 2. 补结果
    idx = {d.date().isoformat(): i for i, d in enumerate(C.index)}
    for k, r in log.iterrows():
        if pd.notna(r.get('ret')) or r.signal_date not in idx:
            continue
        i = idx[r.signal_date]
        if i + 1 + HOLD >= len(C) or r.ticker not in C.columns:
            continue
        e, x = i + 1, i + 1 + HOLD
        pe, pxx = C[r.ticker].iloc[e], C[r.ticker].iloc[x]
        se, sx = C['SPY'].iloc[e], C['SPY'].iloc[x]
        if not (np.isfinite(pe) and np.isfinite(pxx)):
            continue
        ret = (pxx / pe - 1) * 100; spy = (sx / se - 1) * 100
        log.loc[k, ['entry_date', 'entry_px', 'exit_date', 'exit_px', 'ret', 'spy_ret', 'excess_net']] = [
            str(C.index[e].date()), round(pe, 4), str(C.index[x].date()), round(pxx, 4), round(ret, 2), round(spy, 2),
            round(ret - 0.10 - spy, 2)]
    log[COLS].to_csv(LOG, index=False)

    # 3. 汇总
    print(f'{today.date()}：新信号 {len(new)} 条（' + ', '.join(f"{n['ticker']}[{n['rule']}]" for n in new) + '）')
    done = log[pd.to_numeric(log.ret, errors='coerce').notna()]
    for rule, g in [('R1+R2 全部', done), ('R2 深跌+缩量', done[done.rule == 'R2']), ('R1 只深跌', done[done.rule == 'R1'])]:
        if len(g):
            e = pd.to_numeric(g.excess_net)
            print(f'  {rule}：已出结果 {len(g)} 笔，胜率 {(pd.to_numeric(g.ret) - 0.10 > 0).mean()*100:.0f}%，'
                  f'每笔净收益 {(pd.to_numeric(g.ret) - 0.10).mean():+.2f}%，每笔净超额 {e.mean():+.2f}%')
    print(f'规则冻结于 {FROZEN}；复核日期见 analysis/rebuild/FROZEN_RULES.md')


if __name__ == '__main__':
    main()
