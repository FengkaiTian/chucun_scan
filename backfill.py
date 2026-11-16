"""
补算缺失日子的选股 —— 扫描脚本停跑期间，用历史行情按当时能看到的数据逐日重放。

用法：python backfill.py            补算「最后一条记录之后」到最近一个完整交易日
      python backfill.py 2026-06-15  从指定的信号日开始补算（会覆盖该日之后的补算记录）

做法和 daily_scan.py 完全一致：同样的条件、组合、VIX≥30 暂停、14 天内不重复推荐。
每个信号日只用截至当天收盘的数据（最近约 15 个月），不偷看未来。
局限：成分股用的是今天的标普500名单，期间被剔除的股票不在其中。
"""
import sys, json
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import daily_scan as ds

WINDOW = 315   # 约 15 个月的交易日，对应每日扫描的 period='15mo'


def main():
    with open(ds.COMBO_FILE, 'r', encoding='utf-8') as f:
        combos = [c for c in json.load(f) if c['grade'] in ('SSS', 'S', 'A', 'B')]

    history = ds.load_history()
    if not history:
        history = ds.git_history_picks()
        print(f"从 git 历史迁移 {len(history)} 条旧记录")

    today = pd.Timestamp(datetime.now().date())
    if len(sys.argv) > 1:
        first_signal_day = pd.Timestamp(sys.argv[1])
    else:
        # 最后一条记录的入场日当天收盘后，才会产生下一个信号
        first_signal_day = pd.Timestamp(max(p['scan_date'] for p in history))
    history = [p for p in history
               if not (p.get('source') == 'backfill' and pd.Timestamp(p['scan_date']) > first_signal_day)]

    print("获取标普500成分股并下载 3 年行情...")
    sector_map, _ = ds.get_sp500_info()
    tickers = list(sector_map.keys())
    data = yf.download(tickers, period='3y', group_by='ticker',
                       auto_adjust=True, threads=True, progress=False)
    vix = yf.download('^VIX', period='3y', auto_adjust=True, progress=False)['Close'].squeeze()
    frames = {t: data[t].dropna() for t in tickers}

    calendar = data[tickers[0]].dropna(how='all').index
    calendar = calendar[calendar < today]                 # 不用今天未收盘的 K 线
    signal_days = calendar[calendar >= first_signal_day]
    print(f"补算 {len(signal_days)} 个信号日：{signal_days[0].date()} → {signal_days[-1].date()}")

    added = 0
    for i, day in enumerate(signal_days):
        pos = calendar.get_loc(day)
        if pos + 1 < len(calendar):
            entry = calendar[pos + 1]
        else:                                             # 最后一个信号日：入场日是下一个工作日
            entry = day + pd.offsets.BDay(1)
        entry_str = entry.strftime('%Y-%m-%d')
        if any(p['scan_date'] == entry_str and p.get('source') == 'live' for p in history):
            continue                                      # 这天当时真的扫过，以当时的记录为准

        vix_day = vix.loc[:day]
        if len(vix_day) and float(vix_day.iloc[-1]) >= 30:
            print(f"  {day.date()}  VIX≥30，暂停")
            continue

        cutoff = (entry - timedelta(days=14)).strftime('%Y-%m-%d')
        recent = {p['ticker'] for p in history if cutoff <= p['scan_date'] < entry_str}

        picks = []
        for t in tickers:
            td = frames[t].loc[:day].tail(WINDOW)
            if len(td) < 260 or td.index[-1] != day or t in recent:
                continue
            try:
                result = ds.compute_conditions(td['Close'], td['High'], td['Low'], td['Volume'], td['Open'])
            except Exception:
                continue
            if result is None:
                continue
            grade, combo = ds.match_grade(result[0], combos)
            if grade:
                picks.append({'ticker': t, 'scan_date': entry_str, 'grade': grade,
                              'wr5': combo['wr5'], 'source': 'backfill'})
        history += picks
        added += len(picks)
        if picks:
            print(f"  {day.date()} → 入场 {entry_str}：{' '.join(p['ticker'] for p in picks)}")

    ds.save_history(history)
    print(f"\n完成：新增 {added} 条补算记录，picks_history.json 共 {len(history)} 条")


if __name__ == '__main__':
    main()
