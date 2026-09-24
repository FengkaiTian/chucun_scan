"""在 GitHub Actions 中运行（本地容器无法访问 Yahoo）：下载历届成分股 + SPY 的复权日线 OHLCV 与 3 个月国债利率。
Yahoo 批量下载偶尔整天缺一批股票（2026-09-22 曾缺 443 只）：对"前后都有数据的中间缺口"所在股票重下一次补齐。"""
import pandas as pd, yfinance as yf
from horse_race.universe import membership

F = ['Close', 'High', 'Low', 'Volume']


def get(tick, end):
    raw = yf.download(tick, start='2007-01-01', end=end, auto_adjust=True, progress=False, threads=True)
    return {f: raw[f] for f in F}


def holes(c):
    return c.isna() & c.ffill().notna() & c.bfill().notna()


if __name__ == '__main__':
    end = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')      # 不含当日（盘中价不是收盘价）
    tick = list(membership(pd.bdate_range('2008-01-01', end)).columns)
    d = get(tick + ['SPY'], end)
    for k in range(3):
        h = holes(d['Close'])
        bad = sorted(h.columns[h.any()])
        print(f'第 {k + 1} 轮：{int(h.values.sum())} 个中间缺口，涉及 {len(bad)} 只；最多的日期 {h.sum(1).nlargest(3).to_dict()}')
        if not bad:
            break
        for i in range(0, len(bad), 100):
            r = get(bad[i:i + 100], end)
            for f in F:
                d[f] = d[f].combine_first(r[f].reindex(columns=d[f].columns))
    close = d['Close'].dropna(axis=1, how='all').dropna(subset=['SPY'])
    for f in F:
        d[f].loc[close.index, close.columns].astype('float32').to_parquet(f'data/hr_{f.lower()}.parquet')
    irx = yf.download('^IRX', start='2007-01-01', end=end, auto_adjust=False, progress=False)['Close'].squeeze()
    irx.rename('irx').to_frame().to_parquet('data/hr_irx.parquet')
    miss = sorted(set(tick) - set(close.columns))
    print(f'{close.shape[1]} tickers, {close.index[0]:%F} ~ {close.index[-1]:%F}; 无数据 {len(miss)} 只')
