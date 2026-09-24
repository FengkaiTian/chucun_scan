"""在 GitHub Actions 中运行（本地容器无法访问 Yahoo）：下载历届成分股 + SPY 的复权日线 OHLCV 与 3 个月国债利率。"""
import pandas as pd, yfinance as yf
from horse_race.universe import membership

if __name__ == '__main__':
    end = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')      # 不含当日（盘中价不是收盘价）
    tick = list(membership(pd.bdate_range('2008-01-01', end)).columns)
    raw = yf.download(tick + ['SPY'], start='2007-01-01', end=end, auto_adjust=True, progress=False, threads=True)
    close = raw['Close'].dropna(axis=1, how='all').dropna(subset=['SPY'])
    for f in ['Close', 'High', 'Low', 'Volume']:
        raw[f].loc[close.index, close.columns].astype('float32').to_parquet(f'data/hr_{f.lower()}.parquet')
    irx = yf.download('^IRX', start='2007-01-01', end=end, auto_adjust=False, progress=False)['Close'].squeeze()
    irx.rename('irx').to_frame().to_parquet('data/hr_irx.parquet')
    miss = sorted(set(tick) - set(close.columns))
    print(f'{close.shape[1]} tickers, {close.index[0]:%F} ~ {close.index[-1]:%F}; 无数据 {len(miss)} 只')
