"""择时研究数据（只含美股标普 500）：日度 DataFrame[tr 总收益指数, ret 当日总收益, rf 当日无风险收益]
  'spy'  : SPY 复权收盘（含分红）——训练/验证 1998–2025、2026 参考
  'index': ^GSPC 价格 + Shiller 月度股息率按交易日平摊——样本前独立测试；Shiller 不可得时按预注册
           退回 Kenneth French 数据库的美股市场总收益
rf 取 ^IRX（13 周美债）；1960 年前无数据的预热期记 0。仓库里只存 1998–2025 训练集。"""
import io, zipfile
import pandas as pd, requests, yfinance as yf

UA = {'User-Agent': 'Mozilla/5.0'}
SHILLER = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'
FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'


def close(ticker, start, end, adjust):
    return yf.download(ticker, start=start, end=end, auto_adjust=adjust, progress=False)['Close'].squeeze().dropna()


def shiller_yield():
    raw = pd.read_excel(io.BytesIO(requests.get(SHILLER, headers=UA, timeout=60).content), sheet_name='Data', header=None)
    h = raw.index[raw.iloc[:, 0].astype(str).str.strip() == 'Date'][0]
    x = raw.iloc[h + 1:, :3].apply(pd.to_numeric, errors='coerce').dropna()
    y = x[0].astype(int)
    m = ((x[0] - y) * 100).round().astype(int)                  # 1871.01 → 1 月，1871.1 → 10 月
    idx = pd.to_datetime(pd.DataFrame({'year': y, 'month': m, 'day': 1})) + pd.offsets.MonthEnd(0)
    return pd.Series((x[2] / x[1]).values, idx).sort_index()    # 年化股息率 D/P


def french_market():
    z = zipfile.ZipFile(io.BytesIO(requests.get(FRENCH, headers=UA, timeout=60).content))
    rows = [l for l in z.read(z.namelist()[0]).decode('latin1').splitlines() if l.split(',')[0].strip().isdigit()]
    df = pd.read_csv(io.StringIO('\n'.join(rows)), header=None).set_index(0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m%d')
    return (df[1] + df[4]) / 100                                # Mkt-RF + RF


def series(start, end, source='spy'):
    rf = (close('^IRX', start, end, False) / 100 / 252).rename('rf')
    if source == 'spy':
        ret, used = close('SPY', start, end, True).pct_change(), 'SPY 复权'
    else:
        try:
            px = close('^GSPC', start, end, False)
            ret, used = px.pct_change() + shiller_yield().reindex(px.index, method='ffill') / 252, '^GSPC + Shiller 股息'
        except Exception as e:
            print(f'Shiller 不可得（{e}），按预注册退回 French 市场总收益')
            ret, used = french_market().loc[start:end], 'French 美股市场总收益（预注册备选）'
    df = pd.DataFrame({'ret': ret}).join(rf, how='left').dropna(subset=['ret'])
    df['rf'] = df['rf'].ffill().fillna(0)
    df['tr'] = (1 + df['ret']).cumprod()
    df.attrs['source'] = used
    return df[['tr', 'ret', 'rf']]


if __name__ == '__main__':
    df = series('1998-01-01', '2026-01-01', 'spy')
    df.to_parquet('data/timing_train.parquet')
    print(df.attrs['source'], df.index[0].date(), df.index[-1].date(), len(df))
    print(df.describe().to_string())
