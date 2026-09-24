"""训练数据：S&P 500 时点成分股（Wikipedia 变更表回推）+ 2008–2025 日线。
2026 测试集行情不在这里下载，只由 eval_2026.py 在策略冻结后一次性获取。"""
import io, os
import pandas as pd, requests, yfinance as yf

TRAIN_END = '2025-12-31'
WIKI = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
HIST = 'https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500'   # 变更表所在页


def membership(path='data/membership.csv.gz', snapshot='data/wiki'):
    """每个自然月末的成分股（long 表: month_end, ticker）；原始网页存档以便审计复现"""
    t = []
    for name, url in [('current', WIKI), ('historical', HIST)]:
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30).text
        if snapshot:
            open(f'{snapshot}_{name}.html', 'w', encoding='utf-8').write(html)
        t += pd.read_html(io.StringIO(html))
    fix = lambda s: s.fillna('').astype(str).str.strip().str.replace('.', '-', regex=False)
    flat = lambda df: [' '.join(dict.fromkeys(map(str, c if isinstance(c, tuple) else (c,)))).lower() for c in df.columns]
    for k, df in enumerate(t):
        print(f'table[{k}] {df.shape}:', flat(df)[:8])
    chg = next(df for df in t if any('added' in c for c in flat(df)) and any('removed' in c for c in flat(df)))
    chg.columns = flat(chg)
    print('变更表列名:', list(chg.columns))
    col = lambda k, alt=('',): next(c for c in chg.columns if k in c and any(a in c for a in alt))
    ev = pd.DataFrame({'date': pd.to_datetime(chg[col('date')], format='mixed', errors='coerce'),
                       'add': fix(chg[col('added', ('ticker', 'symbol'))]),
                       'rem': fix(chg[col('removed', ('ticker', 'symbol'))])})
    ev = ev.dropna(subset=['date']).sort_values('date', ascending=False).to_dict('records')
    members, rows, i = set(fix(t[0]['Symbol'])), [], 0
    for me in pd.date_range('2008-01-31', pd.Timestamp.today(), freq='ME')[::-1]:   # 从今天往回撤销变更
        while i < len(ev) and ev[i]['date'] > me:
            members.discard(ev[i]['add'])
            if ev[i]['rem']:
                members.add(ev[i]['rem'])
            i += 1
        rows += [(me.date(), m) for m in sorted(members)]
    df = pd.DataFrame(rows, columns=['month_end', 'ticker']).sort_values(['month_end', 'ticker'])
    if path:
        df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    mem = membership()
    raw = yf.download(sorted(set(mem.ticker)) + ['SPY'], start='2008-01-01', end='2026-01-01',
                      auto_adjust=True, progress=False, threads=True)
    close = raw['Close'].loc[:TRAIN_END].dropna(axis=1, how='all')
    dvol = (raw['Close'] * raw['Volume']).loc[:TRAIN_END, close.columns]   # 成交额：不受拆股复权影响
    close.astype('float32').to_parquet('data/train_close.parquet')
    dvol.astype('float32').to_parquet('data/train_dvol.parquet')
    m = mem[pd.to_datetime(mem.month_end) <= TRAIN_END]
    cov = m.assign(y=pd.to_datetime(m.month_end).dt.year).groupby('y').ticker.apply(lambda s: s.isin(close.columns).mean())
    print(f'{close.shape[1]} tickers, {close.index[0]:%F} ~ {close.index[-1]:%F}')
    print('成分股行情覆盖率（按年）:\n' + cov.round(3).to_string())
