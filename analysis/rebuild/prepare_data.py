"""20 年数据准备：时点成分股 + 日线 OHLCV + 基础信号 + 37 个条件 + 远期收益。

不改动 daily_scan 的任何逻辑：指标直接调用 daily_scan 里的 calc_* 函数，
37 个条件按 compute_conditions 的定义逐条向量化，verify_conditions.py 抽样与原函数逐条核对。

用法：python prepare_data.py            全部（成分股 → 下载 → 计算）
      python prepare_data.py --no-download  只重算（用已下载的行情）
输出 data/：membership.csv.gz、ohlcv_*.parquet、signals_20y.parquet、ew_index.parquet、coverage.csv
"""
import io, os, sys, time, warnings
import numpy as np, pandas as pd, requests, yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import daily_scan as ds

warnings.filterwarnings('ignore')
if '--from2000' in sys.argv:  # 2000 年起的版本：python prepare_data.py --from2000
    DATA = os.path.join(HERE, 'data2000')
    START, FIRST_MONTH, SIG_START = '1999-01-01', '1999-12-31', '2000-01-01'
else:
    DATA = os.path.join(HERE, 'data')
    START = '2005-01-01'          # 指标预热约 1 年
    FIRST_MONTH = '2005-12-31'    # 成分股从这个月末开始，信号从 2006 年起
    SIG_START = '2006-01-01'
FIELDS = ['Open', 'High', 'Low', 'Close', 'Volume']
WIKI = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
HIST = 'https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500'
COND_NAMES = list(ds.COND_LABEL)


# ── 1. 时点成分股：今天的名单 + 维基变更表往回撤销 ─────────────────
def membership():
    tables = []
    for name, url in [('current', WIKI), ('historical', HIST)]:
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30).text
        open(os.path.join(DATA, f'wiki_{name}.html'), 'w', encoding='utf-8').write(html)
        tables += pd.read_html(io.StringIO(html))
    fix = lambda s: s.fillna('').astype(str).str.strip().str.replace('.', '-', regex=False)
    flat = lambda df: [' '.join(dict.fromkeys(map(str, c if isinstance(c, tuple) else (c,)))).lower() for c in df.columns]
    chg = next(df for df in tables if any('added' in c for c in flat(df)) and any('removed' in c for c in flat(df)))
    chg.columns = flat(chg)
    col = lambda k, alt=('',): next(c for c in chg.columns if k in c and any(a in c for a in alt))
    ev = pd.DataFrame({'date': pd.to_datetime(chg[col('date')], format='mixed', errors='coerce'),
                       'add': fix(chg[col('added', ('ticker', 'symbol'))]),
                       'rem': fix(chg[col('removed', ('ticker', 'symbol'))])})
    ev = ev.dropna(subset=['date']).sort_values('date', ascending=False).to_dict('records')
    current = tables[0]
    members, rows, i = set(fix(current['Symbol'])), [], 0
    for me in pd.date_range(FIRST_MONTH, pd.Timestamp.today(), freq='ME')[::-1]:
        while i < len(ev) and ev[i]['date'] > me:
            members.discard(ev[i]['add'])
            if ev[i]['rem']:
                members.add(ev[i]['rem'])
            i += 1
        rows += [(me, m) for m in sorted(members) if m]
    mem = pd.DataFrame(rows, columns=['month_end', 'ticker']).sort_values(['month_end', 'ticker'])
    mem.to_csv(os.path.join(DATA, 'membership.csv.gz'), index=False)
    sectors = dict(zip(fix(current['Symbol']), current['GICS Sector']))
    pd.Series(sectors, name='sector').to_csv(os.path.join(DATA, 'sectors_current.csv'))
    return mem


# ── 2. 下载行情（分批，失败的再试一次） ─────────────────────────────
def download(tickers):
    got = {f: {} for f in FIELDS}

    def batch(chunk):
        raw = yf.download(chunk, start=START, auto_adjust=True, group_by='ticker',
                          threads=True, progress=False)
        for t in chunk:
            try:
                df = raw[t] if len(chunk) > 1 else raw
                df = df.dropna(subset=['Close'])
            except Exception:
                continue
            if len(df) < 30:
                continue
            for f in FIELDS:
                got[f][t] = df[f]

    todo = list(tickers)
    for attempt in range(2):
        for k in range(0, len(todo), 80):
            batch(todo[k:k + 80])
            print(f'  下载 {min(k + 80, len(todo))}/{len(todo)}（第 {attempt + 1} 轮）', flush=True)
        todo = [t for t in todo if t not in got['Close']]
        if not todo:
            break
        time.sleep(5)
    for f in FIELDS:
        pd.DataFrame(got[f]).sort_index().to_parquet(os.path.join(DATA, f'ohlcv_{f.lower()}.parquet'))
    for idx in ['^GSPC', '^VIX']:
        s = yf.download(idx, start=START, auto_adjust=True, progress=False)['Close'].squeeze()
        s.rename(idx).to_frame().to_parquet(os.path.join(DATA, f'index_{idx[1:].lower()}.parquet'))
    print(f'  成功 {len(got["Close"])} 只，失败 {len(todo)} 只（多为已退市）')
    pd.Series(todo, name='ticker').to_csv(os.path.join(DATA, 'download_failed.csv'), index=False)


# ── 3. 向量化条件（与 compute_conditions 逐条对应） ──────────────────
def conditions_frame(o, h, l, c, v):
    """返回整段历史每一天的 base 信号 + 37 个条件（bool）+ 辅助数值。"""
    rsi = ds.calc_rsi(c)
    _, _, hist = ds.calc_macd(c)
    r1 = rsi.shift(1)
    base = (hist < 0) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2)) & (r1 < 30) & (rsi > r1)
    base &= pd.Series(np.arange(len(c)) >= 259, index=c.index)          # compute_conditions 要求 ≥260 根

    mfi = ds.calc_mfi(h, l, c, v)
    wr = ds.calc_williams_r(h, l, c)
    k, d = ds.calc_stoch_rsi(c)
    cci = ds.calc_cci(h, l, c)
    cmf = ds.calc_cmf(h, l, c, v)
    obv = ds.calc_obv(c, v)
    kelt = ds.calc_keltner_lower(c, h, l)
    ham = ds.is_hammer(o, h, l, c).astype(bool)
    vma20 = v.rolling(20).mean()
    ma200 = c.rolling(200).mean()
    ma50 = c.rolling(50).mean()
    atr = ds.calc_atr(h, l, c, 14)
    atr_ma20 = atr.rolling(20).mean()
    adx = ds.calc_adx(h, l, c, 14)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_width = (4 * bb_std) / bb_mid.replace(0, 1e-10)
    bb_width_ma = bb_width.rolling(60).mean()
    bb_lower = bb_mid - 2 * bb_std
    c1, c2, c3, o1, l1 = c.shift(1), c.shift(2), c.shift(3), o.shift(1), l.shift(1)
    dist50 = (c / ma50 - 1) * 100
    vpos = vma20 > 0

    # RSI 底背离：过去 20 根（不含当天）的最低收盘价及其当时的 RSI
    cv, rv = c.values, rsi.values
    div = np.zeros(len(c), dtype=bool)
    if len(c) > 20:
        win = sliding_window_view(cv[:-1], 20)                   # win[j] = cv[j:j+20]，对应当天 idx = j+20
        j = np.argmin(win, axis=1)
        pos = np.arange(len(win)) + j
        idx = np.arange(20, len(c))
        div[20:] = (cv[idx] <= win.min(axis=1) * 1.02) & (rv[idx] > rv[pos] + 3)

    conds = {
        'cci_lt100': cci < -100,
        'stoch_cross': (k > d) & (k.shift(1) <= d.shift(1)),
        'near_52w': (c / c.rolling(252).min() - 1) < 0.05,
        'rsi_lt25': r1 < 25,
        'rsi_lt20': r1 < 20,
        'cmf_pos': cmf > 0,
        'below_kelt': c < kelt,
        'wr_lt80': wr < -80,
        'mfi_lt20': mfi < 20,
        'mfi_lt30': mfi < 30,
        'cmf_rising': cmf > cmf.shift(1),
        'obv_rising': obv > obv.rolling(5).mean(),
        'stoch_rising': k > k.shift(1),
        'hammer': ham | ham.shift(1, fill_value=False),
        'above_ma200': c > ma200,
        'below_bb': c < bb_lower,
        'vol_gt12': vpos & (v > vma20 * 1.2),
        'dist_ma50_lt_m10': dist50 < -10,
        'dist_ma200_lt_m15': (c / ma200 - 1) * 100 < -15,
        'ret5d_lt_m3': (c / c.shift(5) - 1) * 100 < -3,
        'ret10d_lt_m5': (c / c.shift(10) - 1) * 100 < -5,
        'cci_rising': cci > cci.shift(1),
        'mfi_rising': mfi > mfi.shift(1),
        'wr_cross80': (wr > -80) & (wr.shift(1) < -80),
        'vol_lt08': vpos & (v < vma20 * 0.8),
        'bullish_engulf': (c > o) & (np.maximum(o, c) > np.maximum(o1, c1))
                          & (np.minimum(o, c) < np.minimum(o1, c1)) & (c1 < o1),
        'rsi_div': pd.Series(div, index=c.index),
        'gap_down_close': (o < c1 * 0.99) & (c > o),
        'ret3d_lt_m5': (c / c3 - 1) * 100 < -5,
        'doji_bottom': ((h - l) > 0) & ((c - o).abs() / (h - l) < 0.2) & (l < l1),
        'adx_lt25': adx < 25,
        'bb_squeeze': bb_width < bb_width_ma * 0.8,
        'vol_spike2x': vpos & (v > vma20 * 2.0),
        'atr_low': atr < atr_ma20 * 0.8,
        'near_ma50': (dist50 > -5.0) & (dist50 < -1.0),
        'wr_rising3': (wr > wr.shift(1)) & (wr.shift(1) > wr.shift(2)),
        'consec_down3': (c < c1) & (c1 < c2) & (c2 < c3),
    }
    out = pd.DataFrame({n: conds[n].fillna(False).astype(bool) for n in COND_NAMES})
    out.insert(0, 'base', base.fillna(False).astype(bool))
    aux = pd.DataFrame({
        'close': c, 'rsi': rsi, 'cci': cci, 'cmf': cmf, 'wr': wr, 'hist': hist,
        'atr_pct': atr / c * 100,
        'dist_52w': (c / c.rolling(252).min() - 1) * 100,
        'dist_high252': (c / c.rolling(252).max() - 1) * 100,
        'dist_ma200': (c / ma200 - 1) * 100,
        'dist_ma50': dist50,
        'ret5d': (c / c.shift(5) - 1) * 100,
        'ret20d': (c / c.shift(20) - 1) * 100,
        'ret60d': (c / c.shift(60) - 1) * 100,
        'vol_ratio20': v / vma20,
        'stop_ma20': bb_mid * 0.97,
        # ── 扩展数值特征（给更大范围的穷举用） ──
        'rsi_prev': r1, 'rsi_min5': rsi.rolling(5).min(), 'mfi': mfi, 'stoch_k': k, 'stoch_d': d, 'adx': adx,
        'hist_norm': hist / c * 100, 'hist_slope': (hist - hist.shift(1)) / c * 100,
        'hist_min20_norm': hist.rolling(20).min() / c * 100,
        'ret1d': (c / c1 - 1) * 100, 'ret3d': (c / c3 - 1) * 100, 'ret10d': (c / c.shift(10) - 1) * 100,
        'ret120d': (c / c.shift(120) - 1) * 100, 'ret250d': (c / c.shift(250) - 1) * 100,
        'vol_ratio5': v / v.shift(1).rolling(5).mean(),
        'bb_pctb': (c - bb_lower) / (4 * bb_std).replace(0, np.nan),
        'bb_width_ratio': bb_width / bb_width_ma, 'atr_ratio': atr / atr_ma20,
        'body_pct': (c - o) / (h - l).replace(0, np.nan),
        'lower_shadow': (np.minimum(o, c) - l) / (h - l).replace(0, np.nan),
        'upper_shadow': (h - np.maximum(o, c)) / (h - l).replace(0, np.nan),
        'range_pct': (h - l) / c * 100, 'gap_today': (o / c1 - 1) * 100,
        'dist_ma20': (c / bb_mid - 1) * 100, 'ma50_vs_ma200': (ma50 / ma200 - 1) * 100,
        'ma200_slope20': (ma200 / ma200.shift(20) - 1) * 100,
        'days_since_low252': c.rolling(252).apply(lambda x: len(x) - 1 - np.argmin(x), raw=True),
        'down_streak': (c < c1).astype(int).groupby((c >= c1).cumsum()).cumsum(),
        'vol60': c.pct_change().rolling(60).std() * np.sqrt(252) * 100,
        'cmf_chg5': cmf - cmf.shift(5),
        'obv_slope20': (obv - obv.shift(20)) / v.rolling(20).sum(),
        'log_price': np.log(c), 'log_dvol20': np.log((c * v).rolling(20).mean() + 1),
    })
    return pd.concat([out, aux], axis=1)


def forward(o, h, l, c):
    """入场 = 信号次日收盘；r{n} = 入场后第 n 个交易日收盘相对入场价；另存信号日收盘起算的 s{n}。"""
    e = c.shift(-1)
    f = {'entry_close': e, 'gap_next_open': (o.shift(-1) / c - 1) * 100}
    for n in (1, 3, 5, 10, 20):
        f[f'r{n}'] = (c.shift(-1 - n) / e - 1) * 100
    for n in (1, 3, 5, 10):
        f[f's{n}'] = (c.shift(-n) / c - 1) * 100
    for n in (5, 10):   # 持有期内最低 / 最高（入场次日起）
        lows = pd.concat([l.shift(-1 - i) for i in range(1, n + 1)], axis=1)
        highs = pd.concat([h.shift(-1 - i) for i in range(1, n + 1)], axis=1)
        f[f'mae{n}'] = (lows.min(axis=1, skipna=False) / e - 1) * 100
        f[f'mfe{n}'] = (highs.max(axis=1, skipna=False) / e - 1) * 100
    return pd.DataFrame(f)


def compute(mem):
    px = {f: pd.read_parquet(os.path.join(DATA, f'ohlcv_{f.lower()}.parquet')) for f in FIELDS}
    gspc = pd.read_parquet(os.path.join(DATA, 'index_gspc.parquet')).iloc[:, 0]
    vix = pd.read_parquet(os.path.join(DATA, 'index_vix.parquet')).iloc[:, 0]
    sectors = pd.read_csv(os.path.join(DATA, 'sectors_current.csv'), index_col=0)['sector'].to_dict()
    today = pd.Timestamp.today().normalize()

    # 成分股：信号日用「上一个月末」的名单（不偷看当月新纳入）
    months = np.sort(mem.month_end.unique())
    mem_sets = {m: set(g.ticker) for m, g in mem.groupby('month_end')}

    def in_index(day, t):
        i = months.searchsorted(np.datetime64(day), side='left') - 1
        return i >= 0 and t in mem_sets[months[i]]

    rows = []
    tickers = list(px['Close'].columns)
    t0 = time.time()
    for n_t, t in enumerate(tickers):
        df = pd.DataFrame({f: px[f][t] for f in FIELDS}).dropna(subset=['Close'])
        df = df[df.index < today]                                  # 不用今天未收盘的 K 线
        if len(df) < 300:
            continue
        o, h, l, c, v = (df[f] for f in FIELDS)
        cf = conditions_frame(o, h, l, c, v)
        sig = cf[cf.base & (cf.index >= SIG_START)]
        if sig.empty:
            continue
        fw = forward(o, h, l, c).loc[sig.index]
        part = pd.concat([sig, fw], axis=1)
        part.insert(0, 'ticker', t)
        part.insert(1, 'sector_now', sectors.get(t, ''))
        part['in_index'] = [in_index(dd, t) for dd in part.index]
        part['entry_day'] = [df.index[df.index.get_loc(dd) + 1] if df.index.get_loc(dd) + 1 < len(df) else pd.NaT
                             for dd in part.index]
        rows.append(part)
        if (n_t + 1) % 100 == 0:
            print(f'  计算 {n_t + 1}/{len(tickers)}  {time.time() - t0:.0f}s', flush=True)

    s = pd.concat(rows).rename_axis('signal_day').reset_index()
    # 市场环境
    s['vix'] = vix.reindex(s.signal_day, method='ffill').values
    g_ma200 = gspc.rolling(200).mean()
    s['spx_dist_ma200'] = ((gspc / g_ma200 - 1) * 100).reindex(s.signal_day, method='ffill').values
    s['spx_ret5d'] = ((gspc / gspc.shift(5) - 1) * 100).reindex(s.signal_day, method='ffill').values
    s['spx_ret20d'] = ((gspc / gspc.shift(20) - 1) * 100).reindex(s.signal_day, method='ffill').values
    s['spx_rsi'] = ds.calc_rsi(gspc).reindex(s.signal_day, method='ffill').values
    s['vix_chg5'] = ((vix / vix.shift(5) - 1) * 100).reindex(s.signal_day, method='ffill').values
    s['rel_ret20d'] = s['ret20d'] - s['spx_ret20d']
    s['weekday'] = s.signal_day.dt.weekday
    s['month'] = s.signal_day.dt.month
    gi = gspc.index
    for n in (1, 3, 5, 10, 20):
        fwd = gspc.shift(-n) / gspc - 1
        s[f'b{n}'] = (fwd.reindex(s.entry_day).values * 100)
    for n in (1, 3, 5, 10, 20):
        s[f'x{n}'] = s[f'r{n}'] - s[f'b{n}']
    s['n_base_day'] = s.groupby('signal_day').ticker.transform('size')
    s['n_base_day_idx'] = s.groupby('signal_day').in_index.transform('sum')
    s['year'] = s.signal_day.dt.year
    s.to_parquet(os.path.join(DATA, 'signals_20y.parquet'), index=False)

    # 等权成分股指数（日收益），用于另一种基准
    close = px['Close'][px['Close'].index < today]
    ret = close.pct_change(fill_method=None)
    mask = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    for m_prev, m_next in zip(months[:-1], list(months[1:])):
        cols = [t for t in mem_sets[m_prev] if t in mask.columns]
        sel = (ret.index > m_prev) & (ret.index <= m_next)
        mask.loc[sel, cols] = True
    last = months[-1]
    mask.loc[ret.index > last, [t for t in mem_sets[last] if t in mask.columns]] = True
    ew = ret.where(mask).mean(axis=1)
    ew.rename('ew_ret').to_frame().to_parquet(os.path.join(DATA, 'ew_index.parquet'))

    # 覆盖率：每年成分股里有行情的比例
    have = set(close.columns)
    cov = []
    for m in months[11::12]:
        ms = mem_sets[m]
        cov.append({'month_end': pd.Timestamp(m).date(), 'members': len(ms),
                    'with_data': len(ms & have), 'pct': round(len(ms & have) / len(ms) * 100, 1)})
    pd.DataFrame(cov).to_csv(os.path.join(DATA, 'coverage.csv'), index=False)
    print(f'\n基础信号 {len(s)} 条（其中当时在指数内 {int(s.in_index.sum())} 条），'
          f'{s.signal_day.min().date()} → {s.signal_day.max().date()}')
    print(pd.DataFrame(cov).to_string(index=False))


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    if '--no-download' in sys.argv:
        mem = pd.read_csv(os.path.join(DATA, 'membership.csv.gz'), parse_dates=['month_end'])
    else:
        print('1/3 时点成分股...', flush=True)
        mem = membership()
        print(f'  {mem.month_end.nunique()} 个月，累计 {mem.ticker.nunique()} 只出现过', flush=True)
        print('2/3 下载 2005 年至今日线...', flush=True)
        download(sorted(set(mem.ticker)))
    print('3/3 计算信号和条件...', flush=True)
    compute(mem)
