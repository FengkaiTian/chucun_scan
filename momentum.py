"""
v6: S&P 500 月度动量（12-1）+ 大盘趋势过滤
参数全部取自文献，不做样本内寻优——v5 穷举 1,626 种组合、实盘失效就是前车之鉴。
  动量排序: Jegadeesh & Titman (1993), J. Finance 48(1):65-91；跳过最近 1 个月（短期反转）
  趋势过滤: SPY 收盘 < 200 日均线时空仓
用法: python momentum.py pick | backtest
"""
import io, sys
import pandas as pd, requests, yfinance as yf

TOP_N, COST = 20, 0.001  # 持仓数（等权）；单边交易成本 10bp


def sp500():
    html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers={'User-Agent': 'Mozilla/5.0'}, timeout=30).text
    return pd.read_html(io.StringIO(html))[0]['Symbol'].str.replace('.', '-', regex=False).tolist()


def load(start):
    px = yf.download(sp500() + ['SPY'], start=start, auto_adjust=True, progress=False, threads=True)['Close']
    return px.dropna(axis=1, how='all')


def weights(px, top_n=TOP_N, trend=True):
    """月末信号 -> 下个月的目标权重（行: 月末，列: 股票）"""
    m = px.resample('ME').last()
    mom = (m.shift(1) / m.shift(12) - 1).drop(columns='SPY')        # t-12 → t-1
    w = (mom.rank(axis=1, ascending=False) <= top_n).astype(float)
    w = w.div(w.sum(axis=1), axis=0).fillna(0)
    if trend:
        risk_on = (px['SPY'] > px['SPY'].rolling(200).mean()).resample('ME').last()
        w = w.mul(risk_on.astype(float), axis=0)
    return w, m, mom


def backtest(px, top_n=TOP_N, trend=True):
    w, m, _ = weights(px, top_n, trend)
    ret = m.pct_change().drop(columns='SPY')
    turnover = w.diff().abs().sum(axis=1)
    net = (w.shift(1) * ret).sum(axis=1) - turnover.shift(1) * COST
    return net, turnover


def stats(r):
    eq = (1 + r).cumprod()
    return {'CAGR': eq.iloc[-1] ** (12 / len(r)) - 1, 'Vol': r.std() * 12 ** .5,
            'Sharpe(rf=0)': r.mean() / r.std() * 12 ** .5, 'MaxDD': (eq / eq.cummax() - 1).min()}


def report(px, start='2006-01'):
    m = px.resample('ME').last().pct_change()
    runs = {f'动量Top{TOP_N}+趋势过滤': backtest(px)[0],
            f'动量Top{TOP_N}(无过滤)': backtest(px, trend=False)[0],
            **{f'  敏感性 Top{n}+过滤': backtest(px, n)[0] for n in (10, 50)},
            '成分股等权(对照)': m.drop(columns='SPY').mean(axis=1),
            'SPY': m['SPY']}
    for lo, hi in [(start, None), (start, '2015-12'), ('2016-01', None)]:
        print(f'\n== {lo} ~ {hi or "今"} ==')
        print(pd.DataFrame({k: stats(v[lo:hi].dropna()) for k, v in runs.items()}).T.to_string(float_format='{:.3f}'.format))
    print(f'\n年化换手: {backtest(px)[1][start:].mean() * 12:.1f} 倍')
    print('⚠️ 幸存者偏差：用的是【当前】成分股回溯，绝对收益偏高；应看相对"成分股等权"的超额。')


def pick(px):
    w, _, mom = weights(px)
    risk_on = bool(px['SPY'].iloc[-1] > px['SPY'].rolling(200).mean().iloc[-1])
    top = mom.iloc[-1].nlargest(TOP_N)
    lines = [f'# v6 动量持仓 {px.index[-1]:%Y-%m-%d}', '',
             f'大盘 {"在" if risk_on else "跌破"} 200 日均线 → {"持有下列等权组合" if risk_on else "空仓"}', '',
             '| # | 股票 | 12-1 动量 |', '|---|---|---|']
    lines += [f'| {i} | {t} | {v:+.1%} |' for i, (t, v) in enumerate(top.items(), 1)]
    out = '\n'.join(lines) + '\n'
    open('momentum_picks.md', 'w', encoding='utf-8').write(out)
    print(out)


if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'pick'
    if task == 'backtest':
        report(load('2004-06-01'))
    else:
        pick(load(pd.Timestamp.today() - pd.DateOffset(months=15)))
