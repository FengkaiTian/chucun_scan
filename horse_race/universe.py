"""时点标普 500 成分股（按日）：解析 Wikipedia 存档快照（results/wiki_test_*.html，2026-09-24 抓取），
从当前成分股出发倒序撤销变更。与 data_pipeline.membership 同一逻辑，只是精度从月末提到每日。"""
import io
import numpy as np, pandas as pd

SNAP = 'results/wiki_test_{}.html'


def events(snap=SNAP):
    t = []
    for name in ['current', 'historical']:
        t += pd.read_html(io.StringIO(open(snap.format(name), encoding='utf-8').read()))
    fix = lambda s: s.fillna('').astype(str).str.strip().str.replace('.', '-', regex=False)
    flat = lambda df: [' '.join(dict.fromkeys(map(str, c if isinstance(c, tuple) else (c,)))).lower() for c in df.columns]
    chg = next(df for df in t if any('added' in c for c in flat(df)) and any('removed' in c for c in flat(df)))
    chg.columns = flat(chg)
    col = lambda k, alt=('',): next(c for c in chg.columns if k in c and any(a in c for a in alt))
    ev = pd.DataFrame({'date': pd.to_datetime(chg[col('date')], format='mixed', errors='coerce'),
                       'add': fix(chg[col('added', ('ticker', 'symbol'))]),
                       'rem': fix(chg[col('removed', ('ticker', 'symbol'))])})
    return set(fix(t[0]['Symbol'])), ev.dropna(subset=['date']).sort_values('date', ascending=False)


def membership(days, snap=SNAP):
    """bool DataFrame(days × tickers)：当日收盘时是否为成分股（生效日 <= 当日的变更已生效）"""
    cur, ev = events(snap)
    ev = ev.to_dict('records')
    rows, i = {}, 0
    for d in sorted(days, reverse=True):
        while i < len(ev) and ev[i]['date'] > d:
            cur.discard(ev[i]['add'])
            if ev[i]['rem']:
                cur.add(ev[i]['rem'])
            i += 1
        rows[d] = frozenset(cur)
    tick = sorted(set().union(*rows.values()))
    pos = {t: k for k, t in enumerate(tick)}
    m = np.zeros((len(days), len(tick)), bool)
    for r, d in enumerate(days):
        m[r, [pos[t] for t in rows[d]]] = True
    return pd.DataFrame(m, index=days, columns=tick)
