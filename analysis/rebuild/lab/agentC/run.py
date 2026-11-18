"""Agent C 全量运行：入场 × 出场库 + 网格加仓 + 随机对照。
用法: python run.py CACHE [validate|time|full]"""
import os, sys, json, time, types
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import lab_common as LC
from sim_core import sim, sim_grid

CACHE = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else 'full'
FAM = '自建模拟'


def leg(k=-1, mh=0, tp=0, sl=0, tr=0, atp=0, asl=0, tsd=0, tsr=0, maxh=10):
    return [k, mh, tp, sl, tr, atp, asl, tsd, tsr, maxh]


def build_exits(xn):
    """返回 [(名字, 类别, leg1, leg2|None)]"""
    X = {n: i for i, n in enumerate(xn)}
    out = []
    for hd in range(1, 31):
        out.append((f'固定{hd}天', '固定持有', leg(maxh=hd), None))
    trend = ['C<MA5', 'C<MA10', 'C<MA20', 'C<MA50', 'C<前10日最低收盘', 'C<前20日最低收盘']
    for n, k in X.items():
        for m in ((5, 10, 20, 40, 60) if n in trend else (5, 10, 20)):
            out.append((f'{n} 或≤{m}天', '指标出场', leg(k=k, maxh=m), None))
    for m in (10, 20, 40):
        for tp in (3, 5, 8, 10, 15):
            for sl in (0, 3, 5, 8, 10):
                out.append((f'止盈{tp}%' + (f' 止损{sl}%' if sl else '') + f' ≤{m}天', '止盈止损', leg(tp=tp, sl=sl, maxh=m), None))
    for m in (5, 10, 20):
        for sl in (3, 5, 8, 10):
            out.append((f'止损{sl}% ≤{m}天', '止盈止损', leg(sl=sl, maxh=m), None))
    for m in (10, 20, 40, 60):
        for tr in (3, 5, 8, 10, 15):
            out.append((f'移动止损{tr}% ≤{m}天', '移动止损', leg(tr=tr, maxh=m), None))
    for m in (10, 20):
        for a in (1, 2, 3):
            for s in (0, 1, 2, 3):
                out.append((f'ATR止盈{a}x' + (f' ATR止损{s}x' if s else '') + f' ≤{m}天', 'ATR', leg(atp=a, asl=s, maxh=m), None))
    for m in (5, 10):
        for s in (1, 2, 3):
            out.append((f'ATR止损{s}x ≤{m}天', 'ATR', leg(asl=s, maxh=m), None))
    for m in (10, 20):
        for d in (2, 3, 5):
            out.append((f'第{d}天仍亏则卖 ≤{m}天', '时间止损', leg(tsd=d, maxh=m), None))
    for tp in (5, 8):
        for d in (2, 3, 5):
            out.append((f'止盈{tp}% 第{d}天仍亏则卖 ≤20天', '时间止损', leg(tp=tp, tsd=d, maxh=20), None))
    core = ['C>MA5', 'C>MA10', 'C>MA20', 'RSI14>=50', 'RSI14>=70', 'RSI2>=70', 'MACD金叉(柱>0)', 'MACD柱变差', '收盘站上前日高', '坏信号>=2']
    adds = [('止损5%', dict(sl=5)), ('止损8%', dict(sl=8)), ('止盈8%', dict(tp=8)), ('止盈10%止损8%', dict(tp=10, sl=8)),
            ('移动止损5%', dict(tr=5)), ('第3天仍亏则卖', dict(tsd=3))]
    for n in core:
        for an, kw in adds:
            for m in (10, 20):
                out.append((f'{n} 或{an} ≤{m}天', '组合', leg(k=X[n], maxh=m, **kw), None))
    for n in core:
        for mh in (2, 3):
            out.append((f'持满{mh}天后{n} ≤10天', '最少持有', leg(k=X[n], mh=mh, maxh=10), None))
    l1s = [('止盈3%', leg(tp=3, maxh=20)), ('止盈5%', leg(tp=5, maxh=20)), ('固定3天', leg(maxh=3)), ('C>MA5', leg(k=X['C>MA5'], maxh=10))]
    l2s = [('C>MA20≤20', leg(k=X['C>MA20'], maxh=20)), ('RSI14>=70≤20', leg(k=X['RSI14>=70'], maxh=20)),
           ('移动止损5%≤20', leg(tr=5, maxh=20)), ('固定10天', leg(maxh=10)), ('固定20天', leg(maxh=20)),
           ('止盈10%止损8%≤20', leg(tp=10, sl=8, maxh=20))]
    for a, p1 in l1s:
        for b, p2 in l2s:
            out.append((f'分批: 半仓{a} + 半仓{b}', '分批卖', p1, p2))
    return out


def summarize(t, r, d, s, year, name, family, params, hold):
    ns = types.SimpleNamespace(year=year)
    sealed = LC._seal(year, t, r, s)
    keep = year[t] < LC.HOLD_OUT_FROM
    t, r, d, s = t[keep], r[keep], d[keep], s[keep]
    res = dict(name=name, family=family, params=params, hold=hold, sealed=sealed)
    res.update(LC.Lab._summ(ns, t, r, d))
    for lab_, lo, hi in (('train', *LC.TRAIN), ('valid', *LC.VALID)):
        m = (year[t] >= lo) & (year[t] <= hi)
        if m.sum():
            ex = r[m] - s[m]
            res[f'{lab_}_excess'] = float(ex.mean()); res[f'{lab_}_t_excess'] = float(LC._cluster_t(ex, t[m]))
    return res


G = {}


def init():
    G['ent'] = np.load(os.path.join(CACHE, 'ent.npy'), mmap_mode='r')
    G['ex'] = np.load(os.path.join(CACHE, 'ex.npy'), mmap_mode='r')
    for k in ('C', 'atr', 'spx', 'year', 'member'):
        G[k] = np.load(os.path.join(CACHE, f'{k}.npy'))
    G['meta'] = json.load(open(os.path.join(CACHE, 'meta.json'), encoding='utf-8'))
    G['exits'] = build_exits(G['meta']['exits'])
    G['ex_arr'] = G['ex']  # memmap，多进程共享页缓存


def run_one(sig, en, xname, xcat, p1, p2, family):
    C = G['C']
    pp = np.array(p1, float); qq = np.array(p2 if p2 is not None else p1, float)
    t, r, d, s = sim(sig, G['ex_arr'], C, G['atr'], G['spx'], pp, qq, p2 is not None, 1)
    hold = f'≤{int(max(pp[9], qq[9]))}'
    return summarize(t, r, d, s, G['year'], f'{en} | {xname}', family, json.dumps({'entry': en, 'exit': xname, 'cat': xcat, 'leg1': p1, 'leg2': p2}, ensure_ascii=False), hold)


def task(job):
    if not G:
        init()
    kind = job[0]
    rows = []
    if kind == 'entry':
        i, idxs = job[1], job[2]
        en = G['meta']['entries'][i]
        sig = np.ascontiguousarray(G['ent'][i])
        fam_base = '基础信号' if i < 9 else '经典入场'
        for xi in idxs:
            xname, xcat, p1, p2 = G['exits'][xi]
            rows.append(run_one(sig, en, xname, xcat, p1, p2, f'出场×{fam_base}-{xcat}（{FAM}）'))
    elif kind == 'grid':
        i = job[1]
        en = G['meta']['entries'][i]
        sig = np.ascontiguousarray(G['ent'][i])
        for X in (3, 5, 8):
            for K in (1, 2, 3):
                for Y in (3, 5, 8):
                    for part in (False, True):
                        if part and K != 2:
                            continue
                        t, r, d, s = sim_grid(sig, G['C'], G['spx'], float(X), K, float(Y), part, 20, 1, LC.COST)
                        xn = f'网格: 跌{X}%加仓(最多{K}次) 均价涨{Y}%' + ('卖半/涨' + str(2 * Y) + '%卖完' if part else '全卖') + ' ≤20天'
                        rows.append(summarize(t, r, d, s, G['year'], f'{en} | {xn}', f'网格加仓（{FAM}，按预留{1 + K}份资金）',
                                              json.dumps({'entry': en, 'exit': xn, 'X': X, 'K': K, 'Y': Y, 'partial': part}, ensure_ascii=False), '≤20'))
    elif kind == 'random':
        dens, seeds = job[1], job[2]
        common = ['固定5天', '固定10天', 'C>MA5 或≤10天', '止盈5% 止损5% ≤20天', 'RSI14>=50 或≤10天']
        ex_by = {e[0]: e for e in G['exits']}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sig = np.ascontiguousarray((rng.random(G['C'].shape) < dens) & G['member'])
            for c in common:
                xname, xcat, p1, p2 = ex_by[c]
                rows.append(run_one(sig, f'随机入场(密度{dens:.4f},seed{seed})', xname, xcat, p1, p2, '随机对照'))
    return rows


def validate():
    from lab_common import Lab
    lab = Lab()
    init()
    X = {n: i for i, n in enumerate(G['meta']['exits'])}
    sig = np.ascontiguousarray(G['ent'][0])
    keys = ['train_n', 'train_avg', 'train_t', 'valid_n', 'valid_avg', 'valid_t', 'train_hold', 'valid_excess', 'sealed']
    a = lab.evaluate(sig, 5, 'x'); b = run_one(sig, 'E01', '固定5天', '', leg(maxh=5), None, '')
    print('fixed5 lab ', [a.get(k) for k in keys]); print('fixed5 mine', [b.get(k) for k in keys])
    for nm, m in (('C>MA5', 10), ('RSI14>=50', 20)):
        a = lab.evaluate_custom(sig, G['ex_arr'][X[nm]], m, 'x'); b = run_one(sig, 'E01', nm, '', leg(k=X[nm], maxh=m), None, '')
        print(nm, 'lab ', [a.get(k) for k in keys]); print(nm, 'mine', [b.get(k) for k in keys])
    sig = np.ascontiguousarray(G['ent'][10])
    a = lab.evaluate_custom(sig, G['ex_arr'][X['C>MA10']], 10, 'x'); b = run_one(sig, 'E11', 'C>MA10', '', leg(k=X['C>MA10'], maxh=10), None, '')
    print('E11 lab ', [a.get(k) for k in keys]); print('E11 mine', [b.get(k) for k in keys])


if __name__ == '__main__':
    if MODE == 'validate':
        validate(); sys.exit()
    init()
    ne = len(G['meta']['entries']); nx = len(G['exits'])
    print('entries', ne, 'exits', nx)
    if MODE == 'time':
        t0 = time.time()
        r = task(('entry', 10, list(range(0, nx, 25))))  # 最密的 RSI2<10
        t1 = time.time(); print('dense entry per strat', (t1 - t0) / len(r))
        r = task(('entry', 0, list(range(0, nx, 25))))
        t2 = time.time(); print('sparse entry per strat', (t2 - t1) / len(r))
        r = task(('grid', 10)); t3 = time.time(); print('grid per strat', (t3 - t2) / len(r))
        sys.exit()
    jobs = []
    chunk = 75
    for i in range(ne):
        for s in range(0, nx, chunk):
            jobs.append(('entry', i, list(range(s, min(s + chunk, nx)))))
    for i in range(ne):
        jobs.append(('grid', i))
    dens = G['meta']['density']
    for di, dv in enumerate((dens[0], dens[23], dens[10])):   # 基础信号 / RSI14<30 / RSI2<10 的稀疏度
        jobs.append(('random', dv, list(range(1000 + di * 100, 1000 + di * 100 + 20))))
    print('jobs', len(jobs))
    from multiprocessing import Pool
    rows = []; t0 = time.time()
    with Pool(5) as pool:
        for k, rr in enumerate(pool.imap_unordered(task, jobs)):
            rows.extend(rr)
            if k % 20 == 0:
                print(f'{k}/{len(jobs)} rows={len(rows)} {time.time() - t0:.0f}s', flush=True)
    p = LC.Lab.save(None, rows, 'agentC')   # 等价于 lab.save(rows, 'agentC')
    print('saved', p, len(rows), time.time() - t0)
