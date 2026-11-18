"""agentA 全量扫描：python run_agentA.py [prep|time|full]
prep: 生成原子矩阵（npy）+ 组合清单；time: 试跑估时；full: 全量 + 随机对照 → lab/out/agentA.parquet"""
import os, sys, json, time, itertools, pickle
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
WORK = os.environ.get('AGENTA_WORK', os.path.join(HERE, 'work'))
os.makedirs(WORK, exist_ok=True)
HOLDS = (1, 3, 5, 10, 20)
NPROC = 5
N_TRIPLES = 3500
SEED = 20261118

_G = {}


def _init():
    from lab_common import Lab
    _G['lab'] = Lab()
    _G['A'] = np.load(os.path.join(WORK, 'atoms.npy'), mmap_mode='r')
    _G['meta'] = json.load(open(os.path.join(WORK, 'atoms_meta.json'), encoding='utf-8'))


def _job(ids):
    lab, A, meta = _G['lab'], _G['A'], _G['meta']
    sig = np.array(A[ids[0]])
    for i in ids[1:]:
        sig &= A[i]
    names = [meta[i]['name'] for i in ids]
    cats = [meta[i]['cat'] for i in ids]
    ids_s = '+'.join(meta[i]['id'] for i in ids)
    fam = f'个股条件{len(ids)}重:' + '×'.join(sorted(set(cats)))
    out = []
    for h in HOLDS:
        nm = ' & '.join(names) + f' | 持有{h}天'
        params = json.dumps({'atoms': ids_s, 'conds': names, 'cats': cats, 'hold': h,
                             'rebuild': 'agentA/atoms.py build_atoms(lab) 按 id 取矩阵后 AND'}, ensure_ascii=False)
        r = _G['lab'].evaluate(sig, h, nm, fam, params)
        r['n_conds'] = len(ids); r['atoms'] = ids_s
        out.append(r)
    return out


def _rand_job(arg):
    dens, seed = arg
    lab = _G['lab']; sig = lab.random_signals(dens, seed)
    out = []
    for h in HOLDS:
        r = lab.evaluate(sig, h, f'随机信号 密度{dens:g} seed{seed} | 持有{h}天', '随机对照',
                         json.dumps({'density': dens, 'seed': seed, 'hold': h}))
        r['n_conds'] = 0; r['atoms'] = ''
        out.append(r)
    return out


def prep():
    from lab_common import Lab
    from atoms import build_atoms
    t0 = time.time()
    lab = Lab()
    atoms = build_atoms(lab)
    T, N = lab.C.shape
    arr = np.lib.format.open_memmap(os.path.join(WORK, 'atoms.npy'), 'w+', np.bool_, (len(atoms), T, N))
    meta = []
    tr = (lab.year <= 2019)[:, None] & lab.member
    va = ((lab.year >= 2020) & (lab.year <= 2024))[:, None] & lab.member
    for k, (aid, cat, name, m) in enumerate(atoms):
        arr[k] = m & lab.member
        meta.append(dict(id=aid, cat=cat, name=name, n_tr=int((m & tr).sum()), n_va=int((m & va).sum())))
    arr.flush()
    json.dump(meta, open(os.path.join(WORK, 'atoms_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    print('atoms', len(atoms), 'build', round(time.time() - t0, 1), 's')
    trm, vam = tr, va
    del atoms
    A = np.load(os.path.join(WORK, 'atoms.npy'), mmap_mode='r')
    K = len(meta)
    # 训练/验证分开计数用的压缩：每个原子只保留 (train 区 & valid 区) 的扁平布尔
    yrs = lab.year
    rows_tr = np.where(yrs <= 2019)[0]; rows_va = np.where((yrs >= 2020) & (yrs <= 2024))[0]
    Atr = np.packbits(np.stack([np.asarray(A[k][rows_tr]).ravel() for k in range(K)]), axis=1)
    Ava = np.packbits(np.stack([np.asarray(A[k][rows_va]).ravel() for k in range(K)]), axis=1)
    pc = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    cnt = lambda P, ids: int(pc[np.bitwise_and.reduce(P[list(ids)], axis=0)].sum())
    singles = [[k] for k in range(K)]
    pairs, pc_tr = [], {}
    for a, b in itertools.combinations(range(K), 2):
        if meta[a]['cat'] == meta[b]['cat']:
            continue
        ntr = cnt(Atr, (a, b)); nva = cnt(Ava, (a, b))
        if ntr == 0 and nva == 0:
            continue          # 互斥条件（如新高&新低），不评
        pairs.append([a, b]); pc_tr[(a, b)] = (ntr, nva)
    print('pairs', len(pairs), round(time.time() - t0, 1), 's')
    # 三重：三个不同类别；只从"两两都有足量样本"的组合里随机抽
    rng = np.random.default_rng(SEED)
    good = {p for p, (x, y) in pc_tr.items() if x >= 2000 and y >= 400}
    cats = sorted({m['cat'] for m in meta})
    triples, seen, tries = [], set(), 0
    while len(triples) < N_TRIPLES and tries < 400000:
        tries += 1
        cs = rng.choice(len(cats), 3, replace=False)
        ids = sorted(int(rng.choice([k for k in range(K) if meta[k]['cat'] == cats[c]])) for c in cs)
        t3 = tuple(ids)
        if t3 in seen:
            continue
        seen.add(t3)
        if not all(p in good for p in itertools.combinations(t3, 2)):
            continue
        ntr = cnt(Atr, t3); nva = cnt(Ava, t3)
        if ntr >= 150 and nva >= 40:
            triples.append(list(t3))
    print('triples', len(triples), 'tries', tries, round(time.time() - t0, 1), 's')
    # 随机对照密度：取组合信号密度（训练期信号数/成员日）的分位
    member_days = int(trm.sum())
    dens = np.array([pc_tr[tuple(p)][0] for p in pairs] + [m['n_tr'] for m in meta]) / member_days
    q = np.quantile(dens[dens > 0], [0.1, 0.25, 0.4, 0.55, 0.7, 0.85])
    json.dump(dict(singles=singles, pairs=pairs, triples=triples, rand_dens=[float(x) for x in q]),
              open(os.path.join(WORK, 'jobs.json'), 'w'))
    print('rand densities', q)


def run(mode):
    from multiprocessing import Pool
    J = json.load(open(os.path.join(WORK, 'jobs.json')))
    jobs = J['singles'] + J['pairs'] + J['triples']
    if mode == 'time':
        rng = np.random.default_rng(1)
        jobs = [jobs[i] for i in rng.choice(len(jobs), 60, replace=False)]
    rand = [(d, s) for d in J['rand_dens'] for s in range(10)]   # 6 档 × 10 种子 × 5 持有 = 300
    print('combos', len(jobs), 'x', len(HOLDS), '=', len(jobs) * len(HOLDS), '+ random', len(rand) * len(HOLDS), flush=True)
    t0 = time.time(); rows = []
    with Pool(NPROC, initializer=_init) as pool:
        print('pool ready', flush=True)
        t1 = time.time()
        for k, res in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            rows.extend(res)
            if (k + 1) % 500 == 0:
                el = time.time() - t1
                print(f'{k+1}/{len(jobs)} {el:.0f}s  ETA {el/(k+1)*(len(jobs)-k-1)/60:.1f}min', flush=True)
            if mode == 'full' and (k + 1) % 3000 == 0:
                pickle.dump(rows, open(os.path.join(WORK, 'partial.pkl'), 'wb'))
        if mode == 'full':
            for res in pool.imap_unordered(_rand_job, rand, chunksize=2):
                rows.extend(res)
    el = time.time() - t1
    print('done', len(rows), 'rows', round(el, 1), 's; per combo', round(el / max(len(jobs), 1), 3), flush=True)
    if mode == 'full':
        from lab_common import Lab
        for r in rows:
            r['样本不足'] = bool(r.get('train_n', 0) < 50 or r.get('valid_n', 0) < 20)
        p = Lab.save(None, rows, 'agentA')     # save 不依赖实例数据，免得主进程再载一份面板
        print('saved', p, len(rows))


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else 'prep'
    prep() if m == 'prep' else run(m)
