"""agentB 主扫描：个股信号 × 市场状态(0/1/2个) × 持有天数 + 随机对照。
用法: python run.py [test N]   （先 python features.py 生成 cache）"""
import os, sys, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
CACHE = os.path.join(HERE, 'cache')
HOLDS = (1, 3, 5, 10, 20)
NPROC = 5

_G = {}


def _init():
    from lab_common import Lab
    lab = Lab()
    del lab.O, lab.H, lab.L, lab.V           # evaluate 只用 C/member/spx/year，省内存
    _G['lab'] = lab
    _G['base'] = np.load(os.path.join(CACHE, 'base.npy'), mmap_mode='r')
    _G['st'] = pd.read_parquet(os.path.join(CACHE, 'states.parquet'))
    _G['names'] = pd.read_csv(os.path.join(CACHE, 'base_names.csv')).iloc[:, 0].tolist()


def _job(job):
    lab = _G['lab']
    kind = job[0]
    if kind == 'rand':
        _, dens, seed, hold = job
        sig = lab.random_signals(dens, seed)
        r = lab.evaluate(sig, hold, f'随机 密度{dens} seed{seed}', '随机对照', f'density={dens};seed={seed}')
        r.update(base='随机', state1='', state2='', n_states=0)
        return r
    _, b, s1, s2, hold = job
    sig = np.asarray(_G['base'][b])
    nm = _G['names'][b]
    m = np.ones(sig.shape[0], bool)
    for s in (s1, s2):
        if s:
            m &= _G['st'][s].values
    if s1:
        sig = sig & m[:, None]
    cond = ' & '.join(x for x in (s1, s2) if x)
    fam = '个股信号' if not s1 else ('个股×1市场状态' if not s2 else '个股×2市场状态')
    name = f'{nm}' + (f' | {cond}' if cond else '') + f' | 持{hold}天'
    r = lab.evaluate(sig, hold, name, fam, f'base={nm};state={cond};hold={hold}')
    r.update(base=nm, state1=s1 or '', state2=s2 or '', n_states=(s1 is not None) + (s2 is not None))
    return r


def make_jobs():
    from features import KEY_STATES
    st = pd.read_parquet(os.path.join(CACHE, 'states.parquet'))
    names = pd.read_csv(os.path.join(CACHE, 'base_names.csv')).iloc[:, 0].tolist()
    yr = st.index.year
    tr = (yr >= 2000) & (yr <= 2019); va = (yr >= 2020) & (yr <= 2024)
    pairs = []
    for a, b in itertools.combinations(KEY_STATES, 2):
        x = (st[a] & st[b]).values
        nested = x.sum() == min(st[a].sum(), st[b].sum())   # 一个包含另一个 = 冗余
        if x[tr].sum() >= 30 and x[va].sum() >= 10 and not nested:     # 只按天数过滤，不看收益
            pairs.append((a, b))
    jobs = []
    for b in range(len(names)):
        for h in HOLDS:
            jobs.append(('sig', b, None, None, h))
            for s in st.columns:
                if st[s].values[tr].sum() >= 30 and st[s].values[va].sum() >= 10:
                    jobs.append(('sig', b, s, None, h))
            for a, c in pairs:
                jobs.append(('sig', b, a, c, h))
    for dens in (0.0005, 0.002, 0.01, 0.04):
        for h in HOLDS:
            for seed in range(15):
                jobs.append(('rand', dens, 1000 + seed + int(dens * 1e6), h))
    return jobs, len(pairs)


if __name__ == '__main__':
    jobs, npairs = make_jobs()
    test = len(sys.argv) > 2 and sys.argv[1] == 'test'
    if test:
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in rng.choice(len(jobs), int(sys.argv[2]), replace=False)]
    print(f'jobs={len(jobs)} pairs={npairs}', flush=True)
    t0 = time.time(); rows = []
    with Pool(NPROC, initializer=_init) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs, chunksize=20)):
            rows.append(r)
            if (i + 1) % 1000 == 0:
                el = time.time() - t0
                print(f'{i+1}/{len(jobs)}  {el:.0f}s  预计剩余 {el/(i+1)*(len(jobs)-i-1):.0f}s', flush=True)
                if not test and (i + 1) % 5000 == 0:
                    pd.DataFrame(rows).to_parquet(os.path.join(HERE, 'partial.parquet'), index=False)
    print(f'done {len(rows)} in {time.time()-t0:.0f}s', flush=True)
    if not test:
        from lab_common import Lab
        Lab.save(None, rows, 'agentB')
    else:
        pd.DataFrame(rows).to_parquet(os.path.join(HERE, 'test.parquet'), index=False)
