"""策略大排行 · 共用框架（所有 agent 必须用它算成绩，保证可比）

规则（不许改）：
- 股票池：当时的标普500 成分股（时点成分：日期 t 属于 t 所在月份之前最近一个月末的名单），只用有行情的。
- 入场：信号日 t 收盘后决定，t+1 收盘买入（ENTRY_LAG=1）。
- 同一只股票，持仓期间的新信号忽略（不重叠）。
- 成本：每笔来回扣 0.10%。
- 时期：训练 2000–2019（只在这里挑/调参），验证 2020–2024（只看不调），
  保留 2025–2026（谁都不许看，只有主会话最后用 holdout_eval 算一次）。
- t 值：按信号日聚类（同一天的多笔交易算一组）。

用法：
    from lab_common import Lab
    lab = Lab()                       # 约 20 秒，载入面板
    sig = (lab.rsi(2) < 10) & lab.member      # T×N 布尔矩阵（numpy），信号日 = True
    res = lab.evaluate(sig, hold=5, name='RSI2<10 持有5天', family='均值回归', params='...')
    # res 是 dict：train_* / valid_* 指标。把 res 追加到列表，最后 lab.save(rows, 'agentA')
自定义出场：lab.evaluate_exit(sig, exit_fn_matrix...) 见 evaluate_custom。
"""
import os
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data2000')
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
COST = 0.10            # % 每笔来回
TRAIN = (2000, 2019); VALID = (2020, 2024); HOLD_OUT_FROM = 2025


@njit(cache=True)
def _trades(sig, C, hold, lag):
    """不重叠地把信号变成交易：返回 (t_signal, j, ret%)；ret 从 t+lag 收盘到 t+lag+hold 收盘"""
    T, N = sig.shape
    out_t = np.empty(T * 4 + 1000000, np.int64); out_j = np.empty_like(out_t); out_r = np.empty(out_t.shape[0])
    k = 0
    for j in range(N):
        busy = -1
        for t in range(T - lag - hold):
            if t <= busy or not sig[t, j]:
                continue
            a = C[t + lag, j]; b = C[t + lag + hold, j]
            if not (a > 0) or not (b > 0):
                continue
            if k >= out_t.shape[0]:
                break
            out_t[k] = t; out_j[k] = j; out_r[k] = (b / a - 1) * 100
            k += 1
            busy = t + hold
    return out_t[:k], out_j[:k], out_r[:k]


@njit(cache=True)
def _trades_custom(sig, C, exit_mat, maxh, lag):
    """自定义出场：exit_mat[t, j] = True 表示第 t 天收盘卖出条件成立（从入场次日起检查），最长 maxh 天"""
    T, N = sig.shape
    out_t = np.empty(T * 4 + 1000000, np.int64); out_j = np.empty_like(out_t); out_r = np.empty(out_t.shape[0])
    out_d = np.empty(out_t.shape[0], np.int64)
    k = 0
    for j in range(N):
        busy = -1
        for t in range(T - lag - maxh):
            if t <= busy or not sig[t, j]:
                continue
            e = t + lag; a = C[e, j]
            if not (a > 0):
                continue
            x = e + maxh
            for u in range(e + 1, e + maxh + 1):
                if exit_mat[u, j]:
                    x = u; break
            b = C[x, j]
            if not (b > 0):
                continue
            if k >= out_t.shape[0]:
                break
            out_t[k] = t; out_j[k] = j; out_r[k] = (b / a - 1) * 100; out_d[k] = x - e
            k += 1
            busy = x - lag
    return out_t[:k], out_j[:k], out_r[:k], out_d[:k]


def _seal(year, t, r, sp):
    """保留期（2025+）成绩：编码后存起来，只有主会话最后解开。agent 不要解码。"""
    import json, base64, zlib
    m = year[t] >= HOLD_OUT_FROM
    if m.sum() == 0:
        d = {'n': 0}
    else:
        x = r[m] - COST
        d = {'n': int(m.sum()), 'win': float((x > 0).mean() * 100), 'avg': float(x.mean()),
             't': float(_cluster_t(x, t[m])), 'excess': float((r[m] - COST - sp[m]).mean())}   # 2026-11-18 起超额也扣成本
    return base64.b64encode(zlib.compress(json.dumps(d).encode())).decode()


def unseal(s):
    import json, base64, zlib
    return json.loads(zlib.decompress(base64.b64decode(s)).decode())


def _cluster_t(y, g):
    if len(y) < 5:
        return np.nan
    mu = y.mean()
    s = pd.Series(y - mu).groupby(g).sum().values
    G = len(s)
    se = np.sqrt((s ** 2).sum() * G / max(G - 1, 1)) / len(y)
    return mu / se if se > 0 else np.nan


class Lab:
    def __init__(self, membership_file='membership.csv.gz', bench='gspc'):
        """membership_file：成分股名单文件（membership_v2.csv.gz 是按入指日期修正过的）；
        bench：'gspc' 标普500 价格指数，'spy_tr' 含分红的 SPY 总回报"""
        rd = lambda k: pd.read_parquet(os.path.join(DATA, f'ohlcv_{k}.parquet'))
        C = rd('close')
        self.dates = C.index; self.tickers = list(C.columns)
        self.C = C.values.astype(np.float64)
        self.O = rd('open')[self.tickers].values.astype(np.float64)
        self.H = rd('high')[self.tickers].values.astype(np.float64)
        self.L = rd('low')[self.tickers].values.astype(np.float64)
        self.V = rd('volume')[self.tickers].values.astype(np.float64)
        self.spx = pd.read_parquet(os.path.join(DATA, 'index_gspc.parquet' if bench == 'gspc' else 'index_spy_tr.parquet')).iloc[:, 0].reindex(self.dates).ffill().values
        self.vix = pd.read_parquet(os.path.join(DATA, 'index_vix.parquet')).iloc[:, 0].reindex(self.dates).ffill().values
        self.year = self.dates.year.values
        # 时点成分股
        m = pd.read_csv(os.path.join(DATA, membership_file), parse_dates=['month_end'])
        col = {t: i for i, t in enumerate(self.tickers)}
        me_sorted = np.sort(m.month_end.unique())
        sets = {me: np.zeros(len(self.tickers), bool) for me in me_sorted}
        for me, t in zip(m.month_end.values, m.ticker.values):
            if t in col:
                sets[me][col[t]] = True
        mem = np.zeros(self.C.shape, bool)
        idx = np.searchsorted(me_sorted, self.dates.values, side='left') - 1   # 严格早于当天的最近月末
        for i, k in enumerate(idx):
            if k >= 0:
                mem[i] = sets[me_sorted[k]]
        self.member = mem & np.isfinite(self.C)
        self.member[self.year < 2000] = False
        self._cache = {}

    # ── 常用指标（T×N numpy）。agent 可以自己加更多 ──
    def df(self, a):
        return pd.DataFrame(a, index=self.dates)

    def rsi(self, n=14):
        key = ('rsi', n)
        if key not in self._cache:
            d = self.df(self.C).diff()
            g = d.clip(lower=0).ewm(com=n - 1, min_periods=n).mean(); l = (-d.clip(upper=0)).ewm(com=n - 1, min_periods=n).mean()
            self._cache[key] = (100 - 100 / (1 + g / l)).values
        return self._cache[key]

    def ma(self, n, src=None):
        return self.df(self.C if src is None else src).rolling(n).mean().values

    def ret(self, n):
        return (self.C / np.roll(self.C, n, axis=0) - 1) * 100 * np.where(np.arange(len(self.C))[:, None] >= n, 1, np.nan)

    def macd_hist(self):
        if 'macd' not in self._cache:
            c = self.df(self.C)
            dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
            self._cache['macd'] = (dif - dif.ewm(span=9, adjust=False).mean()).values
        return self._cache['macd']

    def atr_pct(self, n=14):
        c1 = np.roll(self.C, 1, axis=0)
        tr = np.nanmax(np.stack([self.H - self.L, np.abs(self.H - c1), np.abs(self.L - c1)]), axis=0)
        return self.df(tr).ewm(span=n, adjust=False).mean().values / self.C * 100

    def spx_series(self, kind, n):
        """大盘序列，返回长度 T 的一维数组；用 [:, None] 广播到 T×N"""
        s = pd.Series(self.spx, index=self.dates)
        if kind == 'ret':
            return (s / s.shift(n) - 1).values * 100
        if kind == 'dist_ma':
            return (s / s.rolling(n).mean() - 1).values * 100
        raise ValueError(kind)

    # ── 成绩 ──
    def _summ(self, t_idx, r, days=None):
        out = {}
        yr = self.year[t_idx]
        spx_ok = True
        for lab, lo, hi in (('train', *TRAIN), ('valid', *VALID)):
            m = (yr >= lo) & (yr <= hi)
            x = r[m] - COST
            if m.sum() == 0:
                out.update({f'{lab}_n': 0}); continue
            g = t_idx[m]
            out.update({f'{lab}_n': int(m.sum()), f'{lab}_days': int(len(np.unique(g))),
                        f'{lab}_per_year': m.sum() / (hi - lo + 1),
                        f'{lab}_win': float((x > 0).mean() * 100), f'{lab}_avg': float(x.mean()),
                        f'{lab}_med': float(np.median(x)), f'{lab}_p5': float(np.percentile(x, 5)),
                        f'{lab}_t': float(_cluster_t(x, g))})
            if days is not None:
                out[f'{lab}_hold'] = float(days[m].mean())
                out[f'{lab}_per_day'] = float(x.mean() / max(days[m].mean(), 1e-9))
            # 超额：减同期标普500（同样入场/出场日）
            if days is None:
                pass
        return out

    def evaluate(self, sig, hold, name, family='', params='', lag=1):
        """固定持有 hold 天。sig 会自动 & member"""
        s = np.ascontiguousarray(sig & self.member)
        t, j, r = _trades(s, self.C, int(hold), int(lag))
        sealed = _seal(self.year, t, r, (self.spx[t + lag + hold] / self.spx[t + lag] - 1) * 100)
        keep = self.year[t] < HOLD_OUT_FROM
        t, j, r = t[keep], j[keep], r[keep]
        res = dict(name=name, family=family, params=params, hold=hold, sealed=sealed)
        res.update(self._summ(t, r, np.full(len(t), hold, float)))
        # 同期标普500 超额
        sp = (self.spx[t + lag + hold] / self.spx[t + lag] - 1) * 100
        for lab, lo, hi in (('train', *TRAIN), ('valid', *VALID)):
            m = (self.year[t] >= lo) & (self.year[t] <= hi)
            if m.sum():
                ex = r[m] - COST - sp[m]   # 2026-11-18 修正：超额此前漏扣 0.10% 成本
                res[f'{lab}_excess'] = float(ex.mean()); res[f'{lab}_t_excess'] = float(_cluster_t(ex, t[m]))
        return res

    def evaluate_custom(self, sig, exit_mat, maxh, name, family='', params='', lag=1):
        """自定义出场：exit_mat 是 T×N 布尔矩阵（某天收盘满足就在当天收盘卖），最长 maxh 天"""
        s = np.ascontiguousarray(sig & self.member)
        t, j, r, d = _trades_custom(s, self.C, np.ascontiguousarray(exit_mat.astype(np.bool_)), int(maxh), int(lag))
        sealed = _seal(self.year, t, r, (self.spx[t + lag + d] / self.spx[t + lag] - 1) * 100)
        keep = self.year[t] < HOLD_OUT_FROM
        t, j, r, d = t[keep], j[keep], r[keep], d[keep]
        res = dict(name=name, family=family, params=params, hold=f'≤{maxh}', sealed=sealed)
        res.update(self._summ(t, r, d.astype(float)))
        sp = (self.spx[t + lag + d] / self.spx[t + lag] - 1) * 100
        for lab, lo, hi in (('train', *TRAIN), ('valid', *VALID)):
            m = (self.year[t] >= lo) & (self.year[t] <= hi)
            if m.sum():
                ex = r[m] - COST - sp[m]   # 2026-11-18 修正：超额此前漏扣 0.10% 成本
                res[f'{lab}_excess'] = float(ex.mean()); res[f'{lab}_t_excess'] = float(_cluster_t(ex, t[m]))
        return res

    def random_signals(self, density, seed):
        """随机信号（同样稀疏度），用来当排行榜里的噪音对照"""
        rng = np.random.default_rng(seed)
        return (rng.random(self.C.shape) < density) & self.member

    def save(self, rows, agent):
        df = pd.DataFrame(rows)
        df['agent'] = agent
        p = os.path.join(OUT, f'{agent}.parquet')
        df.to_parquet(p, index=False)
        return p
