"""16 个参赛策略：14 个来自 GitHub 公开仓库（规则与参数照搬原作者，改动逐条写在 note 里），2 个是本仓库已有策略。
所有特征都是因果的（rolling / ewm / 截至第 i 行的切片），run.py 里的未来函数探针会逐一核验。"""
import warnings
import numpy as np, pandas as pd

TD = 252


# ---------- 通用工具 ----------
def sma(x, n):
    return x.rolling(n, min_periods=n).mean()


def wilder_atr(P, n):
    C, H, L = P['C'], P['H'], P['L']
    pc = C.shift()
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def logreg(C, n):
    """ln(close) 对 0..n-1 的滚动 OLS：返回 (日斜率, R²)"""
    y = np.log(C)
    j = pd.Series(np.arange(len(C)), C.index)
    S0, S1, S2 = (v.rolling(n, min_periods=n).sum() for v in (y, y.mul(j, axis=0), y * y))
    start = pd.Series(np.arange(len(C)) - n + 1, C.index)
    sxy = S1 - S0.mul(start, axis=0) - (n - 1) / 2 * S0
    sxx = n * (n * n - 1) / 12
    syy = S2 - S0 * S0 / n
    slope = sxy / sxx
    return slope, (sxy * sxy / (sxx * syy)).clip(0, 1)


def ew(mask):
    mask = np.asarray(mask, bool)
    return mask / mask.sum() if mask.any() else np.zeros(len(mask))


def top(score, elig, k):
    """elig 中 score 最高的 k 个（score 为 NaN 的不参与）"""
    s = np.where(elig & ~np.isnan(score), score, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    out = np.zeros(len(s), bool)
    if k > 0:
        out[np.argpartition(-s, k - 1)[:k]] = True
    return out


def month_ends(P, i):
    """截至第 i 天（含）的月末交易日行号；月末由交易日历决定（不含价格信息）"""
    return np.flatnonzero(P['me'][:i + 1])


def stock_mask(P, mem):
    m = mem.copy()
    m[P['spy']] = False
    return m


class Base:
    freq, name, src, note = 'M', '', '', ''

    def prepare(self, P):
        self.F = {k: np.asarray(v, float) for k, v in self.features(P).items()}
        self.S = {}

    def features(self, P):
        return {}


# ---------- 1. Clenow《Stocks on the Move》 ----------
class Clenow(Base):
    freq, name = 'WED', 'Clenow 动量 (Stocks on the Move)'
    src = 'teddykoker/blog 2019-05-19-momentum-strategy-from-stocks-on-the-move-in-python；skyte/momentum'
    note = ('按 teddykoker 的 backtrader 实现：90 日指数回归年化斜率×R² 排名；每周卖出跌出前 20% 或跌破 100 日线者；'
            'SPY>200 日线才买入，按排名依次买入 前 20%，仓位 = 0.1%×净值/ATR20，现金不足则跳过；隔周把前 20% 的仓位调回 ATR 目标。'
            '修正原代码 getposition(self.data) 的笔误（原意是逐只检查持仓）；另外卖出跌出指数的股票（时点成分股约束）。')

    def features(self, P):
        slope, r2 = logreg(P['C'], 90)
        return {'mom': (1 + slope) ** 252 * r2, 'ma100': sma(P['C'], 100), 'atr': wilder_atr(P, 20),
                'spyma': sma(P['C']['SPY'], 200), 'n': P['C'].notna().rolling(101, min_periods=1).sum()}

    def target(self, i, mem, w, P):
        F, c = self.F, P['C'].values[i]
        elig = stock_mask(P, mem) & (F['n'][i] > 100) & ~np.isnan(F['mom'][i])
        n_top = int(elig.sum() * 0.2)
        ranked = top(F['mom'][i], elig, n_top)
        order = np.flatnonzero(ranked)[np.argsort(-F['mom'][i][ranked])]
        held = w > 0
        w[held & (~ranked | (c < F['ma100'][i]) | ~mem | np.isnan(c))] = 0
        self.S['wk'] = self.S.get('wk', -1) + 1
        if c[P['spy']] < F['spyma'][i]:
            return w
        size = 0.001 * c / F['atr'][i]
        for j in order:
            cash = 1 - w.sum()
            if cash <= 0:
                break
            if w[j] == 0 and np.isfinite(size[j]) and size[j] <= cash:
                w[j] = size[j]
        if self.S['wk'] % 2 == 0:
            for j in order:
                cash = 1 - w.sum()
                if cash <= 0:
                    break
                if np.isfinite(size[j]) and size[j] - w[j] <= cash:
                    w[j] = size[j]
        return w


# ---------- 2. Minervini 趋势模板 ----------
class Minervini(Base):
    freq, name = 'W', 'Minervini 趋势模板'
    src = 'icedevil2001/mark_minervini_stock_screener (indicators.py TrendTemplate)'
    note = ('8 条条件照搬：价>150/200 日线；150>200 日线；200 日线高于 20 日前；50>150>200；价>50 日线；'
            '价≥1.3×52 周低；价≥0.75×52 周高（收盘价，260 日）；RS 评级≥70（原作定义：近 1 年日收益平均涨幅/平均跌幅，'
            '相对指数×100，指数用 SPY 代替 ^GSPC）；价≥$5、20 日均量≥50 万股。原作是筛选器、没有组合规则：'
            '每周五收盘等权持有全部入选股。')

    def features(self, P):
        C = P['C']
        r = C.pct_change(fill_method=None)
        roll = lambda x: x.rolling(251, min_periods=1).sum()
        full = r.notna().rolling(251, min_periods=251).sum() == 251            # 近 1 年日收益齐全
        g = roll(r.where(r >= 0)) / roll((r >= 0).astype(float))
        l = roll(-r.where(r < 0)) / roll((r < 0).astype(float))
        rs = (g / l).where(full)
        s50, s150, s200 = sma(C, 50), sma(C, 150), sma(C, 200)
        return {'rs': rs.div(rs['SPY'], axis=0) * 100, 'c': C, 's50': s50, 's150': s150, 's200': s200,
                's200_20': s200.shift(19), 'lo': C.rolling(260, min_periods=200).min(),
                'hi': C.rolling(260, min_periods=200).max(), 'vol20': P['V'].rolling(20, min_periods=20).mean()}

    def target(self, i, mem, w, P):
        f = {k: v[i] for k, v in self.F.items()}
        with np.errstate(invalid='ignore'):
            ok = ((f['c'] > f['s150']) & (f['c'] > f['s200']) & (f['s150'] > f['s200']) & (f['s200'] > f['s200_20'])
                  & (f['s50'] > f['s150']) & (f['c'] > f['s50']) & (f['c'] >= 1.3 * f['lo']) & (f['c'] >= 0.75 * f['hi'])
                  & (np.minimum(f['rs'], 200) >= 70) & (f['c'] >= 5) & (f['vol20'] >= 5e5))
        return ew(ok & stock_mask(P, mem))


# ---------- 3. HQM 高质量动量 ----------
class HQM(Base):
    freq, name = 'M', 'HQM 高质量动量'
    src = 'nickmccullum/algorithmic-trading-python (002_quantitative_momentum_strategy.ipynb)'
    note = ('1 年/6 月/3 月/1 月收益各算成分股内百分位（percentileofscore），取均值为 HQM 分，前 50 只等权。'
            '原作没有规定调仓频率，取月末；原 notebook 的 sort_values 漏了 inplace，按其文字说明取真正的前 50。')

    def features(self, P):
        C = P['C']
        return {f'r{n}': C / C.shift(n) - 1 for n in (252, 126, 63, 21)}

    def target(self, i, mem, w, P):
        elig = stock_mask(P, mem)
        R = np.vstack([self.F[f'r{n}'][i] for n in (252, 126, 63, 21)])
        elig &= ~np.isnan(R).any(0)
        if elig.sum() < 50:
            return np.zeros(len(mem))
        pct = pd.DataFrame(R[:, elig].T).rank(pct=True).values   # = percentileofscore(kind='rank')/100
        score = np.full(len(mem), np.nan)
        score[elig] = pct.mean(1)
        return ew(top(score, elig, 50))


# ---------- 4. 多因子动量（趋势 + FIP + 偏度，逆波动加权） ----------
class FIPMomentum(Base):
    freq, name = 'D', '多因子动量 (FIP+偏度, Top5)'
    src = 'tanish35/Momentum-Investing (main.py)'
    note = ('照搬原代码：SPY<200 日线则清仓；个股需>200 日线；动量 = backtrader Momentum 指标在 60/120/252 日的均值'
            '（注意这是价差 close−close[−n]，按美元计，原代码如此）且>0；FIP = 252 日内上涨天数占比；偏度 = 90 个价格的'
            '对数收益偏度（有偏估计）；得分 = 0.5×动量 + 0.5×FIP + 0.5×偏度，取前 5；权重 ∝ 1/126 日价格标准差；'
            '只有前 5 名单变化时才调仓（每日检查）。')

    def features(self, P):
        C = P['C']
        lr = np.log(C).diff()
        G1 = lr.rolling(89, min_periods=89).skew()
        g1 = G1 * (89 - 2) / np.sqrt(89 * 88)
        return {'c': C, 'ma': sma(C, 200), 'mom': sum(C - C.shift(n) for n in (60, 120, 252)) / 3,
                'fip': (C.pct_change(fill_method=None) > 0).astype(float).rolling(252, min_periods=252).sum() / 252,
                'skew': g1, 'sd': C.rolling(126, min_periods=126).std(ddof=0), 'spyma': sma(C['SPY'], 200)}

    def target(self, i, mem, w, P):
        F = self.F
        last = self.S.get('last', frozenset())
        if F['c'][i][P['spy']] < F['spyma'][i]:
            self.S['last'] = frozenset()
            return np.zeros(len(mem)) if w.any() else None
        with np.errstate(invalid='ignore'):
            ok = stock_mask(P, mem) & (F['c'][i] > F['ma'][i]) & (F['mom'][i] > 0) & ~np.isnan(F['fip'][i]) & ~np.isnan(F['skew'][i])
        score = 0.5 * F['mom'][i] + 0.5 * F['fip'][i] + 0.5 * F['skew'][i]
        pick = frozenset(np.flatnonzero(top(score, ok, 5)))
        if pick == last:
            return None
        self.S['last'] = pick
        tgt = np.zeros(len(mem))
        if pick:
            idx = list(pick)
            inv = 1 / F['sd'][i][idx]
            tgt[idx] = inv / inv.sum()
        return tgt


# ---------- 5. RSI(2) 均值回归 ----------
class RSI2(Base):
    freq, name = 'D', 'RSI(2) 均值回归 (Connors)'
    src = 'handiko/RSI-2-Stock-Trading-Strategy (RSI-2 Trading Strategy.ipynb)'
    note = ('照搬原作：收盘>200 日线且 RSI(2)<10 买入（RSI 用简单均值版）；收盘>前一日最高价卖出。原作是单股回测，'
            '组合化按 Connors 的通行做法：最多 10 个仓位、每个 10%，候选多于空位时优先 RSI 最低者（并列时 2 日跌幅大者优先）。')

    def features(self, P):
        C = P['C']
        d = C.diff()
        g = d.clip(lower=0).rolling(2, min_periods=2).mean()
        l = (-d.clip(upper=0)).rolling(2, min_periods=2).mean()
        return {'rsi': 100 - 100 / (1 + g / l), 'ma': sma(C, 200), 'c': C, 'ph': P['H'].shift(), 'r2': C / C.shift(2) - 1}

    def target(self, i, mem, w, P):
        F = self.F
        with np.errstate(invalid='ignore'):
            w[(w > 0) & ((F['c'][i] > F['ph'][i]) | ~mem | np.isnan(F['c'][i]))] = 0
            entry = stock_mask(P, mem) & (w == 0) & (F['c'][i] > F['ma'][i]) & (F['rsi'][i] < 10)
        free = 10 - int((w > 0).sum())
        new = np.flatnonzero(entry)
        if free > 0 and len(new):
            new = new[np.lexsort((F['r2'][i][new], F['rsi'][i][new]))][:free]   # RSI 相同（常见于 0）时先选 2 日跌幅大的
            w[new] = min(0.1, max(0.0, 1 - w.sum()) / len(new))
        return w


# ---------- 6–13. Quantpedia / paperswithbacktest 学术因子（多空原文，取多头腿） ----------
PWB = 'paperswithbacktest/awesome-systematic-trading static/strategies/'


class Mom12_1(Base):
    name, src = '动量 12-1 前 10% (Jegadeesh-Titman)', PWB + 'momentum-factor-effect-in-stocks.py'
    note = '过去 12 个月收益（跳过最近 1 个月）排名，前 10% 等权，月末调仓。原文多空，取多头腿。'

    def target(self, i, mem, w, P):
        me = month_ends(P, i)
        if len(me) < 13:
            return np.zeros(len(mem))
        C = P['C'].values
        score = C[me[-2]] / C[me[-13]] - 1
        el = stock_mask(P, mem)
        return ew(top(score, el, int((el & ~np.isnan(score)).sum() * 0.1)))


class High52(Base):
    name, src = '52 周新高接近度 (George-Hwang)', PWB + '52-weeks-high-effect-in-stocks.py'
    note = ('价格/52 周最高收盘价 前 30% 等权，持有 3 个月、每月换 1/3（三档重叠）。原实现先按 20 个行业加权平均再选 6 个行业，'
            '本数据没有时点行业分类，改用 George & Hwang (2004) 的个股版本。原文多空，取多头腿。')

    def features(self, P):
        return {'pr': P['C'] / P['C'].rolling(252, min_periods=252).max()}

    def target(self, i, mem, w, P):
        el = stock_mask(P, mem)
        pr = self.F['pr'][i]
        coh = self.S.setdefault('coh', [])
        coh.append(top(pr, el, int((el & ~np.isnan(pr)).sum() * 0.3)))
        del coh[:-3]
        return sum(ew(c & mem) for c in coh) / 3


class LowVol(Base):
    name, src = '低波动 前 25% (long-only)', PWB + 'low-volatility-factor-effect-in-stocks.py'
    note = '过去 252 日日收益波动率最低的 25% 等权，月末调仓（QC 实现：四分位、12×21 日窗口）。'

    def features(self, P):
        return {'vol': P['C'].pct_change(fill_method=None).rolling(251, min_periods=251).std()}

    def target(self, i, mem, w, P):
        el = stock_mask(P, mem)
        v = self.F['vol'][i]
        return ew(top(-v, el, int((el & ~np.isnan(v)).sum() * 0.25)))


class BAB(Base):
    name, src = '低贝塔 前 10% (Betting Against Beta 多头腿)', PWB + 'betting-against-beta-factor-in-stocks.py'
    note = ('252 日窗口对 SPY 估计贝塔，最低 10% 等权，月度调仓。原文多头腿加杠杆到贝塔 1、并做空高贝塔；'
            '本比赛不允许杠杆和做空，只取不加杠杆的多头腿。')

    def features(self, P):
        r = P['C'].pct_change(fill_method=None)
        m = r['SPY']
        cov = r.rolling(251, min_periods=251).cov(m)
        return {'beta': cov.div(m.rolling(251, min_periods=251).var(), axis=0)}

    def target(self, i, mem, w, P):
        el = stock_mask(P, mem)
        b = self.F['beta'][i]
        return ew(top(-b, el, int((el & ~np.isnan(b)).sum() * 0.1)))


class ResMom(Base):
    name, src = '残差动量 前 10% (Blitz-Huij-Martens)', PWB + 'residual-momentum-factor.py'
    note = ('36 个月月收益对市场回归取残差，t−12..t−2 残差和 / 残差标准差排名，前 10% 等权，月末调仓。'
            '原文用 Fama-French 三因子，本数据没有 SMB/HML，改用单因子（SPY）。原文多空，取多头腿。')

    def target(self, i, mem, w, P):
        me = month_ends(P, i)
        if len(me) < 37:
            return np.zeros(len(mem))
        C = P['C'].values[me[-37:]]
        r = C[1:] / C[:-1] - 1
        el = stock_mask(P, mem) & ~np.isnan(r).any(0)
        m = r[:, P['spy']]
        X = np.column_stack([np.ones(36), m])
        Y = r[:, el]
        e = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        e = e[-12:-1]
        score = np.full(len(mem), np.nan)
        score[el] = e.sum(0) / e.std(0, ddof=1)
        return ew(top(score, el, int(el.sum() * 0.1)))


class STReversal(Base):
    freq, name, src = 'W', '短期反转 周度 10 只', PWB + 'short-term-reversal-in-stocks.py'
    note = ('市值最大的 100 只中，上周跌幅最大的 10 只等权，每周调仓。市值用近 252 日平均成交额代替（无股本数据），'
            '原文多空，取多头腿。')

    def features(self, P):
        return {'r5': P['C'] / P['C'].shift(5) - 1, 'size': P['DV'].rolling(252, min_periods=126).mean()}

    def target(self, i, mem, w, P):
        big = top(self.F['size'][i], stock_mask(P, mem), 100)
        return ew(top(-self.F['r5'][i], big, 10))


class Cycle12(Base):
    name, src = '12 个月季节性 (Heston-Sadka)', PWB + '12-month-cycle-in-cross-section-of-stocks-returns.py'
    note = ('按"一年前同一个月"的月收益排名，前 10% 按市值加权（QC 实现），市值用 252 日平均成交额代替，月末调仓。'
            '原文多空，取多头腿。')

    def target(self, i, mem, w, P):
        me = month_ends(P, i)
        if len(me) < 13:
            return np.zeros(len(mem))
        C = P['C'].values
        score = C[me[-12]] / C[me[-13]] - 1
        el = stock_mask(P, mem)
        pick = top(score, el, int((el & ~np.isnan(score)).sum() * 0.1))
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dv = np.nanmean(P['DV'].values[max(0, i - 251):i + 1], 0)
        dv = np.where(pick & (dv > 0), dv, 0)
        return dv / dv.sum() if dv.sum() > 0 else np.zeros(len(mem))


class ConsistentMom(Base):
    name, src = '一致性动量 (t−7..t−1 与 t−6..t 双前 10%)', PWB + 'consistent-momentum-strategy.py'
    note = ('月末同时处于 (t−7→t−1) 与 (t−6→t) 收益前 10% 的股票组成一档，跳过 1 个月后持有 6 个月、每月一档（六档重叠），'
            '档内等权。原文多空，取多头腿。')

    def target(self, i, mem, w, P):
        me = month_ends(P, i)
        coh = self.S.setdefault('coh', [])
        if len(me) >= 8:
            C = P['C'].values
            a, b = C[me[-2]] / C[me[-8]] - 1, C[me[-1]] / C[me[-7]] - 1
            el = stock_mask(P, mem)
            n = int((el & ~np.isnan(a) & ~np.isnan(b)).sum() * 0.1)
            coh.append(top(a, el, n) & top(b, el, n))
        else:
            coh.append(np.zeros(len(mem), bool))
        del coh[:-7]
        live = coh[:-1][-6:]                              # 最新一档跳过 1 个月
        return sum(ew(c & mem) for c in live) / 6 if live else np.zeros(len(mem))


class TrendATH(Base):
    freq, name, src = 'D', '历史新高趋势跟踪 + ATR 止损', PWB + 'trend-following-effect-in-stocks.py'
    note = ('成交额前 100 且价>$5 的股票，收盘创 10 年（2520 日）新高买入；10 日 ATR 跟踪止损（只上移），收盘跌破止损则卖出；'
            '全部持仓每日调成等权 1/n（QC 实现）。QC 用盘中止损单，这里按收盘判断、次日收盘成交。')

    def features(self, P):
        C = P['C']
        return {'c': C, 'hi': C.shift().rolling(2520, min_periods=252).max(), 'atr': wilder_atr(P, 10), 'dv': P['DV']}

    def target(self, i, mem, w, P):
        F = self.F
        c = F['c'][i]
        sl = self.S.setdefault('sl', {})
        held = set(np.flatnonzero(w > 0)) & set(sl)
        for j in list(held):
            if not mem[j] or np.isnan(c[j]) or c[j] < sl[j]:
                held.discard(j); sl.pop(j)
            else:
                sl[j] = max(sl[j], c[j] - F['atr'][i][j])
        with np.errstate(invalid='ignore'):
            sel = top(F['dv'][i], stock_mask(P, mem) & (c > 5), 100)
            new = sel & (c >= F['hi'][i]) & ~np.isnan(F['atr'][i]) & (F['atr'][i] > 0)
        for j in np.flatnonzero(new):
            if j not in held:
                held.add(j); sl[j] = c[j] - F['atr'][i][j]
        for j in list(sl):
            if j not in held:
                sl.pop(j)
        tgt = np.zeros(len(mem))
        if held:
            tgt[list(held)] = 1 / len(held)
        return tgt


# ---------- 15–16. 本仓库已有策略 ----------
class C20(Base):
    name, src = '本仓库 C20 成交额加权 (strategy_final.py)', 'fengkaitian/chucun_scan strategy_final.py'
    note = '已冻结策略原样调用：成交额加权、单股上限 4%，月末调仓。'

    def target(self, i, mem, w, P):
        import strategy_final as S
        cols = P['C'].columns
        lo = max(0, i - 599)
        wt = S.target_weights(P['C'].iloc[lo:i + 1], (P['C'] * P['V']).iloc[lo:i + 1], set(cols[mem]))
        return wt.reindex(cols, fill_value=0.0).values


class V6Momentum(Base):
    name, src = '本仓库 v6 动量 Top20 + 趋势过滤 (momentum.py)', 'fengkaitian/chucun_scan momentum.py'
    note = '12-1 动量前 20 等权，SPY 低于 200 日线时空仓，月末调仓；改为只在时点成分股内选股。'

    def features(self, P):
        return {'spyma': sma(P['C']['SPY'], 200)}

    def target(self, i, mem, w, P):
        me = month_ends(P, i)
        C = P['C'].values
        if len(me) < 13 or C[i][P['spy']] < self.F['spyma'][i]:
            return np.zeros(len(mem))
        score = C[me[-2]] / C[me[-13]] - 1
        return ew(top(score, stock_mask(P, mem), 20))


# ---------- 基准 ----------
class SPY(Base):
    freq, name, src, note = 'D', 'SPY 买入持有', '基准', ''

    def target(self, i, mem, w, P):
        if w.any():
            return None
        t = np.zeros(len(mem)); t[P['spy']] = 1
        return t


class EqualWeight(Base):
    name, src, note = '成分股等权 (月调仓)', '基准', ''

    def target(self, i, mem, w, P):
        return ew(stock_mask(P, mem) & ~np.isnan(P['C'].values[i]))


CONTESTANTS = [Clenow, Minervini, HQM, FIPMomentum, RSI2, Mom12_1, High52, LowVol, BAB, ResMom, STReversal,
               Cycle12, ConsistentMom, TrendATH, C20, V6Momentum]
BENCHMARKS = [SPY, EqualWeight]
