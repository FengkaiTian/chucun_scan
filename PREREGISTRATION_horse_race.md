# 预注册：GitHub 公开选股策略擂台（2026 年美股）

在下载任何含 2026 年 OHLCV 的数据之前提交（以 git 历史为准）。数据下载后不再改动策略规则、参数和检验口径；
如果发现程序错误必须修改，会在结果报告里逐条记录修改内容和修改前后的数字。

## 目的
从 GitHub 公开仓库选出十几个规则明确的美股选股策略，**照搬原作者的规则和参数**（本方不调参），
放进同一个回测引擎，用最近的美股行情比较，回答"谁的策略最好"，并检验最好的那个是否真的跑赢了 SPY。

## 参赛策略（16 个；规则与改动详见 `horse_race/strategies.py` 各类的 `note`）

| # | 策略 | 调仓 | GitHub 来源（克隆时的提交） | 文献依据 |
|---|---|---|---|---|
| 1 | Clenow《Stocks on the Move》 | 每周三 | teddykoker/blog `46c6694768`；skyte/momentum `9deee370b2` | Clenow (2015)，书籍，非同行评审 |
| 2 | Minervini 趋势模板 | 每周五 | icedevil2001/mark_minervini_stock_screener `403f7aed21` | Minervini (2013)，书籍，非同行评审 |
| 3 | HQM 高质量动量 Top50 | 月末 | nickmccullum/algorithmic-trading-python `e1e3c2e586` | 无（教学项目） |
| 4 | 多因子动量（FIP + 偏度）Top5 | 每日 | tanish35/Momentum-Investing `0af770e023` | 无 |
| 5 | RSI(2) 均值回归 | 每日 | handiko/RSI-2-Stock-Trading-Strategy `0466c03c2b` | Connors & Alvarez (2009)，书籍，非同行评审 |
| 6 | 动量 12-1 前 10% | 月末 | paperswithbacktest/awesome-systematic-trading `4e23dd84c9` | Jegadeesh & Titman (1993), *J. Finance* 48(1) |
| 7 | 52 周新高接近度 前 30% | 月末（3 档重叠） | 同上 | George & Hwang (2004), *J. Finance* 59(5) |
| 8 | 低波动 前 25% | 月末 | 同上 | Blitz & van Vliet (2007), *J. Portfolio Management* 34(1) |
| 9 | 低贝塔 前 10%（BAB 多头腿） | 月末 | 同上 | Frazzini & Pedersen (2014), *J. Financial Economics* 111(1) |
| 10 | 残差动量 前 10% | 月末 | 同上 | Blitz, Huij & Martens (2011), *J. Empirical Finance* 18(3) |
| 11 | 短期反转（大市值 100 只中上周最差 10 只） | 每周五 | 同上 | Jegadeesh (1990), *J. Finance* 45(3) |
| 12 | 12 个月季节性 前 10% | 月末 | 同上 | Heston & Sadka (2008), *J. Financial Economics* 87(2) |
| 13 | 一致性动量 前 10% | 月末（6 档重叠） | 同上 | Quantpedia 条目，未核实同行评审出处 |
| 14 | 历史新高 + ATR 跟踪止损 | 每日 | 同上 | Wilcox & Crittenden (2005)，工作论文，非同行评审 |
| 15 | 本仓库 C20（已冻结） | 月末 | 本仓库 `strategy_final.py` | 见 RESEARCH.md |
| 16 | 本仓库 v6 动量 Top20 + 趋势过滤 | 月末 | 本仓库 `momentum.py` | Jegadeesh & Titman (1993) |

**入选标准**：规则能完全写成代码、只需要价格和成交量（本仓库没有时点基本面数据，价值/质量/Piotroski 这类策略
会引入前视偏差，所以不纳入）、适用于美股个股、原作者给出了明确参数。
**考察过但没有入选**：je-suis-tm/quant-trading（单一资产择时，不是选股）、digitalaw/8-Quant-Algos（Quantopian 平台代码，依赖其数据接口）。

## 统一回测设定
- 股票池：**每日时点**标普 500 成分股（由 Wikipedia 变更表回推，快照 `results/wiki_test_*.html`，2026-09-24 抓取）
- 行情：Yahoo 复权日线 OHLCV，由 GitHub Actions（`.github/workflows/horse_race.yml`）下载，只下载到上一个完整交易日
- 成交：信号日收盘计算，**下一交易日收盘成交**（主口径）；单边成本 10bp；现金按 3 个月国债利率（^IRX）计息
- 只做多、不加杠杆（权重 ≥ 0、合计 ≤ 1）。原文是多空组合的，只取多头腿
- 所有策略 2010-01-01 起连续模拟，各评估区间都从这一次连续模拟里切出来
- 基准：SPY 买入持有；另列成分股等权作为参照

## 评估
- **主区间：2026-01-02 至数据最后一个交易日**
- **主指标：Sharpe 比率**（日收益减国债利率，年化 ×√252）。按它排名次
- 次要区间（仅描述）：近 12 个月、2025 年、2021–2025、2011–2025（2021 年以前有幸存者偏差，见 PREREGISTRATION.md）
- 次要指标：总收益、年化收益、波动、最大回撤、对 SPY 的年化超额、IR、β、α、年换手

## 检验（只在主区间做）
1. 每个策略对 SPY：日超额收益的 Newey-West 单侧 t 检验（`harness.nw_test`，与之前的 2026 检验同一口径），
   16 个 p 值做 Holm (1979) 校正
2. **Hansen (2005) SPA 检验**：H0 为"16 个策略中没有任何一个的期望日收益高于 SPY"；平稳自助法
   （Politis & Romano 1994），平均块长 10，10000 次，种子 20260924；报告 consistent p 值
3. 平稳自助法（块长 10，2000 次）估计每个策略"Sharpe 排第一"的概率，以及 Sharpe 的 90% 区间

**判定**：SPA 的 p < 0.05，才说"最好的策略显著跑赢 SPY"；否则 Sharpe 排名只是描述，不能当作策略有效的证据。

## 敏感性（不参与判定）
当日收盘成交（lag = 0，多数 GitHub 原作者采用这种回测方式）；零交易成本。

## 事先披露的局限
- 统计功效低：主区间约 180 个交易日，某个策略要单独显著跑赢 SPY，年化 IR 大约要到 1.9 以上
- 2026 年的 SPY 行情和 C20 的表现在之前的一次性检验中已经看过（见 REVIEW.md）。另外 14 个 GitHub 策略
  本方从来没有在 2026 年数据上运行过，它们的规则和参数都来自原作者
- GitHub 原作者可能已经用历史数据挑选过参数，这不影响 2026 年样本外的比较
- 原作为单股回测或筛选器的策略（RSI(2)、Minervini），组合化规则是本方补的，在 `note` 里写明
- 市值用近 252 日平均成交额代替（短期反转、12 个月季节性），因为没有股本数据
