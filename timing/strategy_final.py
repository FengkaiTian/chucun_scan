"""冻结的择时规则：标普 500 总收益指数 200 日均线过滤（研究配置 T02）。

规则（每日收盘决定次日仓位，只用引擎传入的最近 ≤600 行 df[tr, ret, rf]）
  - 当日总收益指数 tr 高于其最近 200 个交易日（含当日）的简单均值 → 仓位 1（满仓标普 500）；
  - 否则 → 仓位 0（全部现金，赚 ^IRX）；
  - 历史不足 200 行时 → 仓位 1。不加杠杆，没有其他参数。

文献
  - Brock, Lakonishok & LeBaron (1992, J. Finance)：价格对 200 日（及其他）均线的交叉规则；
  - Siegel《Stocks for the Long Run》、Faber (2007) 的均线择时：主要作用是降低波动与回撤，而非提高收益。
选择过程：15 个配置在训练期 2000–2020 评估，5 个候选（T02/T03/T08/T12/T15）在看验证期之前确定，
按固定规则「验证期 2021–2025 夏普 − 基准夏普最大」选中本配置（验证期夏普差 +0.11）。
"""
import numpy as np

N = 200


def target_position(df):
    tr = df['tr'].to_numpy(dtype=float)
    if len(tr) < N:
        return 1.0
    return float(tr[-1] > np.mean(tr[-N:]))
