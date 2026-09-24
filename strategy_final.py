"""冻结策略：成交额加权的标普 500 成分股组合（单股上限 4%）——研究配置 C20。

规则（每月最后一个交易日收盘调仓，只用引擎传入的 close / dvol / members）
  1. 股票池：当月时点成分股中，最近 253 个交易日收盘价完整的股票（不含 SPY）。
  2. 权重 ∝ 近 252 个交易日平均成交额（收盘价×成交量；成交额为 0 视为缺失）。
  3. 单股权重上限 4%：截断后重新归一化，重复 30 次；满仓，不持现金、不持 SPY。

依据与定位
  - 成交额 ≈ 市值 × 换手率，用作市值权重的代理（无股本数据时的替代），使组合贴近市值加权的
    SPY、跟踪误差低（开发期约 3%）。相对 SPY 的主动暴露是"高换手/高关注度、贝塔略高于 1"的
    大盘股（贝塔约 1.1）。这一方向与换手率文献（Datar, Naik & Radcliffe 1998；
    Lee & Swaminathan 2000：高换手股票未来收益偏低）相反，属于 2010–2025 的经验发现而非
    理论预测；详见 RESEARCH.md 的风险说明。
  - 单股上限用于限制个别超高成交额股票的集中度（不设上限时开发期曾出现单股 >10% 的情形）。
选择过程：25 个配置在 2010–2020 开发期评估，5 个候选进入 2021–2025 验证期，
本配置是唯一在验证期信息比率为正的候选（开发期 IR 0.69，验证期 IR 0.33）。
"""
import pandas as pd

CAP, WIN, HIST = 0.04, 252, 253


def target_weights(close, dvol, members):
    px = close[[t for t in close.columns if t in members and t != 'SPY']]
    if len(px) < HIST:
        return pd.Series(dtype=float)
    ok = px.iloc[-HIST:].notna().all()
    u = ok[ok].index
    dv = dvol[u].astype(float).where(lambda x: x > 0).iloc[-WIN:].mean().dropna()
    w = dv / dv.sum()
    for _ in range(30):
        w = w.clip(upper=CAP)
        w = w / w.sum()
    return w
