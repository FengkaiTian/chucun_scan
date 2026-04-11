# S&P 500 Daily Signal Scanner

每日自动扫描标普500全成分股，基于技术指标组合评级，生成 HTML 报告并发布到 GitHub Pages。

**网站：** https://fengkaitian.github.io/chucun_scan/

---

## Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff', 'primaryBkg': '#ffffff', 'secondaryBkg': '#f0f0f0', 'tertiaryBkg': '#ffffff', 'mainBkg': '#ffffff'}}}%%
flowchart TD
    A([🕓 04:30 AM\nloop_scan.py 触发]) --> B

    B[daily_scan.py 启动] --> C & D & E

    C[📋 加载 validated_combos.json\n206种验证组合\nSSS / S / A / B 分级] --> F
    D[🌐 Wikipedia 获取\nS&P 500 成分股列表\n~503只 + 板块信息] --> F
    E[📈 yfinance 批量下载\n15个月 OHLCV 数据\nSPY大盘 + VIX恐慌指数] --> F

    F{市场环境过滤} -->|VIX ≥ 30\n极度恐慌| G([🚫 暂停出信号\n生成空报告])
    F -->|VIX < 30\n正常/高波动| H

    H[逐股扫描 ~500只] --> I

    I{基础信号硬过滤\n必须同时满足} -->|不满足| J([跳过])
    I -->|✅ 通过| K

    I1["MACD柱状图 < 0\n且连续2天收窄\n（空头力量减弱）"]
    I2["RSI 前日 < 30\n且今日回升\n（超卖反弹）"]
    I1 & I2 --> I

    K[计算37个附加条件] --> L

    subgraph 37个附加条件
        K1["动量类\nRSI<25/20, CCI<-100\nWilliams%R<-80\nStochRSI交叉"]
        K2["资金流类\nCMF>0/上升\nOBV上升\nMFI<20/30"]
        K3["价格位置类\n近52周低, Keltner下轨\nBB下轨, 距MA50/MA200偏离"]
        K4["量价类\n放量1.2x/2x, 缩量0.8x"]
        K5["K线形态\n锤子线, 多头吞没\n十字星底, 跳空收正\nRSI底背离"]
    end

    K --> K1 & K2 & K3 & K4 & K5

    L[匹配验证组合\n穷举1626种→保留206种\n6年2020-2025全成分股回测\nT+1/3/5/10全周期跑赢baseline] --> M

    M{命中最高等级}
    M -->|SSS| N1["🔥 SSS 高收益区\n均收>2%\n~74%胜率"]
    M -->|S| N2["🟣 S级 最强\n5日胜率≥80%"]
    M -->|A| N3["🔴 A级 强\n5日胜率70-80%"]
    M -->|B| N4["🔵 B级 有效\n5日胜率65-70%"]
    M -->|无匹配| J

    N1 & N2 & N3 & N4 --> O

    O[汇总结果\n按级别+胜率排序\n统计板块分布] --> P

    P[生成 index.html\n展示信号列表、匹配条件\nT+1/3/5/10历史胜率\n止损位 = MA20×0.97] --> Q

    Q[git commit + push\n推送到 GitHub Pages] --> R([✅ 网站更新完成])
```

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `daily_scan.py` | 主扫描脚本，运行后生成报告并推送到 GitHub Pages |
| `loop_scan.py` | 定时脚本，每天 04:30 AM 自动触发 |
| `validated_combos.json` | 经6年回测验证的206种信号组合定义 |
| `run_scan.bat` | 手动运行入口 |
| `index.html` | 最新报告（自动生成，每日覆盖） |

---

## 运行方式

**手动运行一次：**
```
python daily_scan.py
```

**开启每日 04:30 AM 自动扫描：**
```
python loop_scan.py
```

---

## 选股逻辑说明

**核心策略：在 MACD 金叉之前入场**

当 MACD 柱状图在零轴下方连续收窄（空头力量减弱），同时 RSI 从超卖区（<30）开始反弹，形成基础信号。在此基础上叠加 37 个附加条件过滤，只推荐在 T+1、T+3、T+5、T+10 全周期均跑赢随机 baseline 的组合。

**评级体系：**

| 等级 | 5日胜率 | 5日中位收益 |
|---|---|---|
| **SSS** 高收益区 | ~74% | +2.0% |
| **S** 最强 | ≥80% | +1.2% |
| **A** 强 | 70–80% | +1.7% |
| **B** 有效 | 65–70% | +1.7% |

> 建议持仓：5个交易日 · 止损参考：MA20 × 0.97  
> 仅供研究参考，不构成投资建议。历史回测不代表未来表现。
