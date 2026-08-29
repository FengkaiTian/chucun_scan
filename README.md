# 学术岗位每日扫描

每天早上 8 点自动扫描美国与加拿大的学术、科研、推广类岗位，去重后输出 xlsx。

针对精准农业 / 农业遥感 / 精准畜牧方向，覆盖 36 个目标职位类别
（助理教授、博后、研究科学家、推广专员、设施与项目经理等）。

---

## 快速开始

> **必须用「Anaconda Prompt」，不要用普通 cmd。**
> Anaconda 只在 Anaconda Prompt 里才进 PATH。普通 cmd 里的 `python` 会命中
> 微软商店的空壳程序，静默退出、什么也不做、也不报错。

```bat
cd /d "C:\Users\<你>\OneDrive\Desktop\JOBSCAN"

pip install -r requirements_jobscan.txt
copy job_config.example.json job_config.json

python job_scan.py --check-sources
```

**第一次务必先跑 `--check-sources`。** 注册表里的 endpoint 未经逐一验证，
它会逐源探测并打印健康表，把坏掉的挑出来。

确认可用后：

```bat
run_job_scan.bat
```

---

## 每天 8 点自动跑

```bat
schtasks /create /tn "JobScan" /tr "\"C:\完整\路径\run_job_scan.bat\"" /sc daily /st 08:00 /f
```

注册后在「任务计划程序」里勾上 **「如果错过计划开始时间，请尽快启动任务」**
——命令行改不了这项，但它决定了关机一天后会不会补跑。

`run_job_scan.bat` 会自己定位 Anaconda 的解释器（查 `CONDA_PREFIX` 及常见安装位置）。
若自动探测失败，在 Anaconda Prompt 里执行 `where python`，把路径填到该文件的
`PY_OVERRIDE=` 那一行。

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `job_scan.py` | 主脚本 |
| `job_sources.json` | **源注册表 —— 唯一需要经常手改的文件**，改它不用动代码 |
| `job_titles.json` | 职位名称正则、领域关键词、初筛规则 |
| `job_config.example.json` | API key 模板，复制为 `job_config.json` 后填写 |
| `run_job_scan.bat` | 启动器 |
| `job_output/` | 输出目录（已 gitignore） |
| `jobs.db` | 去重与状态追踪用的 SQLite（已 gitignore） |

---

## 输出

`job_output/job_scan_latest.xlsx`，五个工作表：

| 工作表 | 内容 |
|---|---|
| 今日新增 | 首次出现的岗位 |
| 全部在招 | 当前所有活跃岗位 |
| 已消失 | 连续 3 次未再出现 —— **判断岗位关闭的唯一信号** |
| 初筛丢弃样本 | 抽样被丢弃的岗位 + 原因，用来检查初筛是否过严 |
| 源健康度 | 每个源的状态、抓取条数、错误与修复建议 |

**表里不做相关性评分**，但携带职位描述原文（`职位描述` 列），交给下游 AI 判断。

---

## 初筛逻辑

三层，全部可在 `job_titles.json` 里调整：

1. **标题负面词** —— 标题直接写明护理 / 音乐 / 法学之类
2. **上下文负面词** —— 标题看不出，但院系或正文明显不对口
3. **领域关键词** —— 必须沾边农业 / 遥感 / 作物 / 土壤 / 地理空间

两条防误杀的豁免：

- `job_sources.json` 里标了 `domain_trusted` 的专业板整源豁免
- 描述近乎为空时（RSS 常见）保留并标记 `需人工确认`

---

## 常用命令

```bat
python job_scan.py                      正常扫描
python job_scan.py --check-sources      逐源健康探测
python job_scan.py --probe "<url>"      单独探测一个 URL
python job_scan.py --only asabe         只跑指定源（忽略 enabled）
python job_scan.py --self-test          离线自测，不联网
```

---

## 覆盖范围

**49 个源 / 18 种抓取器**：RSS、Workday、Greenhouse、Lever、Ashby、PeopleAdmin、
PageUp、Oracle HCM、iCIMS、SmartRecruiters、Workable、Recruitee、
Adzuna、USAJOBS、JSearch、Careerjet、Jooble，以及遵守 robots.txt 的 HTML 兜底。

**关于 LinkedIn / Indeed / Glassdoor / ZipRecruiter**：这四家都已关闭面向个人
求职者的公开 API，直接爬取违反 ToS 且会被封。走 JSearch（第三方持牌聚合转售，
正好覆盖这四家）。注意 JSearch 不含 ATS 岗位，而高校教职大量挂在 Workday /
PeopleAdmin 上——两者互补，都需要。
