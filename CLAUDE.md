# 学术岗位扫描 —— 项目背景

这个仓库原本是股票扫描器（已删）。现在它只做一件事：**帮 Fengkai Tian 找 2026–27 招聘季的学术岗位。**
他很急——博士 2026-05 毕业，现在是博后第一年，需要尽快拿到教职或下一站。

新开的会话请先读完这份文件再动手。

---

## 候选人一句话

精准农业 / 农业遥感 + 精准畜牧。UAV 多传感器（RGB、VNIR-SWIR 高光谱、热红外、LiDAR）
做大田作物全生育期性状反演；另一条线是猪舍机器人成像做母猪发情检测。
统计本科 + 数据科学硕士（HPC）+ 生物工程博士。8 篇已发表（7 篇一作）。

完整画像在 `cv_profile.json`（**不在仓库里**，见下文"本地文件"）。

## 两个硬约束（每次判断岗位都要过一遍）

1. **只看美国和加拿大。**
2. **中国籍，需要签证担保。** 联邦编制岗（USAJOBS、USDA-ARS 正式职位、多数 ORISE）
   要求美国公民身份，直接排除；要求 security clearance 或受 ITAR / 出口管制的同理。
   普通高校教职与博后没问题——高校常规办 H-1B / J-1，**不要因为启事没提签证就降级**。

   注：佛罗里达 SB 846 那条曾经写进过判据，后来删掉了。该法条针对的是从中国域内
   招聘，人已在美国的情形不适用，留着只会造成误判。别再加回去。

---

## 两个脚本，跑的顺序是固定的

```
python job_scan.py          # ① 抓取 + 初筛 → job_output/job_scan_latest.xlsx
python job_rate.py          # ② Gemini 评级 + 核对在招 → job_output/job_rated_latest.xlsx
```

②读的是①写进 `jobs.db` 的东西，先跑②会没数据可评。
Windows 上也可以双击 `run_job_scan.bat` / `run_job_rate.bat`（内含 conda 解释器探测）。

### ① `job_scan.py` —— 抓取与初筛

- 18 种抓取器（rss / workday / greenhouse / lever / ashby / peopleadmin / oracle_hcm /
  pageup / icims / smartrecruiters / workable / recruitee / adzuna / jsearch / careerjet /
  jooble / usajobs / html），源注册表是 `job_sources.json`（91 条，启用 52）。
- 初筛规则在 `job_titles.json`：36 个职位类别 + 178 个领域关键词 + 负面词。
  **`require_domain_for_all: true`** 是主开关——所有岗位都必须命中至少一个领域关键词。
  关掉它会把全美加所有助理教授岗照单全收（实测 10 个不相关岗位会漏进来 8 个）。
- 去重与状态追踪用 SQLite `jobs.db`：`first_seen` / `last_seen` / 连续 3 次未见标 `gone`。
  「今日新增」和「已消失」全靠这张表——**别删 jobs.db**。
- 常用参数：`--check-sources`（源健康探测）、`--only <id>`（单源调试）、
  `--probe <url>`（探测任意 URL 的状态码与内容类型）、`--self-test`（离线自测）。

最近一次实跑：**52 个源里 35 个 OK，抓到 2503 条，初筛后留 363 条（当日新增 140）。**

### ② `job_rate.py` —— Gemini 评级

- 拿 `cv_profile.json` 当判据，每批 8 条丢给 Gemini，判 **绿色 / 黄色 / 红色 / 黑色**，
  理由一起写回 `jobs.db`，再出一份 xlsx（绿色、黄色各一张表）。
- **在招状态以 HTTP 为准，不问模型**：404/410/跳转/页面写着 "no longer accepting" 都是
  确定性判定；只有页面打得开、措辞又含糊时才把正文交给模型。模型没有浏览网页的能力，
  脱离页面内容问它「还开着吗」只会得到幻觉。**别改成让模型直接判**。
- 参数：`--limit N`（试水）、`--rerate`（全量重评）、`--only-liveness`、`--no-liveness`。

---

## 本地文件（**不在仓库里**，被 .gitignore 排除）

| 文件 | 内容 | 没有会怎样 |
|---|---|---|
| `job_config.json` | API key（gemini / adzuna / rapidapi…） | 对应源 SKIP；Gemini 评级直接退出 |
| `cv_profile.json` | 候选人画像，**含身份状况** | `job_rate.py` 报错退出并提示 |
| `jobs.db` | 去重与状态追踪 | 「今日新增/已消失」失效 |
| `job_output/` | xlsx 与日志 | — |

模板：`job_config.example.json`、`cv_profile.example.json`。
**这三样都不许提交**——这是公开仓库，还开着 GitHub Pages。

Gemini 的 key 填在 `job_config.json` 的 `gemini_api_key`，模型是 `gemini-3.7-flash`。
注意：`AQ.` 开头的 key 走的是 Google Cloud 那套鉴权，打 generativelanguage 会回 401，
要 `AIza` 开头的（https://aistudio.google.com/apikey）。脚本碰到 401 会专门提示这一点。

---

## 踩过的坑（别重犯）

- **`.gitignore` 不支持行尾注释。** `job_config.json  # 含 key` 匹配不到任何文件——
  差点把 API key 提交进公开仓库。注释必须独占一行。改完用 `git check-ignore -v` 验证。
- **国别绝不能回落到来源站。** 曾经把摩洛哥的岗位标成美国。顺序是：地点字段 →
  正文里的地点线索 → Unknown。
- **`country: "BOTH"` 会静默丢源。** 13 个源这么写过，`guess_country` 返回 "BOTH"
  不在 {US,CA} 里，整源被丢掉（包括 ASABE）。现在用 `VALID_COUNTRIES` 归一化。
- **`token_set_ratio` 对子集给满分**，导航链接 "Research" 被判成 Research Assistant
  Professor。已改 `token_sort_ratio` + 长度护栏。
- **限速的 sleep 不能放在锁里**，否则 8 个线程退化成 1 个（一轮从 1 分钟变 7 分钟）。
- **HigherEdJobs 的 RSS 路径**是 `/rss/categoryFeed.cfm?catID=NN`，不是 `/search/rss.cfm`。
  Madgex 系招聘板是 `/jobsrss/?keywords=`，不是 `?format=rss`。
- **robots.txt 一律遵守。** 有 6 个源（Academic Positions ×3、ASA-CSSA-SSSA、
  ResearchGate、ASPRS）因此禁用了，让用户手动浏览，**不要绕**。
- **LinkedIn / Indeed / Glassdoor / ZipRecruiter 不直接爬**（ToS + 反爬）。
  合规路径是 RapidAPI 上的 JSearch（聚合这四家），连接器已写好，缺 key 自动 SKIP。

## 环境

用户是 Windows + Anaconda，依赖装在 **`goML`** 环境里（base 里没有 feedparser）。
本地跑要先 `conda activate goML`。`.bat` 里有解释器探测会自动找到装齐依赖的环境。
仓库在 `C:\Users\ft7b6\Desktop\JOBSCAN`。

分支：**`claude/job-search-vdowz1`**。不要往 main 推。

---

## 待办

- [ ] 用 `AIza` 开头的 key 实跑一次 `job_rate.py --limit 20`，确认评级质量
- [ ] 按 CV 拓宽方向：表型组学 / 精准畜牧 / 农业机器人 / 杂草科学 / 土壤传感 /
      地理空间，这几块 CV 都撑得住，但 `cv_profile.json` 的 target_roles 和
      `job_titles.json` 的职位类别还没完全覆盖（例如 "Precision Agriculture Specialist"、
      "Phenomics Scientist"、"Machine Learning Scientist" 目前匹配不上）
- [ ] 跨源去重（同一个岗位从 HigherEdJobs + Chronicle + 学校官网各来一条）
- [ ] xlsx 每天覆盖会冲掉用户自己标的"已投递"，需要持久化
- [ ] 截止日期解析成真正的日期字段
- [ ] LinkedIn/Indeed 邮件订阅 + IMAP 解析（提过方案，没做）
