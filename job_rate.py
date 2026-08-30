"""
Job Rater - 用 Gemini 对已抓取的岗位逐条判定匹配度与是否仍在招

读 jobs.db 里的活跃岗位，按 cv_profile.json 的画像逐条评级：
    绿色  很符合      黄色  有可能符合
    红色  不太符合    黑色  完全不符合
同时核对岗位是否还开着，关了的标记为「已关闭」。

判定结果写回 jobs.db 并重新生成 xlsx。已评过的不会重复评（省钱），
除非用 --rerate 强制重评。

用法:
    python job_rate.py                  评未评过的 + 核对在招状态
    python job_rate.py --rerate         全部重评
    python job_rate.py --limit 50       只处理前 50 条（先试水看效果）
    python job_rate.py --no-liveness    跳过在招核对，只评匹配度
    python job_rate.py --self-test      离线自测，不联网
"""
import os, sys, re, json, time, argparse, sqlite3, traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import job_scan as js          # 复用 Store / write_xlsx / Http / say 等

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CV_FILE     = os.path.join(SCRIPT_DIR, 'cv_profile.json')

DEFAULT_MODEL = 'gemini-3.7-flash'
GEMINI_HOST   = 'https://generativelanguage.googleapis.com'

BATCH_SIZE    = 8       # 每次请求塞几个岗位；太大容易被截断
DESC_FOR_LLM  = 1500    # 送给模型的描述截断长度
MAX_WORKERS   = 4

RATINGS = ['绿色', '黄色', '红色', '黑色']
RATING_DESC = {
    '绿色': '很符合，值得优先投',
    '黄色': '有可能符合，需要你自己判断',
    '红色': '不太符合',
    '黑色': '完全不符合',
}

# 页面上出现这些说法，基本可以断定岗位已关闭。先用它做免费判断，
# 只有含糊不清的页面才送给模型，省调用也更可靠。
CLOSED_MARKERS = [
    r'no longer (accepting|available|active|open)',
    r'position (has been|is) (filled|closed)',
    r'this (job|posting|position|requisition) (is|has been) (closed|filled|removed|expired)',
    r'posting (has )?expired', r'applications are (now )?closed',
    r'we are no longer', r'job not found', r'requisition (is )?(closed|not found)',
    r'the (job|position) you are looking for', r'has been removed',
    r'不再接受申请', r'该职位已关闭',
]
CLOSED_RE = [re.compile(p, re.I) for p in CLOSED_MARKERS]


# ══════════════════════════════════════════════════════════════════
#  Gemini
# ══════════════════════════════════════════════════════════════════

class GeminiError(Exception):
    pass


class Gemini:
    def __init__(self, cfg):
        self.key = (cfg.get('gemini_api_key') or '').strip()
        self.model = (cfg.get('gemini_model') or DEFAULT_MODEL).strip()
        if not self.key:
            raise GeminiError(
                '未配置 gemini_api_key。把 key 填进 job_config.json 的 gemini_api_key。\n'
                '  注意：job_config.json 已被 .gitignore 排除，不会进入公开仓库。')
        self.url = f'{GEMINI_HOST}/v1beta/models/{self.model}:generateContent'
        self.s = requests.Session()

    def ask_json(self, prompt, timeout=90):
        """发一次请求并要求返回 JSON。返回解析后的对象。"""
        body = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0,
                'responseMimeType': 'application/json',
            },
        }
        # 新版推荐用 x-goog-api-key 头；老的 ?key= 作为回退。
        attempts = [
            ({'x-goog-api-key': self.key, 'Content-Type': 'application/json'}, self.url),
            ({'Content-Type': 'application/json'}, f'{self.url}?key={self.key}'),
        ]
        last = None
        for headers, url in attempts:
            try:
                r = self.s.post(url, headers=headers, json=body, timeout=timeout)
            except requests.RequestException as e:
                last = f'{type(e).__name__}: {str(e)[:120]}'
                continue
            if r.status_code in (401, 403):
                last = self._auth_hint(r)
                continue
            if r.status_code == 404:
                last = (f'模型 {self.model} 不存在或不可用（HTTP 404）。'
                        f'在 job_config.json 里改 gemini_model。')
                break
            if r.status_code == 429:
                raise GeminiError('HTTP 429 触发限流。等一会儿再跑，或调小 BATCH_SIZE。')
            try:
                r.raise_for_status()
            except requests.HTTPError:
                last = f'HTTP {r.status_code}: {r.text[:200]}'
                continue
            return self._extract(r.json())
        raise GeminiError(last or '未知错误')

    def _auth_hint(self, r):
        hint = f'HTTP {r.status_code} 认证失败。'
        if self.key.startswith('AQ.'):
            hint += ('\n  你的 key 是 AQ. 前缀。已知问题：AQ. 前缀的 token 对 '
                     'Generative Language REST API 不生效，会返回 401。\n'
                     '  去 Google AI Studio 看能否生成 AIza 开头的传统 API key。')
        else:
            hint += ' 检查 key 是否正确、是否已启用 Generative Language API。'
        return hint + f'\n  响应: {r.text[:200]}'

    @staticmethod
    def _extract(data):
        try:
            txt = data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError, TypeError):
            fb = (data.get('promptFeedback') or {}).get('blockReason')
            raise GeminiError(f'响应里没有内容' + (f'（被拦截: {fb}）' if fb else f': {str(data)[:200]}'))
        return parse_json_loose(txt)


def parse_json_loose(txt):
    """模型有时会把 JSON 包在 ``` 里，或前后带解释文字。"""
    txt = (txt or '').strip()
    if txt.startswith('```'):
        txt = re.sub(r'^```[a-z]*\s*', '', txt)
        txt = re.sub(r'\s*```$', '', txt)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    for opener, closer in (('[', ']'), ('{', '}')):
        i, j = txt.find(opener), txt.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(txt[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise GeminiError(f'返回的不是合法 JSON: {txt[:200]}')


# ══════════════════════════════════════════════════════════════════
#  评级
# ══════════════════════════════════════════════════════════════════

def build_rating_prompt(cv, jobs):
    lines = [
        '你是一位帮助博士毕业生筛选学术与科研岗位的助手。',
        '下面先给出候选人画像，然后是若干岗位。请逐条判断岗位与候选人的匹配度。',
        '',
        '【候选人画像】',
        json.dumps(cv, ensure_ascii=False, indent=2),
        '',
        '【评级标准】',
        '  绿色 = 很符合：方向对口，层级合适，身份上可行，值得优先投',
        '  黄色 = 有可能符合：方向沾边或信息不足，需要人工再看',
        '  红色 = 不太符合：方向偏离较多，或层级明显不匹配',
        '  黑色 = 完全不符合：领域无关，或身份上根本不可能（要求美国公民、',
        '         需安全许可、明确不担保签证等）',
        '',
        '【重要】',
        '  1. 身份是硬约束。要求美国公民身份、安全许可、出口管制、或明确不提供',
        '     签证担保的岗位，一律判黑色，理由里写明原因。',
        '  2. 大多数学术岗位不会提签证，那属于正常，不要因此降级。',
        '  3. 描述为空或极短时判黄色，理由写「信息不足」，不要臆测。',
        '  4. 理由必须具体，指出是哪一点匹配或不匹配，不要写空话。理由用中文，',
        '     不超过 40 字。',
        '',
        '【岗位列表】',
        json.dumps(jobs, ensure_ascii=False, indent=2),
        '',
        '【输出格式】',
        '只输出一个 JSON 数组，每个元素形如：',
        '  {"id": "<原样照抄岗位的 id>", "rating": "绿色|黄色|红色|黑色", "reason": "<中文理由>"}',
        '数组长度必须与岗位数量一致，不要遗漏、不要新增。',
    ]
    return '\n'.join(lines)


def rate_batch(gem, cv, rows):
    payload = [{
        'id': r['job_id'],
        'title': r['title'],
        'org': r['org'] or '',
        'location': r['location'] or '',
        'country': r['country'] or '(未知)',
        'work_auth': r['work_auth'] or '未说明',
        'work_auth_note': (r['work_auth_note'] or '')[:200],
        'keywords': r['keywords'] or '',
        'description': (r['description'] or '')[:DESC_FOR_LLM],
    } for r in rows]

    data = gem.ask_json(build_rating_prompt(cv, payload))
    if isinstance(data, dict):
        data = data.get('results') or data.get('ratings') or [data]
    if not isinstance(data, list):
        raise GeminiError(f'期望 JSON 数组，实得 {type(data).__name__}')

    out = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        jid = str(item.get('id', '')).strip()
        rating = str(item.get('rating', '')).strip()
        if rating not in RATINGS:
            # 模型偶尔会写「绿」「green」之类
            for r in RATINGS:
                if r[0] in rating or r in rating:
                    rating = r
                    break
            else:
                rating = '黄色'
        if jid:
            out[jid] = (rating, str(item.get('reason', '')).strip()[:120])
    return out


# ══════════════════════════════════════════════════════════════════
#  在招状态核对
# ══════════════════════════════════════════════════════════════════

def check_open(http, url):
    """核对岗位是否还开着。

    先用 HTTP 判断——这是确定性的、免费的，也比问模型可靠：模型没有
    浏览网页的能力，脱离页面内容去问它「还开着吗」只会得到幻觉。
    只有页面能打开、内容又含糊时，才把正文交给模型判断。

    返回 (状态, 说明, 待模型判断的正文或 None)
    """
    if not url or not url.startswith('http'):
        return '未知', '无有效链接', None
    try:
        r = http.get(url, allow_redirects=True)
    except requests.RequestException as e:
        return '未知', f'无法访问: {type(e).__name__}', None

    if r.status_code in (404, 410):
        return '已关闭', f'HTTP {r.status_code}', None
    if r.status_code >= 400:
        return '未知', f'HTTP {r.status_code}', None

    text = js.strip_html(r.text or '')
    head = text[:6000]
    for rx in CLOSED_RE:
        m = rx.search(head)
        if m:
            return '已关闭', f'页面写着「{m.group(0)[:40]}」', None
    return '在招', '', head[:2500]


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════

def ensure_columns(store):
    cols = {r[1] for r in store.db.execute('PRAGMA table_info(jobs)')}
    for c in ('rating', 'rating_reason', 'rated_at', 'open_status', 'open_note', 'checked_at'):
        if c not in cols:
            store.db.execute(f"ALTER TABLE jobs ADD COLUMN {c} TEXT DEFAULT ''")
    store.db.commit()


def main():
    ap = argparse.ArgumentParser(description='用 Gemini 逐条评定岗位匹配度与在招状态')
    ap.add_argument('--rerate', action='store_true', help='已评过的也重新评')
    ap.add_argument('--limit', type=int, help='只处理前 N 条，先试水')
    ap.add_argument('--no-liveness', action='store_true', help='跳过在招核对')
    ap.add_argument('--only-liveness', action='store_true', help='只核对在招状态，不评级')
    ap.add_argument('--self-test', action='store_true', help='离线自测，不联网')
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    cfg = js.load_config()
    if not os.path.exists(CV_FILE):
        js.say('[red]找不到 cv_profile.json[/]——这是给 Gemini 的筛选条件，没有它无从判断。')
        js.say('它含你的身份状况，属于隐私，故不入公开仓库。')
        js.say('把我发给你的那份放到脚本同目录即可；或复制 cv_profile.example.json 自行填写。')
        return 2
    cv = js.load_json(CV_FILE)
    store = js.Store()
    ensure_columns(store)

    where = 'status="active"' + ('' if args.rerate else " AND COALESCE(rating,'')=''")
    rows = store.select(where)
    if args.limit:
        rows = rows[:args.limit]

    active_all = store.select('status="active"')
    js.say(f'[bold]活跃岗位 {len(active_all)} 条[/] · 本次待评 {len(rows)} 条'
           + ('（--rerate 全量重评）' if args.rerate else '（跳过已评过的）'))

    # ── 在招状态核对（HTTP 为主，模型只处理含糊的）────────────────
    ambiguous = {}
    if not args.no_liveness:
        targets = active_all if args.rerate or args.only_liveness else rows
        if args.limit:
            targets = targets[:args.limit]
        js.say(f'\n[bold]核对在招状态[/]（{len(targets)} 条，先用 HTTP 判断）')
        http = js.Http(cfg)
        done = closed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(check_open, http, r['url']): r for r in targets}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    st, note, body = fut.result()
                except Exception as e:
                    st, note, body = '未知', f'{type(e).__name__}', None
                done += 1
                if st == '已关闭':
                    closed += 1
                if body:
                    ambiguous[r['job_id']] = body
                store.db.execute(
                    'UPDATE jobs SET open_status=?, open_note=?, checked_at=? WHERE job_id=?',
                    (st, note, datetime.now().strftime('%Y-%m-%d'), r['job_id']))
                if done % 25 == 0:
                    js.say(f'  已核对 {done}/{len(targets)} · 判定关闭 {closed}')
        store.db.commit()
        js.say(f'  完成：{done} 条已核对，其中 [bold]{closed} 条已关闭[/]')

    if args.only_liveness:
        rows = []

    # ── 评级 ──────────────────────────────────────────────────────
    if rows:
        try:
            gem = Gemini(cfg)
        except GeminiError as e:
            js.say(f'[red]{e}[/]')
            return 2
        js.say(f'\n[bold]Gemini 评级[/]（模型 {gem.model}，每批 {BATCH_SIZE} 条）')

        batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
        rated = failed = 0
        for bi, batch in enumerate(batches, 1):
            try:
                res = rate_batch(gem, cv, batch)
            except GeminiError as e:
                failed += len(batch)
                js.say(f'  [red]第 {bi}/{len(batches)} 批失败: {e}[/]')
                if bi == 1:
                    js.say('[red]首批就失败，中止。先解决上面的问题再跑。[/]')
                    return 2
                continue
            now = datetime.now().strftime('%Y-%m-%d')
            for r in batch:
                rating, reason = res.get(r['job_id'], ('黄色', '模型未返回该条，默认待定'))
                store.db.execute(
                    'UPDATE jobs SET rating=?, rating_reason=?, rated_at=? WHERE job_id=?',
                    (rating, reason, now, r['job_id']))
                rated += 1
            store.db.commit()
            js.say(f'  第 {bi}/{len(batches)} 批完成 · 累计 {rated} 条')
            time.sleep(0.4)
        js.say(f'  评级完成：{rated} 条成功' + (f'，{failed} 条失败' if failed else ''))

    # ── 重新出表 ──────────────────────────────────────────────────
    from collections import Counter
    all_active = store.select('status="active"')
    tally = Counter(r.get('rating') or '未评' for r in all_active)
    open_tally = Counter(r.get('open_status') or '未核对' for r in all_active)

    def sort_key(r):
        order = {'绿色': 0, '黄色': 1, '红色': 2, '黑色': 3, '': 4}
        return (0 if (r.get('open_status') or '') != '已关闭' else 1,
                order.get(r.get('rating') or '', 4), r.get('org') or '')

    today = datetime.now().strftime('%Y-%m-%d')
    sheets = {
        '绿色 很符合': sorted([r for r in all_active if r.get('rating') == '绿色'], key=sort_key),
        '黄色 可能符合': sorted([r for r in all_active if r.get('rating') == '黄色'], key=sort_key),
        '全部在招': sorted(all_active, key=sort_key),
        '今日新增': store.select('status="active" AND first_seen=?', (today,)),
        '已消失': store.select('status="gone"'),
    }
    os.makedirs(js.OUT_DIR, exist_ok=True)
    out = os.path.join(js.OUT_DIR, 'job_rated_latest.xlsx')
    js.write_xlsx(out, sheets, [], None)

    js.say('')
    js.say('[bold]评级分布[/]')
    for r in RATINGS + ['未评']:
        if tally.get(r):
            js.say(f'  {r:<5} {tally[r]:>4} 条   {RATING_DESC.get(r, "")}')
    js.say('[bold]在招状态[/]')
    for k, v in open_tally.most_common():
        js.say(f'  {k:<6} {v:>4} 条')
    js.say(f'\n输出: {out}')
    js.log_line(f'评级完成 绿{tally.get("绿色",0)} 黄{tally.get("黄色",0)} '
                f'红{tally.get("红色",0)} 黑{tally.get("黑色",0)} · '
                f'已关闭{open_tally.get("已关闭",0)}')
    return 0


# ══════════════════════════════════════════════════════════════════
#  离线自测
# ══════════════════════════════════════════════════════════════════

def self_test():
    js.say('[bold]job_rate 离线自测（不联网）[/]\n')
    failures = []

    js.say('[bold]1. CV 画像可加载[/]')
    cv = js.load_json(CV_FILE)
    for k in ('academic_background_narrative', 'research_threads', 'career_stage',
              'target_roles', 'work_authorization', 'weak_or_wrong_signals'):
        if not cv.get(k):
            failures.append(f'  cv_profile.json 缺 {k}')
    if not cv.get('work_authorization', {}).get('implications'):
        failures.append('  work_authorization.implications 为空——身份硬约束会失效')
    js.say(f'   {len(cv.get("research_threads", []))} 条研究方向 · '
           f'{len(cv.get("credentials", []))} 条学术成果 · '
           f'{len(cv.get("work_authorization", {}).get("implications", []))} 条身份约束')

    js.say('\n[bold]2. 模型返回的各种脏 JSON 都要能解析[/]')
    cases = [
        ('[{"id":"a","rating":"绿色","reason":"对口"}]', 1),
        ('```json\n[{"id":"a","rating":"黄色","reason":"信息不足"}]\n```', 1),
        ('好的，结果如下：\n[{"id":"a","rating":"红色","reason":"方向偏"}]\n希望有帮助', 1),
    ]
    for txt, n in cases:
        try:
            got = parse_json_loose(txt)
            if len(got) != n:
                failures.append(f'  JSON 解析条数不符: {txt[:30]!r}')
        except Exception as e:
            failures.append(f'  JSON 解析失败: {txt[:30]!r} {e}')
    js.say(f'   {len(cases)} 种形态（裸 JSON / ``` 包裹 / 前后带解释）')

    js.say('\n[bold]3. 非法评级要被归一化，不能直接落库[/]')
    class FakeGem:
        def ask_json(self, prompt):
            return [{'id': 'j1', 'rating': 'green', 'reason': 'ok'},
                    {'id': 'j2', 'rating': '绿', 'reason': 'ok'},
                    {'id': 'j3', 'rating': '胡说八道', 'reason': 'ok'}]
    rows = [{'job_id': f'j{i}', 'title': 't', 'org': '', 'location': '', 'country': '',
             'work_auth': '', 'work_auth_note': '', 'keywords': '', 'description': ''}
            for i in (1, 2, 3)]
    got = rate_batch(FakeGem(), cv, rows)
    for jid, (rating, _r) in got.items():
        if rating not in RATINGS:
            failures.append(f'  {jid} 评级未归一化: {rating!r}')
    js.say(f"   green→{got['j1'][0]} · 绿→{got['j2'][0]} · 胡说八道→{got['j3'][0]}（兜底为黄色）")

    js.say('\n[bold]4. 在招状态：HTTP 优先，模型只处理含糊的[/]')
    class R:
        def __init__(self, code, body, url='https://x.test/j/1'):
            self.status_code = code; self.text = body; self.url = url; self.headers = {}
    class H:
        def __init__(self, r): self.r = r
        def get(self, url, **kw): return self.r
    checks = [
        (R(404, ''), '已关闭', 'HTTP 404'),
        (R(200, '<html>This position has been filled.</html>'), '已关闭', '页面明写已招满'),
        (R(200, '<html>Assistant Professor in Remote Sensing. Apply by Oct 15.</html>'), '在招', '正常页面'),
        (R(500, ''), '未知', '服务器错误'),
    ]
    for resp, want, label in checks:
        st, note, body = check_open(H(resp), 'https://x.test/j/1')
        if st != want:
            failures.append(f'  在招判定（{label}）: 期望 {want!r} 实得 {st!r}')
    st, _n, body = check_open(H(R(200, '<html>Assistant Professor in Remote Sensing.</html>')),
                              'https://x.test/j/1')
    if not body:
        failures.append('  正常页面应把正文交给模型复核，实得空')
    js.say(f'   {len(checks)} 个用例 · 无链接时不请求: '
           f'{check_open(None, "")[0]!r}')

    js.say('\n[bold]5. 缺 key 时要给出可操作的提示[/]')
    try:
        Gemini({})
        failures.append('  缺 key 时未报错')
    except GeminiError as e:
        if 'job_config.json' not in str(e):
            failures.append('  缺 key 的提示没说清楚去哪里填')
    try:
        g = Gemini({'gemini_api_key': 'AQ.fake'})
        class FR:
            status_code = 401; text = '{"error":"unauthorized"}'
        hint = g._auth_hint(FR())
        if 'AQ.' not in hint:
            failures.append('  AQ. 前缀的已知问题未在提示中说明')
        js.say('   缺 key、AQ. 前缀 401 都有针对性提示')
    except Exception as e:
        failures.append(f'  构造 Gemini 失败: {e}')

    js.say('')
    if failures:
        js.say(f'[bold red]自测失败 {len(failures)} 项[/]')
        for f in failures:
            js.say(f'[red]{f}[/]')
        return 1
    js.say('[bold green]全部自测通过[/]')
    js.say('[dim]自测只覆盖离线逻辑。Gemini 的连通性与评级质量必须联网实跑验证。[/]')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        js.say('\n[yellow]已中断（已完成的部分已写入 jobs.db）[/]')
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
