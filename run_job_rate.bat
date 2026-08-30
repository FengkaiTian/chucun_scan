@echo off
REM ── Gemini 岗位评级 ───────────────────────────────────────────────
REM 先跑 run_job_scan.bat 抓取，再跑这个给每条岗位判绿/黄/红/黑，
REM 并核对岗位是否还开着。需要 job_config.json 里填好 gemini_api_key。
REM 参数会原样透传，例如:  run_job_rate.bat --limit 20

chcp 65001 >nul
cd /d "%~dp0"

REM ── 定位 Python 解释器 ────────────────────────────────────────────
REM 这里必须显式找解释器，不能裸写 python。原因：
REM   Anaconda 只在「Anaconda Prompt」里才进 PATH，而任务计划程序启动的是
REM   干净的 cmd。那里的 python 会命中微软商店的空壳程序（WindowsApps\
REM   python.exe），它静默退出、什么也不做、也不报错 —— 任务看起来"成功"了，
REM   实际什么都没跑。
REM
REM 如果下面的自动探测没找对，就在这一行手工填死路径
REM （在 Anaconda Prompt 里执行 where python，把第一行结果贴进来）：
set "PY_OVERRIDE="

set "PY="
if defined PY_OVERRIDE if exist "%PY_OVERRIDE%" set "PY=%PY_OVERRIDE%"
if not defined PY if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\miniconda3\python.exe" set "PY=%LOCALAPPDATA%\miniconda3\python.exe"
if not defined PY if exist "C:\ProgramData\anaconda3\python.exe" set "PY=C:\ProgramData\anaconda3\python.exe"
if not defined PY if exist "C:\ProgramData\Anaconda3\python.exe" set "PY=C:\ProgramData\Anaconda3\python.exe"

if not defined PY (
    echo.
    echo   [!] 找不到 Python 解释器。
    echo.
    echo   请打开「Anaconda Prompt」执行:  where python
    echo   把输出的第一行路径，填到本文件的 PY_OVERRIDE= 那一行。
    echo.
    pause
    exit /b 1
)

REM ── 验证这个解释器真的装了依赖 ────────────────────────────────────
REM 关键：conda 的 base 环境通常没装 feedparser，而依赖多半是在某个子环境
REM （如 goML）里 pip install 的。任务计划程序不带 CONDA_PREFIX，上面的探测
REM 会选中 base —— 于是 18 个 RSS 源全部 SKIP，任务"成功"却几乎没抓到东西。
REM 所以这里实际 import 一次；不通就去各 conda 子环境里找一个装齐的。
"%PY%" -c "import feedparser, openpyxl" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [i] %PY% 缺少依赖，正在其他 conda 环境中查找...
    for /d %%E in ("%USERPROFILE%\anaconda3\envs\*" "%USERPROFILE%\miniconda3\envs\*" "%LOCALAPPDATA%\anaconda3\envs\*") do (
        if exist "%%E\python.exe" (
            "%%E\python.exe" -c "import feedparser, openpyxl" >nul 2>&1 && set "PY=%%E\python.exe"
        )
    )
    "%PY%" -c "import feedparser, openpyxl" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo   [!] 所有环境都缺依赖。请在装有依赖的 Anaconda Prompt 里执行:
        echo         where python
        echo       把路径填到本文件的 PY_OVERRIDE= 那一行。
        echo.
        pause
        exit /b 1
    )
)

echo.
echo   Gemini 岗位评级  %DATE% %TIME%
echo   解释器: %PY%
echo.

"%PY%" -X utf8 job_rate.py %*
set RC=%ERRORLEVEL%

echo.
if %RC% NEQ 0 (
    echo   [!] 脚本以退出码 %RC% 结束，上面有错误信息。
) else (
    echo   完成。结果见 job_output\job_rated_latest.xlsx
)
echo.
pause
