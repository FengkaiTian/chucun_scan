@echo off
REM ── 学术岗位每日扫描 ──────────────────────────────────────────────
REM 直接双击运行，或由「任务计划程序」每天 08:00 触发。
REM 参数会原样透传，例如:  run_job_scan.bat --check-sources

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

echo.
echo   学术岗位扫描  %DATE% %TIME%
echo   解释器: %PY%
echo.

"%PY%" -X utf8 job_scan.py %*
set RC=%ERRORLEVEL%

echo.
if %RC% NEQ 0 (
    echo   [!] 脚本以退出码 %RC% 结束，上面有错误信息。
) else (
    echo   完成。结果见 job_output\ 目录，运行日志见 job_output\job_scan_log.txt
)
echo.

REM 保持窗口打开，让你回来还能看到状态表；6 小时后自动关闭，
REM 避免每天一个窗口越堆越多。按任意键可立即关闭。
REM 注意：任务计划程序有时会重定向 stdin，此时 timeout 会直接报
REM "Input redirection is not supported" 并退出，窗口就白闪一下没了。
REM 所以失败时回落到 pause。
timeout /t 21600
if %ERRORLEVEL% NEQ 0 pause
