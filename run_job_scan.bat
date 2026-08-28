@echo off
REM ── 学术岗位每日扫描 ──────────────────────────────────────────────
REM 直接双击运行，或由「任务计划程序」每天 08:00 触发。
REM 参数会原样透传，例如:  run_job_scan.bat --check-sources
REM
REM 用 %~dp0 定位脚本所在目录，所以整个文件夹可以随便搬。

chcp 65001 >nul
cd /d "%~dp0"

echo.
echo   学术岗位扫描  %DATE% %TIME%
echo.

python -X utf8 job_scan.py %*
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
