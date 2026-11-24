@echo off
rem 每个交易日收盘后由 Windows 任务计划程序运行：前向跟踪 + 扫描 + 推送网页。日志写到 scan_log.txt
chcp 65001 >nul
cd /d "C:\Users\ft7b6\Desktop\美股\chucun_scan"
echo ==== %date% %time% ==== >> scan_log.txt
python -X utf8 forward_track.py >> scan_log.txt 2>&1
python -X utf8 daily_scan.py --no-open >> scan_log.txt 2>&1
git add forward_log.csv
git diff --cached --quiet || git commit -qm "forward track: %date%" >> scan_log.txt 2>&1
git push -q origin main >> scan_log.txt 2>&1
