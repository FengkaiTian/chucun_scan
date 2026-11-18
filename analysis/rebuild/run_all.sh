#!/bin/bash
# 依次跑：4 重（打乱 10 次）→ 5 重（样本≥100，只存 t≥3，打乱 3 次）
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
echo "=== 4 重 开始 $(date)" >> run_all.log
python -u mega_search.py 4 10 60 2.0 65 0.01 >> run_all.log 2>&1
echo "=== 4 重 结束 $(date)" >> run_all.log
python -u mega_search.py 5 3 100 3.0 75 0.002 >> run_all.log 2>&1
echo "=== 5 重 结束 $(date)" >> run_all.log
