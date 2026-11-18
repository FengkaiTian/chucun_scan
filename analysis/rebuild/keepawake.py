"""穷举跑完之前不让 Windows 因空闲睡眠（只对本进程生效，不改系统设置；合盖仍会睡眠）。"""
import ctypes, time, os
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_all.log')
while True:
    time.sleep(60)
    if os.path.exists(log) and '5 重 结束' in open(log, encoding='utf-8', errors='ignore').read():
        break
