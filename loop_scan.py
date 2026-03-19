import subprocess, time, datetime

def wait_until_next_430():
    now = datetime.datetime.now()
    target = now.replace(hour=4, minute=30, second=0, microsecond=0)
    if now >= target:
        target += datetime.timedelta(days=1)
    wait_sec = (target - now).total_seconds()
    print(f'Next run at 04:30 - waiting {wait_sec/3600:.1f} hours...')
    time.sleep(wait_sec)

while True:
    wait_until_next_430()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f'[{now}] Running scan...')
    subprocess.run(['python', r'C:\Users\ft7b6\OneDrive\Desktop\STOCK\daily_scan.py'])
    print(f'[{now}] Done.')
