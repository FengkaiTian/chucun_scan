import yfinance as yf
import pandas as pd
import requests
import io
import sys

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200 or 'Please set a user-agent' in response.text:
       headers = {'User-Agent': 'Sp500Fetcher/1.0 (contact@example.com)'}
       response = requests.get(url, headers=headers)
    
    df = pd.read_html(io.StringIO(response.text))[0]
    tickers = df['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

def calc_td_seq(close_prices):
    b_count = 0
    s_count = 0
    buy_setup = [0] * len(close_prices)
    sell_setup = [0] * len(close_prices)
    
    for i in range(4, len(close_prices)):
        if close_prices.iloc[i] < close_prices.iloc[i-4]:
            b_count += 1
            s_count = 0
            buy_setup[i] = b_count
            if b_count == 9: 
                b_count = 0
        elif close_prices.iloc[i] > close_prices.iloc[i-4]:
            s_count += 1
            b_count = 0
            sell_setup[i] = s_count
            if s_count == 9: 
                s_count = 0
        else:
            b_count = 0
            s_count = 0
            
    return buy_setup, sell_setup

def main():
    print("正在获取标普500成分股列表...")
    tickers = get_sp500_tickers()
    print(f"共获取到 {len(tickers)} 只股票代码。")
    
    print("正在下载过去 3 年的日线数据 (这需要两三分钟)...")
    data = yf.download(tickers, period='3y', group_by='ticker', auto_adjust=True, threads=True)
    
    results = []
    
    for ticker in tickers:
        try:
            if ticker in data:
                df = data[ticker].dropna()
                if len(df) < 15:
                    continue
                
                prices_close = df['Close']
                prices_open = df['Open']
                prices_vol = df['Volume']
                
                # Calculate 5-day moving average of volume
                vol_ma5 = prices_vol.rolling(window=5).mean()
                
                b_s, s_s = calc_td_seq(prices_close)
                
                for i in range(len(prices_close)):
                    is_buy_9 = b_s[i] == 9
                    is_sell_9 = s_s[i] == 9
                    
                    if is_buy_9:
                        # Apply Volume filter: VOL(T0) > VOL_MA5(T0) * 1.2
                        if i < 4 or pd.isna(vol_ma5.iloc[i]):
                            continue
                            
                        if prices_vol.iloc[i] <= vol_ma5.iloc[i] * 1.2:
                            continue
                            
                        t0_close = float(prices_close.iloc[i])
                        t0_open = float(prices_open.iloc[i])
                        
                        def get_price(idx, prices_series):
                            return float(prices_series.iloc[idx]) if idx < len(prices_series) else None
                        
                        def day_pct(cur, prev):
                            if cur is None or prev is None or prev == 0:
                                return None
                            return round((cur - prev) / prev * 100, 2)
                        
                        closes = [t0_close]
                        for j in range(1, 6):
                            closes.append(get_price(i+j, prices_close))
                        
                        row = {
                            '日期 (T0)': prices_close.index[i].strftime('%Y-%m-%d'),
                            '股票代码': ticker,
                            '信号': 'Buy 9',
                            'T0 开盘价': round(t0_open, 2),
                            'T0 收盘价': round(t0_close, 2),
                            'T0 成交量': int(prices_vol.iloc[i]),
                            'T0 5日均量': int(vol_ma5.iloc[i]),
                        }
                        
                        for j in range(1, 6):
                            op = get_price(i+j, prices_open)
                            cp = closes[j]
                            prev_cp = closes[j-1]
                            row[f'T+{j} 开盘价'] = round(op, 2) if op is not None else None
                            row[f'T+{j} 开盘涨跌幅(%)'] = day_pct(op, prev_cp)
                            row[f'T+{j} 收盘价'] = round(cp, 2) if cp is not None else None
                            row[f'T+{j} 收盘涨跌幅(%)'] = day_pct(cp, prev_cp)
                            
                        results.append(row)
        except Exception as e:
            pass
            
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        results_df = results_df.sort_values(by=['日期 (T0)', '股票代码'], ascending=[False, True])
    
    output_file = r"c:\Users\ft7b6\OneDrive\Desktop\STOCK\magic_nine_3y_vol_history.csv"
    if not results_df.empty:
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    summary_data = []
    
    if not results_df.empty:
        for signal in ['Buy 9']:
            subset = results_df[results_df['信号'] == signal]
            count = len(subset)
            if count == 0:
                continue
                
            row_summary = {'信号': '底部买入(Buy 9)' if signal == 'Buy 9' else '顶部卖出(Sell 9)', '总次数': count}
            
            for j in range(1, 6):
                col_name = f'T+{j} 收盘涨跌幅(%)'
                valid_returns = subset[col_name].dropna()
                if len(valid_returns) == 0:
                    row_summary[f'T+{j} 胜率'] = "N/A"
                    row_summary[f'T+{j} 平均涨跌幅'] = "N/A"
                    continue
                
                if signal == 'Buy 9':
                    wins = valid_returns[valid_returns > 0].count()
                else:
                    wins = valid_returns[valid_returns < 0].count()
                    
                win_rate = wins / len(valid_returns) * 100
                avg_return = valid_returns.mean()
                
                row_summary[f'T+{j} 胜率'] = f"{win_rate:.2f}%"
                row_summary[f'T+{j} 平均涨跌幅'] = f"{avg_return:.2f}%"
                
            summary_data.append(row_summary)
            
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n========================================")
    print(f"数据采集完成！共收集到 {len(results)} 条符合量价条件 (VOL > MA(VOL,5)*1.2) 的九转信号记录。")
    if not results_df.empty:
        print(f"结果已保存到：{output_file}")
    print(f"========================================\n")
    
    print("【 加入放量条件 (VOL > MA120%) 的神奇九转胜率与平均涨跌幅统计 (近3年) 】\n")
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            print(f"信号类型: {row['信号']} (共计 {row['总次数']} 次)")
            print("-" * 30)
            for j in range(1, 6):
                print(f"T+{j} 胜率: {row.get(f'T+{j} 胜率', 'N/A')} | T+{j} 平均涨跌幅: {row.get(f'T+{j} 平均涨跌幅', 'N/A')}")
            print("\n")
    else:
        print("没有找到符合放量条件的九转信号。")
    
    summary_file = r"c:\Users\ft7b6\OneDrive\Desktop\STOCK\magic_nine_3y_vol_summary.csv"
    if not summary_df.empty:
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"数据统计摘要已保存至：{summary_file}")

if __name__ == '__main__':
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
