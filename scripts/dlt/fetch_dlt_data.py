
# fetch_dlt_data.py

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

def fetch_dlt_data():
    """
    获取大乐透历史数据并保存到 CSV 文件中
    """
    url = "https://datachart.500.com/dlt/history/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 解析数据，这需要根据实际网页结构调整
    data = []
    table = soup.find('table', {'id': 'history_table'})  # 假设数据在ID为history_table的表格中
    for row in table.find_all('tr')[1:]:  # 跳过表头
        cols = row.find_all('td')
        if len(cols) >= 8:  # 假设每行有至少8列
            issue = cols[0].text.strip()
            red_balls = [int(ball) for ball in cols[1:6]]
            blue_balls = [int(ball) for ball in cols[6:8]]
            data.append([issue] + red_balls + blue_balls)

    df = pd.DataFrame(data, columns=['Issue'] + [f'Red_{i+1}' for i in range(5)] + [f'Blue_{i+1}' for i in range(2)])

    # 保存到 CSV
    save_path = "./dlt_history.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"大乐透历史数据已保存到 {save_path}")

if __name__ == "__main__":
    fetch_dlt_data()
