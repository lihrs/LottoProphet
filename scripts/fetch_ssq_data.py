# fetch_ssq_data.py

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import urllib3
import sys

# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_url(name):
    """
    构建数据爬取的URL
    :param name: 玩法名称 ('ssq')
    :return: (url, path)
    """
    url = f"https://datachart.500.com/{name}/history/"
    path = "newinc/history.php?start={}&end="
    return url, path

def get_current_number(name):
    """
    获取最新一期的期号
    :param name: 玩法名称 ('ssq')
    :return: current_number (字符串)
    """
    url, _ = get_url(name)
    full_url = f"{url}history.shtml"
    print(f"Fetching URL: {full_url}")
    try:
        response = requests.get(full_url, verify=False)
        if response.status_code != 200:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            sys.exit(1)
        response.encoding = "gb2312"
        soup = BeautifulSoup(response.text, "lxml")
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
        return current_num
    except Exception as e:
        print(f"Error fetching current number: {e}")
        sys.exit(1)

def spider(name, start, end):
    """
    爬取历史数据
    :param name: 玩法名称 ('ssq')
    :param start: 开始期数
    :param end: 结束期数
    :return: DataFrame
    """
    url, path = get_url(name)
    full_url = f"{url}{path.format(start)}{end}"
    print(f"Fetching URL: {full_url}")
    try:
        response = requests.get(full_url, verify=False)
        if response.status_code != 200:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            sys.exit(1)
        response.encoding = "gb2312"
        soup = BeautifulSoup(response.text, "lxml")
        trs = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
        data = []
        for tr in trs:
            item = {}
            try:
                tds = tr.find_all("td")
                item["期数"] = tds[0].get_text().strip()
                for i in range(6):
                    item[f"红球_{i+1}"] = tds[i+1].get_text().strip()
                item["蓝球"] = tds[7].get_text().strip()
                data.append(item)
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error in spider: {e}")
        sys.exit(1)

def fetch_ssq_data():
    """
    获取并保存双色球历史数据到 'ssq/ssq_history.csv'
    """
    name = "ssq"
    current_number = get_current_number(name)
    print(f"最新一期期号：{current_number}")
    df = spider(name, 1, current_number)
    save_path = os.path.join(name_path[name]['path'], data_file_name)
    os.makedirs(name_path[name]["path"], exist_ok=True)
    df.to_csv(save_path, encoding="utf-8", index=False)
    print(f"数据已保存至 {save_path}")

if __name__ == "__main__":
    # 配置
    name_path = {
        "ssq": {
            "name": "双色球",
            "path": "./ssq/"
        }
    }
    data_file_name = "ssq_history.csv"

    fetch_ssq_data()
