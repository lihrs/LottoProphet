

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import sys
import logging
import urllib3

# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_url(name):
    """
    构建数据爬取的URL
    :param name: 玩法名称 ('dlt')
    :return: (url, path)
    """
    url = f"https://datachart.500.com/{name}/history/"
    path = "newinc/history.php?start={}&end="
    return url, path

def get_current_number(name):
    """
    获取最新一期的期号
    :param name: 玩法名称 ('dlt')
    :return: current_number (字符串)
    """
    url, _ = get_url(name)
    full_url = f"{url}history.shtml"
    logging.info(f"Fetching URL: {full_url}")
    try:
        response = requests.get(full_url, verify=False, timeout=10)
        if response.status_code != 200:
            logging.error(f"Failed to fetch data. Status code: {response.status_code}")
            sys.exit(1)
        response.encoding = "gb2312"
        soup = BeautifulSoup(response.text, "lxml")
        # 根据实际网页结构调整
        current_num_input = soup.find("input", id="end")
        if not current_num_input:
            logging.error("Could not find the 'end' input element on the page.")
            sys.exit(1)
        current_num = current_num_input.get("value", "").strip()
        if not current_num:
            logging.error("The 'end' input element does not have a 'value' attribute.")
            sys.exit(1)
        logging.info(f"最新一期期号：{current_num}")
        return current_num
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching current number: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

def spider(name, start, end):
    """
    爬取历史数据
    :param name: 玩法名称 ('dlt')
    :param start: 开始期数
    :param end: 结束期数
    :return: DataFrame
    """
    url, path = get_url(name)
    full_url = f"{url}{path.format(start)}{end}"
    logging.info(f"Fetching URL: {full_url}")
    try:
        response = requests.get(full_url, verify=False, timeout=10)
        if response.status_code != 200:
            logging.error(f"Failed to fetch data. Status code: {response.status_code}")
            sys.exit(1)
        response.encoding = "gb2312"
        soup = BeautifulSoup(response.text, "lxml")
        tbody = soup.find("tbody", attrs={"id": "tdata"})
        if not tbody:
            logging.error("Could not find the table body with id 'tdata'.")
            sys.exit(1)
        trs = tbody.find_all("tr")
        data = []
        for tr in trs:
            item = {}
            try:
                tds = tr.find_all("td")
                if len(tds) < 8:
                    logging.warning(f"Skipping incomplete row: {tr}")
                    continue
                item["期数"] = tds[0].get_text().strip()
                for i in range(5):
                    red_ball = tds[i+1].get_text().strip()
                    item[f"红球_{i+1}"] = int(red_ball) if red_ball.isdigit() else 0
                for i in range(2):
                    blue_ball = tds[6+i].get_text().strip()
                    item[f"蓝球_{i+1}"] = int(blue_ball) if blue_ball.isdigit() else 0
                data.append(item)
            except Exception as e:
                logging.warning(f"Error parsing row: {e}")
                continue
        df = pd.DataFrame(data)
        # 排序期数
        df['期数'] = pd.to_numeric(df['期数'], errors='coerce')
        df = df.dropna(subset=['期数']).sort_values(by='期数').reset_index(drop=True)
        logging.info(f"成功爬取 {len(df)} 条数据。")
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error in spider: {e}")
        sys.exit(1)

def fetch_dlt_data():
    """
    获取并保存大乐透历史数据到 'scripts/dlt/dlt_history.csv'
    """
    name = "dlt"
    current_number = get_current_number(name)
    df = spider(name, 1, current_number)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlt_history.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        df.to_csv(save_path, encoding="utf-8", index=False)
        logging.info(f"数据已保存至 {save_path}")
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    fetch_dlt_data()
