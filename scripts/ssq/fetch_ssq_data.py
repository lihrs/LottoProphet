import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import urllib3
import sys
import logging

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------- 配置 ----------------
name_path = {
    "ssq": {
        "name": "双色球",
        "path": "./"  
    }
}
data_file_name = "ssq_history.csv"

# 设置日志
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
    logging.info(f"Fetching URL: {full_url}")
    try:
        response = requests.get(full_url, verify=False, timeout=10)
        if response.status_code != 200:
            logging.warning(f"Failed to fetch data. Status code: {response.status_code}")
            sys.exit(1)
        response.encoding = "gb2312"
        soup = BeautifulSoup(response.text, "lxml")
        current_num_input = soup.find("input", id="end")
        if not current_num_input:
            logging.warning("Could not find the 'end' input element on the page.")
            sys.exit(1)
        current_num = current_num_input.get("value", "").strip()
        if not current_num:
            logging.warning("The 'end' input element does not have a 'value' attribute.")
            sys.exit(1)
        logging.info(f"最新一期期号：{current_num}")
        return current_num
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error fetching current number: {e}")
        sys.exit(1)
    except Exception as e:
        logging.warning(f"Unexpected error: {e}")
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
    logging.info(f"爬取URL: {full_url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 增加重试次数和User-Agent
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = requests.get(full_url, headers=headers, verify=False, timeout=30)
                if response.status_code == 200:
                    break
                logging.warning(f"重试第 {retry+1} 次，状态码: {response.status_code}")
                if retry == max_retries - 1:
                    logging.warning(f"无法获取数据。状态码: {response.status_code}")
                    sys.exit(1)
            except requests.exceptions.RequestException as e:
                logging.warning(f"重试第 {retry+1} 次，错误: {e}")
                if retry == max_retries - 1:
                    raise
        
        # 尝试不同的编码
        encodings = ["gb2312", "gbk", "utf-8"]
        for encoding in encodings:
            try:
                response.encoding = encoding
                response.text
                logging.info(f"成功使用 {encoding} 编码解析内容")
                break
            except UnicodeDecodeError:
                continue
        
        # 验证解码是否成功
        if not response.text:
            logging.warning("无法解码页面内容，请尝试其他编码方式。")
            sys.exit(1)
        
        soup = BeautifulSoup(response.text, "lxml")
        tbody = soup.find("tbody", attrs={"id": "tdata"})
        if not tbody:
            logging.warning("找不到ID为 'tdata' 的表格体。尝试查找其他表格...")
            
            # 尝试查找其他可能的表格
            tables = soup.find_all("table")
            if tables:
                logging.info(f"找到 {len(tables)} 个表格，尝试第一个...")
                tbody = tables[0].find("tbody")
            
            if not tbody:
                # 尝试查找任何包含数据的元素
                logging.warning("找不到任何表格体，尝试查找行...")
                trs = soup.find_all("tr")
                if not trs:
                    logging.warning("找不到任何数据行。")
                    sys.exit(1)
            else:
                trs = tbody.find_all("tr")
        else:
            trs = tbody.find_all("tr")
        
        logging.info(f"找到 {len(trs)} 行数据")
        data = []
        for tr in trs:
            item = {}
            try:
                tds = tr.find_all("td")
                if len(tds) < 9:  # 双色球需要至少9列(期数+6红球+1蓝球+其他信息)
                    logging.warning(f"跳过不完整行: {len(tds)} 列")
                    continue
                
                # 获取期数
                issue_num = tds[0].get_text().strip()
                if not issue_num:
                    logging.warning("跳过期数为空的行")
                    continue
                item["期数"] = issue_num
                
                # 获取红球
                for i in range(6):
                    red_ball = tds[i+1].get_text().strip()
                    try:
                        item[f"红球_{i+1}"] = int(red_ball) if red_ball.isdigit() else 0
                    except ValueError:
                        logging.warning(f"无法解析红球_{i+1}: {red_ball}")
                        item[f"红球_{i+1}"] = 0
                
                # 获取蓝球
                if 7 < len(tds):
                    blue_ball = tds[7].get_text().strip()
                    try:
                        item["蓝球"] = int(blue_ball) if blue_ball.isdigit() else 0
                    except ValueError:
                        logging.warning(f"无法解析蓝球: {blue_ball}")
                        item["蓝球"] = 0
                else:
                    logging.warning("蓝球不存在")
                    item["蓝球"] = 0
                
                # 验证球号是否在有效范围内
                valid_data = True
                for i in range(6):
                    if not (1 <= item[f"红球_{i+1}"] <= 33):
                        logging.warning(f"红球_{i+1} 值无效: {item[f'红球_{i+1}']}")
                        valid_data = False
                        break
                
                if not (1 <= item["蓝球"] <= 16):
                    logging.warning(f"蓝球值无效: {item['蓝球']}")
                    valid_data = False
                
                if valid_data:
                    data.append(item)
                else:
                    logging.warning(f"跳过无效数据行: {item}")
                
            except Exception as e:
                logging.warning(f"解析行错误: {e}")
                continue
        
        if not data:
            logging.warning("未找到有效数据!")
            sys.exit(1)
        
        df = pd.DataFrame(data)
        
        # 转换期数为数字并排序
        try:
            df['期数'] = pd.to_numeric(df['期数'], errors='coerce')
            df = df.dropna(subset=['期数']).sort_values(by='期数', ascending=False).reset_index(drop=True)
        except Exception as e:
            logging.warning(f"排序期数时出错: {e}")
            # 保持原始顺序
        
        logging.info(f"成功爬取 {len(df)} 条数据。")
        return df
    except requests.exceptions.RequestException as e:
        logging.warning(f"获取数据错误: {e}")
        sys.exit(1)
    except Exception as e:
        logging.warning(f"spider 函数错误: {e}")
        sys.exit(1)

def fetch_ssq_data():
    """
    获取并保存双色球历史数据到 'scripts/ssq/ssq_history.csv'
    """
    name = "ssq"
    current_number = get_current_number(name)
    df = spider(name, 1, current_number)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        df.to_csv(save_path, encoding="gbk", index=False)
        logging.info(f"数据已保存至 {save_path}")
    except Exception as e:
        logging.warning(f"保存数据到CSV时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_ssq_data()
