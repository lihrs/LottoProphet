import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONIOENCODING'] = 'utf-8'

def get_url():
    url = "https://datachart.500.com/dlt/history/"
    path = "newinc/history.php?start={}&end="
    return url, path

def fetch_dlt_data():
    url, path = get_url()
    full_url = f"{url}newinc/history.php?start=1&end=99999"
    print(f"Fetching URL: {full_url}")
    response = requests.get(full_url, verify=False)
    response.encoding = "gb2312"
    soup = BeautifulSoup(response.text, "lxml")
    trs = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
    data = []
    for tr in trs:
        try:
            tds = tr.find_all("td")
            row = {
                "期数": tds[0].get_text().strip(),
                "红球_1": tds[1].get_text().strip(),
                "红球_2": tds[2].get_text().strip(),
                "红球_3": tds[3].get_text().strip(),
                "红球_4": tds[4].get_text().strip(),
                "红球_5": tds[5].get_text().strip(),
                "蓝球_1": tds[6].get_text().strip(),
                "蓝球_2": tds[7].get_text().strip()
            }
            data.append(row)
        except Exception as e:
            print(f"Error parsing row: {e}")
    df = pd.DataFrame(data)
    os.makedirs("./dlt", exist_ok=True)
    file_path = "./dlt/dlt_history.csv"
    df.to_csv(file_path, encoding="utf-8", index=False)
    print(f"数据保存至 {file_path}")

if __name__ == "__main__":
    fetch_dlt_data()
