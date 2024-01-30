#This one simply download the mp3 file base on the URL in the CSV file.

import csv
import os
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from bs4 import BeautifulSoup

# 基本配置
csv_file_path = r"1st.csv"
save_directory = r"mp3"  # 替换成你的保存路径
log_file = r"downloaded_log.txt"  # 已下载的日志文件

# 并发和速度限制配置
workers = 10  # 并发数
max_speed = 99999999999999999999999999999999999999999*1024  # 速度限制，这里是10000KB/s

# 网速显示功能
def display_speed(downloaded_bytes, elapsed_time):
    if elapsed_time > 0:
        speed = downloaded_bytes / elapsed_time
        units = "B/s"
        if speed > 1024:  # 如果速度超过1KB，转换为KB/s
            speed /= 1024
            units = "KB/s"
        if speed > 1024:  # 如果速度超过1MB，转换为MB/s
            speed /= 1024
            units = "MB/s"
        return f"{speed:.2f} {units}"
    else:
        return "Calculating..."


# 速度限制器类
class SpeedLimiter:
    def __init__(self, max_speed):
        self.max_speed = max_speed  # in bytes per second
        self.last_check = time.time()

    def wait(self, bytes):
        now = time.time()
        elapsed = max(now - self.last_check, 0.1)  # Avoid division by zero, assume at least 0.1 second has passed
        if bytes/elapsed > self.max_speed:
            sleep_time = bytes/self.max_speed - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.last_check = time.time()

# 读取CSV文件
def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [(row["Title"], row["MP3 URL"]) for row in reader]
    return data

# 清理文件名
def clean_filename(filename):
    # 无差别删除 "机核 GCORES"
    filename = filename.replace("机核 GCORES", "")

    # 删除或替换文件名中的非法字符
    filename = re.sub(r'[\<\>:"/\\|?*]', '', filename)
    filename = filename.strip()

    return filename


# 下载MP3文件，并计算显示网速
def download_mp3(title, url, limiter):
    try:
        # Clean and prepare the URL and file path
        url = url.strip('\"')
        title = clean_filename(title)
        file_path = os.path.join(save_directory, f"{title}.mp3")
        
        # Start a session to handle redirects
        session = requests.Session()
        pre_response = session.get(url, allow_redirects=True)
        pre_response.raise_for_status()  # Ensure request was successful

        # Check if redirected URL is an HTML page containing the true MP3 link
        if 'text/html' in pre_response.headers.get('Content-Type', ''):
            soup = BeautifulSoup(pre_response.text, 'html.parser')
            audio_link = soup.find('a', class_='p_button')  # Adjust based on actual HTML structure
            if audio_link and 'href' in audio_link.attrs:
                url = audio_link['href']  # Extract actual MP3 URL
        
        # Proceed with downloading the MP3 file
        response = session.get(url, stream=True)
        response.raise_for_status()

        downloaded = 0
        start_time = time.time()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # limiter.wait(downloaded)  # 应用速度限制

                    # 显示下载速度
                    elapsed_time = time.time() - start_time
                    speed = display_speed(downloaded, elapsed_time)
                    print(f"Downloading {title} at {speed}", end='\r')

        # 下载完成后，立即记录到日志
        with open(log_file, "a") as log:
            log.write(f"{url}\n")

        print(f"\nCompleted: {title}")

    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as exc:
        print(f"Unexpected error occurred while downloading {url}: {exc}")



# 检查已下载文件
def check_downloaded(log_file):
    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            downloaded = set(file.read().splitlines())
    else:
        downloaded = set()
    return downloaded

# 主函数
def main():
    os.makedirs(save_directory, exist_ok=True)
    downloaded = check_downloaded(log_file)
    to_download = read_csv(csv_file_path)
    limiter = SpeedLimiter(max_speed)

    with tqdm(total=len(to_download)) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_mp3 = {executor.submit(download_mp3, title, url, limiter): url for title, url in to_download if url not in downloaded}

            for future in as_completed(future_to_mp3):
                url = future_to_mp3[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f"{url} generated an exception: {exc}")
                else:
                    pbar.update(1)
                    with open(log_file, "a") as log:
                        log.write(f"{url}\n")

if __name__ == "__main__":
    main()
