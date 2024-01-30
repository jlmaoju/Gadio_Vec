#This one scan the website and collect the mp3 link/Gadio tile/and orther info, then save it into the csv file

import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm
import logging

# Setup Logging
logging.basicConfig(filename=r"errors.log", level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Specify the path to save the CSV file
csv_file_path = r"1st.csv"

def fetch_and_extract(url_num):
    try:
        url = f'https://www.gcores.com/radios/{url_num}'
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title').text if soup.find('title') else "No Title Found"

            category_tag = soup.find('a', class_='u_color-category')
            category = category_tag.text if category_tag else "No Category Found"

            story_desc_blocks = soup.find_all("div", class_="story_block-text")
            story_description = " ".join(block.get_text(strip=True) for block in story_desc_blocks)

            original_desc_tag = soup.find('p', class_='originalPage_desc')
            original_description = original_desc_tag.text if original_desc_tag else ""

            full_description = (story_description + " " + original_description).replace('\n', ' ').replace('\r', '')

            mp3_url = None
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and '.mp3' in href:
                    mp3_url = href
                    break

            if mp3_url:  # Proceed only if mp3 link exists
                # Return the data with each field enclosed in quotes
                return (f'"{url_num}"', f'"{title}"', f'"{category}"', f'"{full_description}"', f'"{mp3_url}"')
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException for URL number {url_num}: {e}")
        return None
    except Exception as e:
        logging.error(f"General Exception for URL number {url_num}: {e}")
        return None

def main():
    try:
        with open(csv_file_path, 'x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["URL Number", "Title", "Category", "Description", "MP3 URL"])
    except FileExistsError:
        pass  # If file already exists, skip creating it

    for url_num in tqdm(range(0, 176001)):  # Update the range if necessary
        result = fetch_and_extract(url_num)
        if result:  # If a result was returned, open the file and write it
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:  # 'a' is for append
                writer = csv.writer(file)
                writer.writerow(result)

if __name__ == "__main__":
    main()
