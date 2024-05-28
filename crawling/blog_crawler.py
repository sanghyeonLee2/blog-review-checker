import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import csv
import re
import uuid
import time
import json

load_dotenv()

api_url = os.getenv("CLOVA_OCR_API_URL")
secret_key = os.getenv("CLOVA_OCR_SECRET_KEY")

# 광고성 키워드 셋
promotional_text_set = {
    '지원받고', '지원받아', '지원받았', '지원받을수', '지원을받은', '지원을받아', '지원을받고', '지원을받았',
    '제공받고', '제공받아', '제공을받은', '제공을받아', '제공을받고', '제공을받았', '제공받을수', '제공받았',
    '서포터즈', '파트너스', '소정의', '수수료', '고료', '협찬', '대가', '일정의', '지원금', '원고료'
}

# OCR 요청용 기본 JSON
request_json = {
    'images': [{'format': 'jpg', 'name': 'demo'}],
    'requestId': str(uuid.uuid4()),
    'version': 'V2',
    'timestamp': int(round(time.time() * 1000))
}
payload = {'message': json.dumps(request_json).encode('UTF-8')}
headers = {'X-OCR-SECRET': secret_key}

# 디렉토리 생성
image_dir = "images"
csv_dir = "../data"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, "output.csv")

def get_ocr_data(image_url, cnt):
    img_text = ''
    filename = os.path.join(image_dir, f"{cnt}.jpg")
    img_response = requests.get(image_url)
    if img_response.status_code == 200:
        with open(filename, 'wb') as image_file:
            image_file.write(img_response.content)
    files = [('file', open(filename, 'rb'))]
    response = requests.post(api_url, headers=headers, data=payload, files=files)
    json_data = json.loads(response.text)
    for image in json_data.get('images', []):
        for field in image.get('fields', []):
            img_text += field.get('inferText', '')
    return img_text.strip()

def tag_get_text(tag):
    return tag.get_text().strip() if tag else None

# 드라이버 실행
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
query = "리뷰"
pageCnt = 380
cnt = 1
desired_cnt = 30

blog_posts = []

while cnt <= desired_cnt:
    try:
        driver.get(f"https://section.blog.naver.com/Search/Post.naver?pageNo={pageCnt}&rangeType=PERIOD&orderBy=recentdate&startDate=2022-03-01&endDate=2022-08-18&keyword={query}")
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        content_list = soup.find('div', class_='area_list_search')
        for a_tag in content_list.find_all('a', 'desc_inner'):
            blog_content, blog_title, ocr_data = "", "", ""
            blog_is_promotional, comments_cnt = 0, 0

            href = a_tag['href']
            driver.get(href)
            time.sleep(2)

            current_html = driver.page_source
            current_soup = BeautifulSoup(current_html, 'html.parser')
            iframe_src = current_soup.find('iframe').get('src')
            if iframe_src.startswith('/'):
                iframe_src = 'https://blog.naver.com' + iframe_src
            driver.get(iframe_src)
            time.sleep(2)

            iframe_html = driver.page_source
            iframe_soup = BeautifulSoup(iframe_html, 'html.parser')

            title = iframe_soup.find('h3', class_='se_textarea')
            content = iframe_soup.select('span[class^="se-fs-"]')
            date = tag_get_text(iframe_soup.find('span', class_='se_publishDate pcol2'))
            writer = tag_get_text(iframe_soup.find('a', class_='link pcol2'))
            images = iframe_soup.select('img[class$="egjs-visible"]')
            empathy_cnt = tag_get_text(iframe_soup.find('em', class_='u_cnt _count'))
            writer_review_str = tag_get_text(iframe_soup.find('h4', class_='category_title pcol2'))
            comments_cnt = tag_get_text(iframe_soup.find('em', class_='_commentCount'))

            writer_reviews_cnt = 0
            if writer_review_str:
                writer_reviews = re.findall(r'\d+', writer_review_str)
                writer_reviews_cnt = int(writer_reviews[0]) if writer_reviews else 0

            blog_title = title.get_text() if title else (content[0].get_text() if content else "")
            for tag in content:
                blog_content += tag.get_text()
            blog_content = blog_content.replace('\u200b', '').replace(' ', '')

            if images:
                image_url = images[-1]['src']
                ocr_data = get_ocr_data(image_url, cnt)
            else:
                ocr_data = ""

            combined_text = blog_content + blog_title + ocr_data
            for promo_word in promotional_text_set:
                if promo_word in combined_text:
                    blog_is_promotional = 1
                    break

            blog_posts.append({
                'cnt': cnt,
                'writer': writer,
                'date': date,
                'url': href.strip(),
                'title': blog_title.strip(),
                'content': blog_content,
                'ocr_data': ocr_data,
                'comments_cnt': comments_cnt,
                'empathy_cnt': empathy_cnt,
                'writer_reviews_cnt': writer_reviews_cnt,
                'blog_is_promotional': blog_is_promotional
            })

            if cnt % 500 == 0:
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as file:
                    fieldnames = ['cnt','writer','date','url','title','content','ocr_data','comments_cnt','empathy_cnt','writer_reviews_cnt','blog_is_promotional']
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in blog_posts:
                        writer.writerow(item)
                print(f"🔄 중간 저장 완료 - {cnt}개")

            cnt += 1
            if cnt > desired_cnt:
                break
    except Exception as e:
        print("발생한 오류:", e, '\n다음 리뷰로 넘어갑니다.')
    pageCnt += 1
    print('성공', '  pageCnt : ', pageCnt, '   href : ', href)

# CSV 저장
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as file:
    fieldnames = ['cnt','writer','date','url','title','content','ocr_data','comments_cnt','empathy_cnt','writer_reviews_cnt','blog_is_promotional']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for item in blog_posts:
        writer.writerow(item)

driver.quit()