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

# Needed for OCR
import requests
import uuid
import time
import json

load_dotenv()

api_url = os.getenv("CLOVA_OCR_API_URL")
secret_key = os.getenv("CLOVA_OCR_SECRET_KEY")

promotional_text_set = {
'지원받고', '지원받아' '지원받았','지원받을수','지원을받은','지원을받아','지원을받고','지원을받았',
'제공받고', '제공받아','제공을받은','제공을받아','제공을받고','제공을받았','제공받을수', '제공받았',
'서포터즈', '파트너스', '소정의','수수료', '고료', '협찬','대가', '일정의', '지원금', '원고료'
}


request_json = {
    'images': [
        {
            'format': 'jpg',
            'name': 'demo'
        }
    ],
    'requestId': str(uuid.uuid4()),
    'version': 'V2',
    'timestamp': int(round(time.time() * 1000))
}

payload = {'message': json.dumps(request_json).encode('UTF-8')}

headers = {
  'X-OCR-SECRET': secret_key
}

image_dir = "images"
os.makedirs(image_dir, exist_ok=True)  

def get_ocr_data(image_url,cnt):
    img_text = ''
    img_response = requests.get(image_url)
    filename = os.path.join(image_dir, f"{cnt}.jpg")"
     # 요청이 성공적으로 수행되었는지 확인
    if img_response.status_code == 200:
        # 이미지 데이터를 바이너리 형태로 파일에 저장
        with open(filename, 'wb') as image_file:
            image_file.write(img_response.content)
            
    files = [('file', open(filename,'rb'))]
    response = requests.request("POST", api_url, headers=headers, data = payload, files = files)
    # 가정: response.text에 위에서 언급한 JSON 문자열이 들어 있다고 가정합니다.
    json_data = json.loads(response.text)
    
    # JSON 데이터 구조를 파악하여, 필요한 정보에 접근합니다.
    for image in json_data['images']:
        for field in image['fields']:
            img_text += field['inferText']

    return img_text.strip()

def tag_get_text(tag):
    if(tag):
        return tag.get_text().strip()
    else:
        return "없음"


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
query = "리뷰"
pageCnt = 380 #페이지 넘버
cnt = 1; #현재 크롤링 완료 블로그 수 
desired_cnt = 4

blog_posts = [] #블로그 정보를 저장할 배열

while cnt <= desired_cnt:
    try:
        driver.get(f"https://section.blog.naver.com/Search/Post.naver?pageNo={pageCnt}&rangeType=PERIOD&orderBy=recentdate&startDate=2022-03-01&endDate=2022-08-18&keyword={query}")
        time.sleep(2)
        all_html = driver.page_source
        soup = BeautifulSoup(all_html, 'html.parser')
        content_list = soup.find('div', class_='area_list_search') #현재 블로그 페이지에서 블로그 리스트를 변수에 저장  
        for a_tag in content_list.find_all('a', 'desc_inner'): #블로그 링크인 a태크를 찾아서 반복문 시작
            blog_content = "" #블로그 본문내용 초기화
            blog_title = "" #블로그 제목 초기화
            ocr_data = "" #OCR text 초기화
            blog_is_promotional = 0 #블로그 홍보성 여부 초기화
            comments_cnt = 0 #블로그 댓글 수 초기화
            
            
            href = a_tag['href'] #블로그 주소 변수에 저장
            driver.get(href)#블로그 주소로 이동
            time.sleep(2) 
            
            current_html = driver.page_source
            current_soup = BeautifulSoup(current_html, 'html.parser')
            
            iframe_src = current_soup.find('iframe').get('src') #iframe 태그안에 있는 각각의 블로그 정보를 크롤링위한 iframe의 경로 저장 
        
            if iframe_src.startswith('/'):
                iframe_src = 'https://blog.naver.com' + iframe_src #iframe 경로와 블로그 주소를 합침
            
            driver.get(iframe_src) #iframe 경로로 이동
            time.sleep(2)
            
            iframe_html = driver.page_source
            iframe_soup = BeautifulSoup(iframe_html, 'html.parser')
            title = iframe_soup.find('h3', class_='se_textarea')#블로그 제목 탐색
            content = iframe_soup.select('span[class^="se-fs-"]') #블로그 본문 탐색
            date = iframe_soup.find('span',class_='se_publishDate pcol2').get_text() #블로그 게시일 탐색
            writer = iframe_soup.find('a',class_='link pcol2').get_text() #블로그 작성자 탐색
            images = iframe_soup.select('img[class$="egjs-visible"]') or "이미지 없음"
            empathy_cnt = iframe_soup.find('em',class_='u_cnt _count')#블로그 공감 수
            writer_review_str = iframe_soup.find('h4',class_='category_title pcol2')#블로그 개시글 수
            comments_cnt = iframe_soup.find('em',class_='_commentCount')#블로그 댓글 수

            empathy_cnt = tag_get_text(empathy_cnt)
            writer_review_str = tag_get_text(writer_review_str)
            comments_cnt = tag_get_text(comments_cnt)

            print(href)
            if(writer_review_str != "없음"):
                writer_reviews = re.findall(r'\d+', writer_review_str)
                writer_reviews_cnt = int(writer_reviews[0])

            if title:
                blog_title = title.get_text()
            elif content:
                blog_title = content[0].get_text()
            
            for tag in content:
               blog_content += tag.get_text()

            blog_content = blog_content.replace('\u200b', '').replace(' ', '')
            
            if(images != "이미지 없음"):
                image_url = images[-1]['src']
                ocr_data = get_ocr_data(image_url,cnt)
    
            # Combine blog content and title for efficiency
            combined_text = blog_content + blog_title + ocr_data
            
            for promo_word in promotional_text_set:
                if promo_word in combined_text:  # Checks if the promo_word is a substring of combined_text
                    blog_is_promotional = 1
                    break  # A promotional word is found, no need to check further
            
            blog_posts.append({'cnt':cnt,'writer':writer.strip(),'date':date.strip(),'url':href.strip(),'title':blog_title.strip(),'content':blog_content,'ocr_data':ocr_data,'comments_cnt':comments_cnt,'empathy_cnt':empathy_cnt,'writer_reviews_cnt':writer_reviews_cnt,'blog_is_promotional':blog_is_promotional})
            cnt += 1;
    
    except Exception as e:
        # 모든 종류의 예외를 잡습니다.
        print("발생한 오류:", e, '\n다음 리뷰로 넘어갑니다.')
    pageCnt += 1
    print('성공','  pageCnt : ',pageCnt, '   href : ',href)
# CSV 파일로 저장
with open('output.csv', 'w', newline='', encoding='utf-8-sig') as file:
    # csv.DictWriter 객체 생성, fieldnames에는 사전의 모든 키를 리스트 형태로 제공
    fieldnames = ['cnt','writer','date','url','title','content','ocr_data','comments_cnt','empathy_cnt','writer_reviews_cnt','blog_is_promotional']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # 열 제목(헤더) 작성
    writer.writeheader()
    
    # 사전 데이터를 행으로 작성
    for item in blog_posts:
        writer.writerow(item)

driver.quit()

# 결과 출력 (여기서는 처음 5개의 게시물만 출력합니다)

