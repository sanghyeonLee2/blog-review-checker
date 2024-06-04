import os
import requests
import uuid
import time
import json
from dotenv import load_dotenv

dotenv_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    '.env'
)

load_dotenv(dotenv_path)

CLOVA_OCR_URL = os.getenv("CLOVA_OCR_API_URL")
CLOVA_SECRET_KEY = os.getenv("CLOVA_OCR_SECRET_KEY")

def get_ocr_text(image_url):
    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(time.time() * 1000)
    }
    headers = {
        'X-OCR-SECRET': CLOVA_SECRET_KEY
    }
    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    # 이미지 다운로드
    image_data = requests.get(image_url).content
    files = [('file', ('image.jpg', image_data, 'application/octet-stream'))]

    response = requests.post(CLOVA_OCR_URL, headers=headers, data=payload, files=files)
    result = response.json()

    text = ''
    for image in result.get('images', []):
        for field in image.get('fields', []):
            text += field.get('inferText', '') + ' '

    return text.strip()
