import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# 데이터 불러오기
df = pd.read_csv(r'C:\Users\NM333-85\Desktop\blog_crawling_file.csv', encoding='cp949')

# 불용어 리스트 정의
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 텍스트 전처리 함수
def text_preprocessing(text):
    print("2")
    if pd.isnull(text) or text.strip() == "":  # NaN 값 및 빈 문자열 처리
        return ""
    okt = Okt()
    text = text.replace("\n", " ")  # 줄바꿈을 공백으로 변경
    text = okt.morphs(text, stem=True)  # 형태소 분석을 통한 토큰화 및 어간 추출
    text = [word for word in text if word not in stopwords]  # 불용어 제거
    return " ".join(text)

# 'content'와 'ocr_data' 열에 텍스트 전처리 적용
df['content'] = df['content'].apply(text_preprocessing)
df['ocr_data'] = df['ocr_data'].apply(text_preprocessing)

# 타이틀과 OCR 데이터를 합쳐서 새로운 특성 생성
df['combined_text'] = df['title'] + " " + df['ocr_data']

# 전처리된 데이터 저장
df.to_csv(r'C:\Users\NM333-85\Desktop\processed_blog_crawling_file.csv', index=False, encoding='cp949')

# KoBERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

# 최대 시퀀스 길이 설정
MAX_LEN = 128

# 텍스트 데이터를 KoBERT 입력 형식으로 변환하는 함수
def convert_to_kobert_inputs(text_list, max_len):
    print("1")
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for text in text_list:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 스페셜 토큰 추가
            max_length=max_len,  # 최대 길이 설정
            padding='max_length',  # 패딩
            return_attention_mask=True,  # 어텐션 마스크 생성
            truncation=True
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])  # 세그먼트 인덱스

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(token_type_ids)

# 'combined_text' 열을 KoBERT 입력 형식으로 변환
input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(df['combined_text'].values, MAX_LEN)
labels = torch.tensor(df['blog_is_promotional'].values)

# 학습 데이터와 테스트 데이터 분리
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.2, random_state=42)
train_type_ids, validation_type_ids, _, _ = train_test_split(token_type_ids, labels, test_size=0.2, random_state=42)

# 데이터 로더 생성
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_type_ids, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=4)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_type_ids, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, num_workers=4)

# 데이터 로더 확인
for batch in train_dataloader:
    b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
    print(b_input_ids.shape)
    print(b_input_mask.shape)
    print(b_segment_ids.shape)
    print(b_labels.shape)
    break