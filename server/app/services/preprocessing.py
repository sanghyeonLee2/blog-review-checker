from konlpy.tag import Okt
import torch

okt = Okt()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 텍스트 전처리 함수
def text_preprocessing(text: str, stopwords: list = stopwords) -> str:
    if not text or not text.strip():
        return ""
    text = text.replace("\n", " ")
    text = text.replace("\u200b", "")
    text = " ".join(text.split())  # 여러 개의 공백을 하나로 정리
    tokens = okt.morphs(text, stem=True)
    return " ".join([word for word in tokens if word not in stopwords])

# dict 형태 입력을 받아 전처리하는 함수 (title + ocr_data + content 합쳐서 한 번에 처리)
def preprocess_text_object(data: dict) -> str:
    combined_text = f"{data.get('title', '')} {data.get('ocr_data', '')} {data.get('content', '')}"
    return text_preprocessing(combined_text)

# KoBERT 입력 형식 변환 함수
def convert_to_kobert_inputs(text, max_len, tokenizer):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        truncation=True
    )

    input_ids = torch.tensor([encoded_dict['input_ids']], dtype=torch.long)
    attention_mask = torch.tensor([encoded_dict['attention_mask']], dtype=torch.long)
    token_type_ids = torch.tensor([encoded_dict['token_type_ids']], dtype=torch.long)

    return input_ids, attention_mask, token_type_ids
