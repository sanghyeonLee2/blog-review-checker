import pandas as pd
from konlpy.tag import Okt
from kobert_transformers import get_tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import re

okt = Okt()

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.replace("\n", " ").replace("\u200b", "")
    text = " ".join(text.split())

    text = re.sub(r'[-=~_*]{2,}', ' ', text)  # 특수문자 반복 제거
    text = re.sub(r'\b[a-zA-Z]{5,}\b', '', text)  # 긴 영문 제거
    text = re.sub(r'\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}', '', text)  # 전화번호 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', text)  # 단어 반복 제거

    return text.strip()

def text_preprocessing(text, stopwords):
    text = clean_text(text)
    tokens = okt.morphs(text, stem=True)
    return " ".join([word for word in tokens if word not in stopwords])

def convert_to_kobert_inputs(text_list, max_len, tokenizer):
    input_ids, attention_masks, token_type_ids = [], [], []

    for text in text_list:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(token_type_ids, dtype=torch.long)
    )

def main():
    df = pd.read_csv('../data/output.csv', encoding='utf-8-sig')

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    df['title'] = df['title'].fillna("")
    df['ocr_data'] = df['ocr_data'].fillna("")
    df['content'] = df['content'].fillna("")

    df['title'] = df['title'].apply(lambda x: text_preprocessing(x, stopwords))
    df['ocr_data'] = df['ocr_data'].apply(lambda x: text_preprocessing(x, stopwords))
    df['content'] = df['content'].apply(lambda x: text_preprocessing(x, stopwords))

    df = df[df['content'].str.strip() != '']

    df.reset_index(drop=True, inplace=True)
    df['cnt'] = df.index + 1

    print("\n전처리 샘플 확인 (상위 5개)")
    for i in range(5):
        print(f"▶ [index {i}]")
        print("title:", df.loc[i, 'title'])
        print("ocr_data:", df.loc[i, 'ocr_data'])
        print("content:", df.loc[i, 'content'])
        print("-" * 50)

    df['combined_text'] = df['title'] + " " + df['ocr_data'] + " " + df['content']
    df = df[['cnt', 'combined_text', 'blog_is_promotional']]

    df.to_csv('../data/processed_output.csv', index=False, encoding='utf-8-sig')
    print("전처리 완료 및 저장: processed_output.csv")

    tokenizer = get_tokenizer()
    MAX_LEN = 128

    input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(
        df['combined_text'].values, MAX_LEN, tokenizer
    )
    labels = torch.tensor(df['blog_is_promotional'].values, dtype=torch.long)

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_inputs, val_inputs = input_ids[train_idx], input_ids[val_idx]
    train_masks, val_masks = attention_masks[train_idx], attention_masks[val_idx]
    train_types, val_types = token_type_ids[train_idx], token_type_ids[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_types, train_labels)
    val_data = TensorDataset(val_inputs, val_masks, val_types, val_labels)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size, num_workers=0)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size, num_workers=0)

    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
        print("\n첫 번째 배치 텐서 크기:")
        print("input_ids:", b_input_ids.shape)
        print("attention_mask:", b_input_mask.shape)
        print("token_type_ids:", b_segment_ids.shape)
        print("labels:", b_labels.shape)
        break

if __name__ == '__main__':
    main()
