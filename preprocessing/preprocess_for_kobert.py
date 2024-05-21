import pandas as pd
from konlpy.tag import Okt
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

okt = Okt()

def text_preprocessing(text, stopwords):
    if pd.isnull(text) or text.strip() == "":
        return ""
    text = text.replace("\n", " ")
    text = okt.morphs(text, stem=True)
    text = [word for word in text if word not in stopwords]
    return " ".join(text)

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
        torch.tensor(input_ids),
        torch.tensor(attention_masks),
        torch.tensor(token_type_ids)
    )

def main():
    df = pd.read_csv('../data/output.csv', encoding='utf-8-sig')

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    df['content'] = df['content'].apply(lambda x: text_preprocessing(x, stopwords))
    df['ocr_data'] = df['ocr_data'].apply(lambda x: text_preprocessing(x, stopwords) if pd.notnull(x) else "")

    df['title'] = df['title'].fillna("")
    df['ocr_data'] = df['ocr_data'].fillna("")
    df['content'] = df['content'].fillna("")

    df['combined_text'] = df['title'] + " " + df['ocr_data'] + " " + df['content']

    # 필요한 컬럼만 추출
    df = df[['cnt', 'combined_text', 'blog_is_promotional']]
    df.to_csv('../data/processed_output.csv', index=False, encoding='utf-8-sig')

    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    MAX_LEN = 128

    input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(
        df['combined_text'].values, MAX_LEN, tokenizer
    )
    labels = torch.tensor(df['blog_is_promotional'].values)

    # 인덱스를 기준으로 train/val 나누기
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

    # 첫 배치 shape 확인용 출력
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
        print(b_input_ids.shape)
        print(b_input_mask.shape)
        print(b_segment_ids.shape)
        print(b_labels.shape)
        break

if __name__ == '__main__':
    main()
