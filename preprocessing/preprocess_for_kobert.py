import pandas as pd
from konlpy.tag import Okt
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def text_preprocessing(text, stopwords):
    if pd.isnull(text) or text.strip() == "":
        return ""
    okt = Okt()
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
    df['ocr_data'] = df['ocr_data'].apply(lambda x: text_preprocessing(x, stopwords))
    df['combined_text'] = df['title'] + " " + df['ocr_data']
    df.to_csv('../data/processed_output.csv', index=False, encoding='utf-8-sig')

    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    MAX_LEN = 128

    input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(
        df['combined_text'].values, MAX_LEN, tokenizer
    )
    labels = torch.tensor(df['blog_is_promotional'].values)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels, test_size=0.2, random_state=42
    )
    train_masks, val_masks, _, _ = train_test_split(
        attention_masks, labels, test_size=0.2, random_state=42
    )
    train_types, val_types, _, _ = train_test_split(
        token_type_ids, labels, test_size=0.2, random_state=42
    )

    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_types, train_labels)
    val_data = TensorDataset(val_inputs, val_masks, val_types, val_labels)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size, num_workers=0)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size, num_workers=0)

    # 첫 배치 출력
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
        print(b_input_ids.shape)
        print(b_input_mask.shape)
        print(b_segment_ids.shape)
        print(b_labels.shape)
        break

if __name__ == '__main__':
    main()
