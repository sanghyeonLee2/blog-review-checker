import pandas as pd
from konlpy.tag import Okt
from transformers import BertTokenizer

def text_preprocessing(text, stopwords):
    if pd.isnull(text) or text.strip() == "":
        return ""
    okt = Okt()
    text = text.replace("\n", " ")
    text = okt.morphs(text, stem=True)
    text = [word for word in text if word not in stopwords]
    return " ".join(text)

def convert_to_kobert_inputs(text_list, max_len, tokenizer):
    input_ids = []
    attention_masks = []
    token_type_ids = []

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

    return input_ids, attention_masks, token_type_ids

def main():
    df = pd.read_csv('../data/output.csv', encoding='utf-8-sig')

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    df['content'] = df['content'].apply(lambda x: text_preprocessing(x, stopwords))
    df['ocr_data'] = df['ocr_data'].apply(lambda x: text_preprocessing(x, stopwords))
    df['combined_text'] = df['title'] + " " + df['ocr_data']
    df.to_csv('../data/processed_output.csv', index=False, encoding='utf-8-sig')

    # KoBERT tokenizer 로드
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    MAX_LEN = 128

    # 입력 변환
    input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(
        df['combined_text'].values, MAX_LEN, tokenizer
    )

if __name__ == '__main__':
    main()
