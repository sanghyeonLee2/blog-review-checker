import pandas as pd
from konlpy.tag import Okt

def text_preprocessing(text, stopwords):
    if pd.isnull(text) or text.strip() == "":
        return ""
    okt = Okt()
    text = text.replace("\n", " ")
    text = okt.morphs(text, stem=True)
    text = [word for word in text if word not in stopwords]
    return " ".join(text)

def main():
    df = pd.read_csv('../data/output.csv', encoding='utf-8-sig')

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    df['content'] = df['content'].apply(lambda x: text_preprocessing(x, stopwords))
    df['ocr_data'] = df['ocr_data'].apply(lambda x: text_preprocessing(x, stopwords))

    # title + ocr_data 필드 결합
    df['combined_text'] = df['title'] + " " + df['ocr_data']

    # 전처리된 파일 저장
    df.to_csv('../data/processed_output.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()
