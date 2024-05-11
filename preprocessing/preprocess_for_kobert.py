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

    # 불용어 리스트 정의
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    # 전처리 적용
    df['content'] = df['content'].apply(lambda x: text_preprocessing(x, stopwords))
    df['ocr_data'] = df['ocr_data'].apply(lambda x: text_preprocessing(x, stopwords))

    print(df[['content', 'ocr_data']].head())

if __name__ == '__main__':
    main()
