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
    print(df.head())

if __name__ == '__main__':
    main()
