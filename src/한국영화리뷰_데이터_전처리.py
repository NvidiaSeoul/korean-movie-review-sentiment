import numpy as np
import pandas as pd

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth',50)

reviewdf = pd.read_csv('/home/sckit/deeplearning_prj/20260618/ratings_train.csv',
                       header=0,delimiter='\t',quoting=3)
print(reviewdf)
reviewdf.info()
reviewdf.dropna(how='any', inplace=True)
# 라벨(타겟) 컬럼데이터의 type을 실수에서 정수로 변환해주자.
reviewdf['label'] = reviewdf['label'].astype('int64')


# 리뷰 데이터의 항목중 중복 데이터가 있으면 찾아서 제거.
print(reviewdf['document'].nunique()) # nnunique() ==> 유니크한 항목의 개수를반환
reviewdf.drop_duplicates(subset='document', inplace=True)

reviewdf.info()
print(reviewdf.head())
# 결측치 제거되고 중복이  제거된 총데이터의 개수는 ==> 32163 개

# 한글과 공백을 제외한 모든 문자를 제거
#reviewdf['document'] = reviewdf['document'].str.replace(r'[^ㄱ-힣\s]','')
import re
def reviewfiltering(arg):
    return re.sub(r'[^ㄱ-힣\s]','',arg)
    
reviewdf['document'] = reviewdf['document'].apply(reviewfiltering)

print(reviewdf.sample(100))
print(reviewdf.info())

from konlpy.tag import Okt
from tqdm import tqdm  # 처리 상태를 막대 바로 표현

okt = Okt()

stopwords = ['의','가','이','은','들','는','좀','줄', '잘','걍','과','도','를','으로','자','에','와','한','하다']

X_train = []
for sentence in tqdm(reviewdf['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 각 문장을 토큰화
    sentence_removed_stopwords = \
    [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    #불용어제거된단어리스트를한문장으로합친다음X_trainlist에추가
    X_train.append(' '.join(sentence_removed_stopwords))

# print(reviewdf[:5])
# print('='*80)
# print(X_train[:5])

reviewdf['document'] = X_train

print(reviewdf)