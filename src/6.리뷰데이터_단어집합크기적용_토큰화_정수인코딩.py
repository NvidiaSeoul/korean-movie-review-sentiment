import numpy as np
import pandas as pd

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

# 한국어 불용어 제거 데이터 로딩
X_train_data = pd.read_csv('train_stopwords_reviews.csv',usecols=['document','label'])
X_test_data = pd.read_csv('test_stopwords_reviews.csv',usecols=['document','label'])
print(X_train_data.head(10))
print(X_test_data.head(10))
print(X_train_data.info())
print(X_test_data.info())

# 한국어 불용어 제거로 발생한 결측 데이터 삭제
print(X_train_data.loc[X_train_data['document'].isnull()])
print(X_test_data.loc[X_test_data['document'].isnull()])
X_train_data.dropna(how='any',inplace=True)
X_test_data.dropna(how='any',inplace=True)


# 훈련, 테스트 데이터 토큰화 및 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_size = 11775 # 단어 빈도수 체크 결과에 따른 11775개 단어 집합 사용
tokenizer = Tokenizer(word_size) # ( 0 예약 패딩, 1~11774까지 단어 집합 사용 )
tokenizer.fit_on_texts(X_train_data['document'])
print(tokenizer.word_index) # 21800 개 이상의 단어가 존재

# train 데이터와 test data 리뷰 문장을 0 패딩, 1~11774까지 단어 집합 사용해 정수 시퀀스 데이터 형태로 변환
# 변환된 데이터를 'sequences'컬럼으로 추가
X_train_data['sequences'] = tokenizer.texts_to_sequences(X_train_data['document'])
X_test_data['sequences'] = tokenizer.texts_to_sequences(X_test_data['document'])

X_train_data.reset_index(drop=True, inplace=True)  # 인덱스 초기화
X_test_data.reset_index(drop=True, inplace=True)   # 인덱스 초기화

print(X_train_data[25:30])
print(X_test_data[57:62])

# 11775 개 단어 집합만 고려 했음으로 빈도수가 1 이하인 단어로 이루어진 문장은 텅빈( [ ] )
# 형태로 변환 됨, 따라서 해당 문장의 인덱스를 찾아 제거 해줌
drop_train_idx = [idx for idx, sentence in enumerate(X_train_data['sequences']) if len(sentence) < 1]
print('drop_train_idx : \n', drop_train_idx)

drop_test_idx = [idx for idx, sentence in enumerate(X_test_data['sequences']) if len(sentence) < 1]
print('drop_test_idx : \n', drop_test_idx)

# 텅빈([ ]) sequence 데이터 위치 인덱스 활용해서  Dataframe 해당 행 삭제
X_train_data.drop(drop_train_idx,axis=0, inplace=True)
X_test_data.drop(drop_test_idx, axis=0, inplace=True)

X_train_data.reset_index(drop=True, inplace=True)  # 인덱스 초기화
X_test_data.reset_index(drop=True, inplace=True)   # 인덱스 초기화

print("========= 삭제 완료 검증 수행 ===========")
for idx, sequence in enumerate(X_train_data['sequences']):
    if(len(sequence) < 1):
        print(idx, sequence)

print(X_train_data[25:30])
print(X_test_data[57:62])

# 타깃 라벨 추출
y_train = np.array(X_train_data['label'])
y_test = np.array(X_test_data['label'])

print(len(X_train_data['sequences']))   # 최종 훈련데이터 31901 개 샘플
print(len(y_train))                     # 최종 훈련데이터 라벨 31901 개
print(len(X_test_data['sequences']))    # 최종 테스트데이터 31554 개 샘플
print(len(y_test))                      # 최종 테스트데이터 라벨 31554 개

train_review_sequences_len = [len(sequence) for sequence in  X_train_data['sequences']]
train_review_sequences_arr = np.array(train_review_sequences_len)
print('max : ', np.max(train_review_sequences_arr))  # 훈련 리뷰데이터 최대 길이 63
print('mean : ', np.mean(train_review_sequences_arr)) # 평균 길이 10.734114918027648
#
# # import matplotlib.pyplot as plt
# # plt.hist(train_review_sequences_len, bins=50)
# # plt.show() # pad 적용 30 길이로 동일하게 맞추자
#
X_train_pades = pad_sequences(X_train_data['sequences'], maxlen=30)
X_test_pades = pad_sequences(X_test_data['sequences'], maxlen=30)

print(len(X_train_pades[0]))
print(X_train_pades[:1])
print(len(X_test_pades[0]))
print(X_test_pades[:1])

# 최종 LSTM 모델 훈련 데이터 준비 완료
# 훈련데이터 ( X_train_pades , y_train )
# 테스트데이터 ( X_test_pades, y_test )