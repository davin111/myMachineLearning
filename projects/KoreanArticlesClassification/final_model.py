# made by Byeon Davin
https://github.com/davin111/myMachineLearning

### 0. 사용할 패키지 불러오고 함수 정의하기
from os import listdir
import re
from itertools import repeat

from konlpy.tag import Mecab
import numpy as np

import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dropout, Flatten, Dense, GlobalMaxPooling1D, Merge, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D

import matplotlib.pyplot as plt

# 랜덤시드 고정시키기 및 임베딩 차원 수 결정
np.random.seed(7)
EMBEDDING_DIM = 100


# 파일 불러와 읽기
def load_doc(filename):
    file = open(filename, 'rU', encoding='utf-8')
    text = file.read()
    file.close()
    return text

# 단어를 형태소 단위로 나눠 태그를 붙여 반환, 유효 형태소만 취급
def clean_doc(doc):
    words = doc.split()
    tokens = []
    get = ['NNG', 'NNP', 'VV', 'VA', 'VV+EC', 'VA+EC', 'VV+EP', 'VA+EP', 'VV+ETM', 'VA+ETM', 'VV+EP+EC', 'VA+EP+EC', 'VV+EC+VX+EP', 'VA+EC+VX+EP', 'VV+EC+VX+EC', 'VA+EC+VX+EC', 'VV+EC+VX+ETM', 'VA+EC+VX+ETM']
    for word in words:
        morphs = [tag for tag in tagger.pos(word)]
        for morph in morphs:
            if morph[1] in get:
                tokens.append(morph[0])
    return tokens

# 데이터를 불러와 전처리하기
def process_docs(directory, is_train):
    documents = list()
    p = re.compile('.1[6-9]+')
    for filename in listdir(directory):
        # train set인지 아닌지에 따라 구분
        if is_train and p.match(filename):
            continue
        if not is_train and not p.match(filename):
            continue
        documents.append(clean_doc(load_doc(directory + '/' + filename)))
    return documents

# padding 또는 cutting
def pad_cut_sequences(sequences, number, width):
    cnt_up = 0
    cnt_down = 0
    for doc in sequences:
        if(len(doc) <= width):
            doc.extend(repeat(number, width - len(doc)))
            cnt_up += 1
        else:
            del doc[width:]
            cnt_down += 1
    pad_cut_seq_out = sequences
    print("PAD:", cnt_up, "CUT:", cnt_down)
    return pad_cut_seq_out



### 1. 데이터 준비하고 데이터셋 생성하기
tagger = Mecab()

# train set 전처리
train_docs = []
train_t = []
for i in range(8):
    train_docs += process_docs('newsData/' + str(i), True)
    train_t += [i for _ in range(160)]

# test set 전처리
test_docs = []
test_t = []
for i in range(8):
    test_docs += process_docs('newsData/' + str(i), False)
    test_t += [i for _ in range(40)]

# 데이터셋 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)
encoded_docs_train = tokenizer.texts_to_sequences(train_docs)
max_length_train = max([len(s) for s in encoded_docs_train])
encoded_docs_test = tokenizer.texts_to_sequences(test_docs)
max_length_test = max([len(s) for s in encoded_docs_test])

# length 선택 받기
L = int(input("\n최장 문서의 길이로 padding할 것이면 0(기본),\n그렇지 않으면 원하는 길이를 입력하세요.(예: 100)\n"))
# vocab_size 및 length 준비
if L == 0:
    length = max(max_length_train, max_length_test)
else:
    length = L
vocab_size = len(tokenizer.word_index) + 1

# train set과 test set의 길이를 조절하고 numpy로 변환
x_train = pad_cut_sequences(encoded_docs_train, 0, length)
x_train = np.array(x_train)
t_train = np.array(train_t)

x_test = pad_cut_sequences(encoded_docs_test, 0, length)
x_test = np.array(x_test)
t_test = np.array(test_t)



### 2. 모델 구성하기
# 동일한 크기 필터 여부 선택 받기
F = int(input("\nCONV층에서 동일한 크기의 필터를 사용할 것이면 0(기본),\n그렇지 않으면 1을 입력하세요.\n"))
# 활성화 함수 종류 선택 받기
A = int(input("\n활성화 함수로 relu를 사용할 것이면 0(기본),\ntanh를 사용할 것이면 1을 입력하세요.\n"))
if A == 0:
    act = 'relu'
elif A == 1:
    act = 'tanh'

# 동일한 크기의 필터 사용
if F == 0:
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=length))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=8, activation=act))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

# 서로 다른 크기의 필터 사용
elif F == 1:
    # 서로 다른 크기의 필터 세 개를 잇기
    submodels = []
    for ks in (7, 8, 9, 10): # kernel_size
        submodel = Sequential()
        submodel.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=length, trainable=True))
        submodel.add(Conv1D(filters=8, kernel_size=ks, padding='valid', activation=act, strides=1))
        submodel.add(GlobalMaxPooling1D())
        submodels.append(submodel)
        print(submodel.summary())
    model = Sequential()
    model.add(Merge(submodels, mode="concat"))

# 모델의 공통 부분 구성
model.add(Dropout(0.2))
model.add(Dense(50, activation=act))
model.add(Dense(8, activation='softmax'))
print(model.summary())



### 3. 모델 학습과정 설정 및 학습시키기 & 평가하기
if F == 0:
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    hist = model.fit(x_train, t_train, epochs=10, batch_size=10, validation_data=(x_test, t_test))
    scores = model.evaluate(x_test, t_test)

elif F == 1:
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    hist = model.fit([x_train, x_train, x_train, x_train], t_train, epochs=10, batch_size=10, validation_data=([x_test, x_test, x_test, x_test], t_test))
    scores = model.evaluate([x_test, x_test, x_test, x_test], t_test)



### 4. 최종 결과 출력
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# epoch에 따른 성능 그래프로 나타내기
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'g', label='val loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'r', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='lower left')
acc_ax.legend(loc='upper left')

plt.show()
