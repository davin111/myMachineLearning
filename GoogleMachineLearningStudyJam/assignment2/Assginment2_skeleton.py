# <이 괄호 안에 들어간 내용은 직접 구현할 부분입니다>
# 과제 1에서의 모델이 수행하는 일의 유형은 동일 - 따라서 그대로 써먹거나 참고해서 발전시킬 부분도 많음
# 자료의 개수를 5000개에서 10000개로 더욱 많게 하고, 하나의 자료에 대한 값들도 12개에서 200개로 늘림
# 과제 1은 데이터를 그대로 한 번 처리해서 찍는 수준(10% 정확도)을 보인, 사실상 '학습'이 없는 프로그램이었으나,
# 과제 2를 완료하면 드디어 기초적인 '학습'을 하는 모델을, 기존 머신러닝 라이브러리 없이 구현하는 위업을 달성

# 필요한 모듈들을 불러오기
import sys, os
sys.path.append(os.pardir)
from csv import reader
import numpy as np
from random import randrange


# 사용할 함수들, 클래스를 미리 정의해두기

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# 소프트맥스 함수
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

# 교차 엔트로피 오차 함수 - 과제 2에서 새로 등장('학습'에 필요)
def cross_entropy_error(x, t):
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)
    
    batch_size = x.shape[0]
    # <직접 구현하세요 - 정답 레이블을 one hot encoding했다고 전제>


# Load a CSV file
def load_csv(filename):
    x_train = list() # 필요한 리스트를 미리 만들어두기
    y_train = list() # 필요한 리스트를 미리 만들어두기
    
    x_test = list() # 필요한 리스트를 미리 만들어두기
    y_test = list() # 필요한 리스트를 미리 만들어두기
    
    with open(filename, 'r') as file: # 지정된 파일을 읽기 모드로 열어서 동작
        csv_reader = reader(file)
        
    # <직접 구현하세요 - 읽어 온 자료를 numpy 배열로 바꾸고 이 때 x_train, x_test와 y_train, y_test는 각각 float형과 int형으로
    # <직접 구현하세요 - 자료의 80%를 train set으로, 20%를 test set으로 랜덤하게(randrange() 함수 사용) 나눠서 사용>
    
    return x_train, y_train, x_test, y_test


# y_train, y_test를 one hot encoding으로 변형한다
def one_hot_encoding(x):
    # <직접 구현하세요 - y_train, y_test 정답 레이블을 one hot encoding으로 변형>



# Python에서 class는 여러 함수를 묶어 효율적으로 사용하는 것인데(엄밀한 설명 아님),
# https://wikidocs.net/28 참고(이 과제에 있어 근본적 이해는 딱히 필요 없으나, 'self가 뭐지?'라는 질문과 관련)
class TwoLayerNet:
    
    # 가중치와 편향을 필요에 맞게 생성한다
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {} # 필요한 딕셔너리를 미리 만들어 두기
        # 가중치와 편향을 입력층, 은닉층, 출력층 개수에 맞게 랜덤하게 생성
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    
    # 각 층들을 통해 정답을 예측한다
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        # <직접 구현하세요 - weight와 bias를 내적(product)한 후 sigmoid로 활성화>
        # <직접 구현하세요 - 출력층에서는 softmax 함수로 클래스별 확률을 구하기>
        return y
    
    
    # 정답 예측의 정확도를 계산한다
    def accuracy(self, x, t): # x는 자료에서 예측된 값 t는 정답으로 주어진 값
        y = self.predict(x) # x_train의 각 자료들마다 클래스별 확률을 구해서 y에 저장

        # <직접 구현하세요 - 예측된 값과 one hot encoding으로 된 y_train의 값을 비교하여 정확도를 출력>
        
        return accuracy

    
    # 손실 함수 - 과제 2에서 새로 등장('학습'에 필요)
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    

    # 기울기 계산 함수 - 과제 2에서 새로 등장('학습'에 필요)
    def gradient(self, x, t): # '밑바닥부터 시작하는 딥러닝' 4장의 numerical gradient와 방식 다름! 훨씬 연산이 빠르며, 5장과 관련
        # 현재의 weight와 bias 가져오기
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        grads = {} # weight와 bias의 기울기를 담을 딕셔너리 만들어두기
        batch_num = x.shape[0]
        
        # forward(순전파)
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward - '밑바닥부터 시작하는 딥러닝' 5장과 관련(역전파)
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy) # 기울기 결과 저장
        grads['b2'] = np.sum(dy, axis=0) # 기울기 결과 저장
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1) # 기울기 결과 저장
        grads['b1'] = np.sum(dz1, axis=0) # 기울기 결과 저장
        return grads



# 프로그램의 본격적 부분(indentation 없음)

# 데이터 준비시키기
filename = 'SimpleNetData2.csv'
x_train, y_train, x_test, y_test= load_csv(filename) # 지정된 csv 파일을 열어 x_train과 y_train, x_test와 y_test를 만들기
y_train = one_hot_encoding(y_train) # y_train을 one hot encoding으로 변형
y_test = one_hot_encoding(y_test) # y_test를 one hot encoding으로 변형


# 모델의 뼈대 만들어두기(설계한 층들 준비시키기)
network = TwoLayerNet(input_size=200, hidden_size=50, output_size=10)


# 하이퍼파라미터 설정하기
iters_num = 10000 # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100 # 미니배치 크기
learning_rate = 0.1 # 학습률 설정

iter_per_epoch = max(train_size / batch_size, 1) # 1에폭당 반복 수


# 학습 경과를 저장할 list 만들어두기
train_loss_list = []
train_acc_list = []
test_acc_list = []


# 학습하기
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    # 기울기 계산
    grad = network.gradient(x_batch, y_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        # <직접 구현하세요 - 현재 기울기를 기준으로 학습률을 고려하여 경사하강>

    # 학습 경과 기록
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    # 1에폭마다 정확도 계산하고 출력
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
