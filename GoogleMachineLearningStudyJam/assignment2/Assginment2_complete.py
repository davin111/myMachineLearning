# 필요한 모듈들을 불러오기
import sys, os
sys.path.append(os.pardir)
from csv import reader
import numpy as np
from random import randrange


# 사용할 함수들 미리 def로 정의해두기

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

# 교차 엔트로피 오차 함수
def cross_entropy_error(x, t):
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)
    
    batch_size = x.shape[0]
    return -np.sum(t * np.log(x + 1e-7)) / batch_size


# Load a CSV file
def load_csv(filename):
    x_train = list() # 필요한 리스트를 미리 만들어두기
    y_train = list() # 필요한 리스트를 미리 만들어두기
    
    x_test = list()
    y_test = list()
    
    with open(filename, 'r') as file: # 지정된 파일을 읽기 모드로 열어서 동작
        csv_reader = reader(file)
        
        for row in csv_reader: # 주어진 csv의 각 행에 대해서
            x_train.append(list(map(float, row[:200])))
            y_train.append(int(row[-1]))

    train_size = 0.2 * len(x_train)
    while len(x_test) < train_size:
        index = randrange(len(x_train))
        x_test.append(x_train.pop(index))
        y_test.append(y_train.pop(index))
        
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    return x_train, y_train, x_test, y_test


# y_train, y_test를 one hot encoding으로 변형한다
def one_hot_encoding(x):
    x = np.eye(10)[x] # 출력층의 개수 10에 맞게 numpy 단위행렬 기능을 사용하여 one hot encoding
    return x


class TwoLayerNet:
    # 가중치와 편향을 필요에 맞게 생성한다
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {} # 필요한 딕셔너리를 미리 만들어 두기
        # 가중치와 편향을 입력층, 은닉층, 출력층 개수에 맞게 랜덤하게 생성
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    # x_train의 각 입력층들을 통해 정답을 예측한다
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    # 정답 예측의 정확도를 계산한다
    def accuracy(self, x, t): # x는 자료에서 예측된 값 t는 정답으로 주어진 값
        y = self.predict(x) # x_train의 각 자료들마다 클래스별 확률을 구해서 y에 저장
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        #print(dy)
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        #print(grads['b2'])
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        return grads


filename = 'SimpleNetData2.csv'
x_train, y_train, x_test, y_test= load_csv(filename) # 지정된 csv 파일을 열어 x_train과 y_train, x_test와 y_test를 만들기
y_train = one_hot_encoding(y_train) # y_train을 one hot encoding으로 변형
y_test = one_hot_encoding(y_test) # y_test를 one hot encoding으로 변형

network = TwoLayerNet(input_size=200, hidden_size=50, output_size=10)



# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.5

train_loss_list = []
train_acc_list = []
test_acc_list = []


# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)


for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    # 기울기 계산
    grad = network.gradient(x_batch, y_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)
    

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
