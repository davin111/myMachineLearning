#필요한 모듈들을 불러오기
import sys, os
sys.path.append(os.pardir)
from csv import reader
import numpy as np


# 사용할 함수들 미리 def로 정의해두기

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 소프트맥스 함수
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x) # overflow 대책
    return np.exp(x) / np.sum(np.exp(x))


# Load a CSV file
def load_csv(filename):
    x_train = list() #필요한 리스트를 미리 만들어두기
    y_train = list() #필요한 리스트를 미리 만들어두기
    with open(filename, 'r') as file: #지정된 파일을 열어서 동작
        csv_reader = reader(file)
        for row in csv_reader: #주어진 csv의 각 행에 대해서
            x_train.append(list(map(float, row[:12]))) #마지막 열을 빼고 모든 내용을 각각 float 자료형으로 바꾸어 리스트로 만들고 x_train에 붙이기
            y_train.append(int(row[-1])) #마지막 열의 내용을 int 자료형으로 바꾸어 y_train에 붙이기

    return x_train, y_train


# y_train을 one hot encoding으로 변형한다
def one_hot_encoding(x):
    x = np.eye(10)[x] #출력층의 개수 10에 맞게 numpy 단위행렬 기능을 사용하여 one hot encoding
    return x


# 가중치와 편향을 필요에 맞게 생성한다
def init_network(input_size, hidden_size, output_size):
    
    network = {} #필요한 딕셔너리를 미리 만들어 두기
    weight_int_std = 0.01
    #가중치와 편향을 입력층, 은닉층, 출력층 개수에 맞게 랜덤하게 생성
    network['W1'] = weight_int_std * np.random.randn(input_size, hidden_size)
    network['b1'] = np.zeros(hidden_size)
    network['W2'] = weight_int_std * np.random.randn(hidden_size, hidden_size)
    network['b2'] = np.zeros(hidden_size)
    network['W3'] = weight_int_std * np.random.randn(hidden_size, output_size)
    network['b3'] = np.zeros(output_size)
    
    return network


# x_train의 각 입력층들을 통해 정답을 예측한다
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3) #출력층에서는 소프트맥스 함수로 클래스별 확률을 구함

    return y


# 정답 예측의 정확도를 계산한다
def accuracy(x, t):      #x는 자료에서 예측된 값 t는 정답으로 주어진 값
    y = predict(network,x) #x_train의 각 자료들마다 클래스별 확률을 구해서 y에 저장
        
    accuracy = 0 #정확도를 0으로 초기화

    for i in range(len(x)): #x_train의 자료 5000개 하나씩에 대해서
        if np.argmax(y[i]) == np.argmax(t[i]): #가장 확률이 높은 클래스가 정답과 같다면
            accuracy += 1 #정확도에 1을 더하기

    accuracy = accuracy/len(x) #정확도를 x_train의 자료 개수로 나누기

    return accuracy



filename = 'SimpleNetData.csv'
x_train, y_train = load_csv(filename) #지정된 csv 파일을 열어 x_train, y_train을 만들기
y_train = one_hot_encoding(y_train) #y_train을 one hot encoding으로 변형

network = init_network(len(x_train[1]), 100, 10) #가중치와 편향을 입력층 12개, 은닉층 100개, 출력층 10개에 맞게 생성

accuracy = accuracy(x_train, y_train) #정답 예측의 정확도를 y_train과 비교하여 계산

print(accuracy) #정확도를 출력
