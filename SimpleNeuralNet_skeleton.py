# <이 괄호 안에 들어간 내용은 힌트입니다>


# 필요한 모듈들을 불러오기
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
    x_train = list() # 필요한 리스트를 미리 만들어두기
    y_train = list() # 필요한 리스트를 미리 만들어두기
    with open(filename, 'r') as file: # 지정된 파일을 열어서 동작
        csv_reader = reader(file)
    # <직접 구현하세요 - 읽어 온 자료를 numpy 배열로 바꾸고 이 때 x_train과 y_train은 각각 float형과 int형으로!
    # How to Implement Simple Linear Regression From Scratch with Python(Slack에 링크) 참조>
    # <직접 구현하세요>
    return x_train, y_train


# y_train을 one hot encoding으로 변형한다
def one_hot_encoding(x):
    # <직접 구현하세요 - y_train 정답 값을 one hot encoding으로 변형>


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

    # <직접 구현하세요 - weight와 bias를 내적(product)한 후 sigmoid로 활성화>
    # <직접 구현하세요 - 출력층에서는 softmax 함수로 클래스별 확률을 구하기>
    return y


# 정답 예측의 정확도를 계산한다
def accuracy(x, t):      #x는 자료에서 예측된 값 t는 정답으로 주어진 값
    y = predict(network,x) #x_train의 각 자료들마다 클래스별 확률을 구해서 y에 저장
    
    # <직접 구현하세요 - 예측된 값과 one hot encoding으로 된 y_train의 값을 비교하여 정확도를 출력>

    return accuracy



filename = 'SimpleNetData.csv'
x_train, y_train = load_csv(filename) #지정된 csv 파일을 열어 x_train, y_train을 만들기
y_train = one_hot_encoding(y_train) #y_train을 one hot encoding으로 변형

network = init_network(len(x_train[1]), 100, 10) #가중치와 편향을 입력층 12개, 은닉층 100개, 출력층 10개에 맞게 생성

accuracy = accuracy(x_train, y_train) #정답 예측의 정확도를 y_train과 비교하여 계산

print(accuracy) #정확도를 출력
