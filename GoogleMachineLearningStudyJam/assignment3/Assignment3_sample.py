import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(5)

dataset = np.loadtxt("./SimpleNetData2.csv", delimiter=",")

x_train = dataset[:8000,0:200]
y_train = dataset[:8000,200]
x_test = dataset[8000:,0:200]
y_test = dataset[8000:,200]

model = Sequential()
model.add(Dense(50, input_dim = 200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=100)

scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
