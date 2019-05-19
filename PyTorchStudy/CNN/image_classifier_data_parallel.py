# studying: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
# want to use multiple GPU!

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import CNN_3ch_nn_data_parallel as CNN
import torch.nn as nn
import torch.optim as optim


transform = transforms.Compose(
	[transforms.ToTensor(),
	 #3개 채널에 대한 평균들, 3개 채널에 대한 표준편차들
	 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
										download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))

dataiter = iter(trainloader)
images, labels = dataiter.next()
#print(type(images))
#print(images.size())
#imshow(torchvision.utils.make_grid(images))

#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


net = CNN.CNN_3ch_Net()
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	net = nn.DataParallel(net)

#if we can use GPU, use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('device: ', end = '')
print(device)
net.to(device)
#입력과 정답도 GPU로 보내야 함

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


"""
training
"""
print()
print('<training>')
for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)

		optimizer.zero_grad()

		outputs = net(inputs)
		#print("Outside: input size", inputs.size(), "output size", outputs.size())

		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 2000 == 1999: #for every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')


"""
test
"""
print()
print('<test>')
#dataiter = iter(testloader)
#images, labels = dataiter.next()

#outputs = net(images)
#_, predicted = torch.max(outputs, 1)
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#							  for j in range(4)))

correct = 0
total = 0
with torch.no_grad(): # parameter 갖는 모델이지만, test이므로 auto_grad 필요 없음
	for data in testloader:
		images, labels = data
		images, labels = images.to(device), labels.to(device)

		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1) #1차원 기준의 최대값 index -> predicted
		total += labels.size(0)
		correct += (predicted == labels).sum().item() #predicted == labels인 위치의 원소들이 1이니까 그것을 sum으로 더하고 그 값을 item으로 가져옴

print('Accurach of the netword on the 10000 test images: %d %%'
	% (100 * correct / total))


"""
analyzing result
"""
print()
print('<analyzing result>')
class_correct = list(0. for  i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	for data in testloader:
		images, labels = data
		images, labels = images.to(device), labels.to(device)	
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1

#print
for i in range(10):
	print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] /class_total[i]))
