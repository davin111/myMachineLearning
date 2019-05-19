#studying: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		# input C = 1*1*32*32, filter(kernel) 6*1*5*5, output 1*6*28*28
		self.conv1 = nn.Conv2d(1, 6, 5)
		# input 1*6*14*14, filter 16*6*5*5, output 1*16*10*10
		self.conv2 = nn.Conv2d(6, 16, 5)
		# 16*5*5에서 5*5는 filter(kernel) 사이즈 X
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		#max_pooling
		
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		#print(x.size())
		y = F.relu(self.conv2(x))
		#print(y.size())
		x = F.max_pool2d(y, 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		#print(x.size())
		size = x.size()[1:]
		#[16, 5, 5]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


net = Net()
print(net)


params = list(net.parameters())
print(len(params))
#print(params)
for param in params:
	print(param.size())

print()

input = torch.randn(1, 1, 32, 32) #mini_batch_size, channel, H, W
out = net(input)
#1*10
print(out)

net.zero_grad()
out.backward(torch.randn(1,10))

print()


output = net(input)
target = torch.arange(1, 11)
print(target)
target = target.view(1, -1)
print(target)
target = target.type(torch.FloatTensor)
criterion = nn.MSELoss()
print(criterion)

loss = criterion(output, target)
print(loss)

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
#print('conv2.weight.grad before backward')
#print(net.conv2.weight.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
#print('conv2.weight.grad after backward')
#print(net.conv2.weight.grad)

learning_rate = 0.01
for f in net.parameters():
	f.data.sub_(f.grad.data * learning_rate)

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update
