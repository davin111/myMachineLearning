#studying: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN_3ch_Net(nn.Module):

	def __init__(self):
		super(CNN_3ch_Net, self).__init__()

		# 4 = mini_batch_size, 3 = image_channel
		# input C = 4*3*32*32, filter(kernel) 6*3*5*5, output 4*6*28*28
		self.conv1 = nn.Conv2d(3, 6, 5)
		# input 4*6*14*14, filter 16*6*5*5, output 4*16*10*10
		self.conv2 = nn.Conv2d(6, 16, 5)
		
		self.pool = nn.MaxPool2d(2, 2)

		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		#max_pooling
		
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
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
