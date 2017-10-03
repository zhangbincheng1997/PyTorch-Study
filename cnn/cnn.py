# -*- encoding: utf-8 -*-

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# reproducible
torch.manual_seed(1)

# download it if you don't have it
PATH_CIFAR = './data'
DOWNLOAD_CIFAR = True

# hyper parameters
EPOCH = 2
STEP = 2000
BATCH_SIZE = 4
NUM_WORKERS = 2
LR = 0.001
MOMENTUM = 0.9

# transforms.ToTensor(): channel = channel / 255
# transforms.Normalize(mean, std): channel = (channel - mean) / std
# [0, 255] -> [0, 1] -> [-1, 1]
transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
# train datasets 50000 (3, 32, 32)
trainset = torchvision.datasets.CIFAR10(
            root=PATH_CIFAR, train=True, download=DOWNLOAD_CIFAR, transform=transform)
trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# test datasets 10000 (3, 32, 32)
testset = torchvision.datasets.CIFAR10(
            root=PATH_CIFAR, train=False, download=DOWNLOAD_CIFAR, transform=transform)
testloader = torch.utils.data.DataLoader(
            testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# dataset classes 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
        self.conv1 = nn.Conv2d(3, 6, 5)
        # MaxPool2d(size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        # Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Linear (400 -> 120)
        self.fc1 = nn.Linear(16*5*5, 120)
        # Linear (120 -> 84)
        self.fc2 = nn.Linear(120, 84)
        # Linear (84 -> 10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            # input: 3 RGB image
            # output: 6 feture map
            # kernel: 5*5
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            # kernel: 2*2
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            # input: 6 feature map
            # output: 16 feature map
            # kernel: 5*5
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            # kernel: 2*2
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            # 16*5*5 -> 120
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            # 120 -> 84
            nn.Linear(120, 84),
            nn.ReLU(),
            # 84 -> 10
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # view(dimensions, num_flat_features)
        # -1 mean automatic calculation
        x = x.view(-1, 16*5*5)
        x = self.fc(x)
        return x

net = Net()
print(net)

# Stochastic Gradient Descent
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
# Cross Entropy Loss
loss_func = nn.CrossEntropyLoss()

print('Start Training!!!')
for epoch in range(EPOCH):
    running_loss = 0.0
    for step, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        
        # forward + backward
        optimizer.zero_grad()
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        if (step+1) % STEP == 0:
            print('Epoch: [%d, %d] | Loss: %.4f' % (epoch+1, step+1, running_loss / STEP))
            running_loss = 0.0
print('End Training!!!')

# --------------------

correct = 0
total = 0
class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))
for data in testloader:
    images, labels = data
    output = net(Variable(images))
    predicted = torch.max(output.data, 1)[1]
    total += labels.size(0) # total
    correct += (predicted == labels).sum() # correct
    c = (predicted == labels).squeeze()
    for i in range(BATCH_SIZE): # batch_size
        label = labels[i]
        class_correct[label] += c[i] # class_correct
        class_total[label] += 1 # class_total
for i in range(10):
    print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
