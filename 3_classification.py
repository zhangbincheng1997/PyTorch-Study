# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

# reproducible
torch.manual_seed(1)

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)
x, y = Variable(x), Variable(y)

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # Linear (2 -> 10)
        self.hidden = nn.Linear(n_feature, n_hidden)
        # Linear (10 -> 2)
        self.output = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.02)
loss_func = nn.CrossEntropyLoss()

plt.ion()

for i in range(100):
    output = net(x)

    optimizer.zero_grad()
    loss = loss_func(output, y)
    loss.backward()
    optimizer.step()

    if i % 2 == 0:
        plt.cla()
        # softmax
        prediction = torch.max(F.softmax(output), 1)[1]
        predict_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=predict_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(predict_y == target_y) / 200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

# stop draw
plt.ioff()
plt.show()
