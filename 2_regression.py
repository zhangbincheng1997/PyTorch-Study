# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

# reproducible
torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # Linear (1 -> 10)
        self.hidden = nn.Linear(n_feature, n_hidden)
        # Linear (10 -> 1)
        self.output = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.5)
loss_func = nn.MSELoss()

plt.ion()

for i in range(100):
    output = net(x)

    optimizer.zero_grad()
    loss = loss_func(output, y)
    loss.backward()
    optimizer.step()

    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
