# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

# reproducible
torch.manual_seed(1)

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Linear (1 -> 20)
        self.hidden = nn.Linear(1, 20)
        # Linear (20 -> 1)
        self.output = nn.Linear(20, 1)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

# net
net_SGD         = Net()
net_Momentum    = Net()
net_RMSprop     = Net()
net_Adam        = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# optimizer
opt_SGD         = optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
opts = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

# loss
loss_func = nn.MSELoss()
loss_his = [[], [], [], []]

for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        
        for net, opt, his in zip(nets, opts, loss_his):
            output = net(b_x)
            opt.zero_grad()
            loss = loss_func(output, b_y)
            loss.backward()
            opt.step()
            his.append(loss.data[0])

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, his in enumerate(loss_his):
    plt.plot(his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(0, 0.2)
plt.show()
