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

def save():
    # create
    net1 = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    optimizer = optim.SGD(net1.parameters(), lr=0.5)
    loss_func = nn.MSELoss()
    
    for i in range(100):
        output = net1(x)
        optimizer.zero_grad()
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)

    # way to save all
    torch.save(net1, 'net.pkl')
    # way to save params
    torch.save(net1.state_dict(), 'net_params.pkl')

def restore_net():
    net2 = torch.load('net.pkl')
    output = net2(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)

def restore_params():
    net3 = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    net3.load_state_dict(torch.load('net_params.pkl'))
    output = net3(x)

    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
    plt.show()

# save net1
save()
# restore net2
restore_net()
# restore net3
restore_params()
