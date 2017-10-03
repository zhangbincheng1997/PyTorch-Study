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
PATH_MNIST = './mnist'
DOWNLOAD_MNIST = True

# hyper parameters
EPOCH = 1
STEP = 50
BATCH_SIZE = 64
LR = 0.01

# train datasets 60000 (1, 28, 28)
train_data = torchvision.datasets.MNIST(
            root=PATH_MNIST, train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# test datasets 2000 (1, 28, 28) (only test 2000)
test_data = torchvision.datasets.MNIST(
            root=PATH_MNIST, train=False, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
# reshape 2000 (28, 28) value in range (0, 1)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels.numpy().squeeze()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # LSTM(28, 64, batch_first=True)
        self.rnn = nn.LSTM(
            input_size = 28,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True
        )
        # Linear (64 -> 10)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
print(rnn)

optimizer = optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    running_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        # reshape x to (batch, time_step, input_size)
        b_x = Variable(x.view(-1, 28, 28))
        b_y = Variable(y)

        # forward + backward
        output = rnn(b_x)
        optimizer.zero_grad()
        loss = loss_func(output, b_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        if (step+1) % STEP == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print('Epoch: [%d, %d] | Loss: %.4f | Accuracy: %.2f' % (epoch+1, step+1, running_loss / STEP, accuracy))
            running_loss = 0.0

# test dataset (only test 10)
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
