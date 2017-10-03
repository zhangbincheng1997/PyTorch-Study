# -*- encoding: utf-8 -*-

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# reproducible
torch.manual_seed(1)

# download it if you don't have it
PATH_MNIST = './mnist'
DOWNLOAD_MNIST = True

# hyper parameters
EPOCH = 10
STEP = 100
BATCH_SIZE = 64
LR = 0.005

# only test 10
N_TEST_IMG = 10

# train datasets 60000 (1, 28, 28)
train_data = torchvision.datasets.MNIST(
            root=PATH_MNIST, train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)    # compress to 3 features
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()        # compress to range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
print(autoencoder)

optimizer = optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# init figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()

# show figure
view_data = Variable(train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor) / 255)
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
    a[0][i].set_xticks(());
    a[0][i].set_yticks(());

for epoch in range(EPOCH):
    running_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, 28*28))   # batch y, shape (batch, 28*28)
        b_label = Variable(y)               # batch label

        # forward + backward
        encoded, decoded = autoencoder(b_x)
        optimizer.zero_grad()
        loss = loss_func(decoded, b_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        if (step+1) % STEP == 0:
            print('Epoch: [%d, %d] | Loss: %.4f' % (epoch+1, step+1, running_loss / STEP))
            running_loss = 0.0
            # plot decoded image
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw();
            plt.pause(0.05)

plt.ioff()
plt.show()
