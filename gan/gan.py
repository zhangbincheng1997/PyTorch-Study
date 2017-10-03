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
np.random.seed(1)

# hyper parameters
NUM = 10000
STEP = 50
BATCH_SIZE = 64
LR_G = 0.0001   # learning rate for generate
LR_D = 0.0001   # learning rate for discriminator
N_IDEAS = 5     # this ideas for generating G

# other parameters
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# real target
def real_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)

# generator
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),        # random ideas
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)  # do from these random ideas
)

# discriminator
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128), # real or fake
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()                    # probability for output is real
)

opt_D = optim.Adam(D.parameters(), lr=LR_D)
opt_G = optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

for step in range(NUM):
    # real
    real = real_works()
    # random
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))
    # fake
    fake = G(G_ideas)
    
    # D try to increase this prob
    prob_real = D(real)
    # D try to reduce this prob
    prob_fake = D(fake)
    
    # log(D(x)) + log(1 - D(G(z)))
    D_loss = -torch.mean(torch.log(prob_real) + torch.log(1.0 - prob_fake))
    G_loss = torch.mean(torch.log(1.0 - prob_fake))
    
    opt_D.zero_grad()
    D_loss.backward(retain_variables=True) # reusing computational graph
    opt_D.step()
    
    opt_G.zero_grad()
    G_loss.backward() # None
    opt_G.step()
    
    if (step+1) % STEP == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], fake.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper painting')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower painting')
        plt.text(-0.5, 2.3, 'D accuracy=%.2f (0.50 for D to converge)' % prob_real.data.numpy().mean(), fontdict={'size': 15})
        plt.text(-0.5, 2.0, 'D score=%.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=12);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()
