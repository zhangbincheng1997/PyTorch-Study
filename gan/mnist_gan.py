import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


GPU_MODE = False
DIR = 'MNIST_GAN_results'
utils.create_dir(DIR)


# G(z)
class generator(nn.Module):
    # init
    def __init__(self, input_size=100, n_class=28*28):
        super(generator, self).__init__()
        # in_features, out_features
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.tanh(self.fc4(x))
        return x


# D(x)
class discriminator(nn.Module):
    # init
    def __init__(self, input_size=28*28):
        super(discriminator, self).__init__()
        # in_features, out_features
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.sigmoid(self.fc4(x))
        return x


# hyper parameters
BATCH_SIZE = 128
LR_G = 0.0002
LR_D = 0.0002
EPOCH = 100

# data_loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)

# network
G = generator(input_size=100, n_class=28*28)
D = discriminator(input_size=28*28)
if GPU_MODE:
    G.cuda()
    D.cuda()

# Binary Cross Entropy Loss
BCE_loss = nn.BCELoss()

# Adam Optimizer
G_optimizer = optim.Adam(G.parameters(), lr=LR_G)
D_optimizer = optim.Adam(D.parameters(), lr=LR_D)

# fixed input
fixed_z = torch.randn(5*5, 100)
if GPU_MODE:
    fixed_z = Variable(fixed_z.cuda(), volatile=True)
else:
    fixed_z = Variable(fixed_z, volatile=True)


########## train ##########
print("Training start!!!...")

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

y_real = torch.ones(BATCH_SIZE, 1)
y_fake = torch.zeros(BATCH_SIZE, 1)
if GPU_MODE:
    y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())
else:
    y_real, y_fake = Variable(y_real), Variable(y_fake)

for epoch in range(EPOCH):
    D_losses = []
    G_losses = []
    for step, (x, y) in enumerate(train_loader):
        if step == train_loader.dataset.__len__() // BATCH_SIZE:
            break

        x = x.view(-1, 28 * 28)
        z = torch.randn((BATCH_SIZE, 100))
        if GPU_MODE:
            x, z = Variable(x.cuda()), Variable(z.cuda())
        else:
            x, z = Variable(x), Variable(z)

        ###### train D #####
        D.zero_grad()

        D_real = D(x)
        D_real_loss = BCE_loss(D_real, y_real)

        G_ = G(z)
        D_fake = D(G_)
        D_fake_loss = BCE_loss(D_fake, y_fake)

        # - (log(D(x)) + log(1 - D(G(z))))
        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()
        D_losses.append(D_train_loss.data[0])

        ###### train G ######
        G.zero_grad()

        G_ = G(z)
        D_fake = D(G_)
        G_train_loss = BCE_loss(D_fake, y_real)

        G_train_loss.backward()
        G_optimizer.step()
        G_losses.append(G_train_loss.data[0])

    ########## summary ##########
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' %
          ((epoch + 1), EPOCH,
           torch.mean(torch.FloatTensor(D_losses)),
           torch.mean(torch.FloatTensor(G_losses))))

    ########## test ##########
    G.eval()  # stop train and start test
    z = torch.randn((5*5, 100))
    if GPU_MODE:
        z = Variable(z.cuda(), volatile=True)
    else:
        z = Variable(z, volatile=True)
    random_image = G(z)
    fixed_image = G(fixed_z)
    G.train()  # stop test and start train

    ########## save ##########
    p = DIR + '/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    fixed_p = DIR + '/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    utils.save_result(random_image, (epoch+1), save=True, path=p)
    utils.save_result(fixed_image, (epoch+1), save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
print("Training finish!!!...")

# save parameters
torch.save(G.state_dict(), DIR + "/generator_param.pkl")
torch.save(D.state_dict(), DIR + "/discriminator_param.pkl")

# save history
p = DIR + '/history.png'
utils.save_history(train_hist, save=True, path=p)

# save animation
prefix = DIR + '/Fixed_results/MNIST_GAN_'
p = DIR + '/animation.gif'
utils.save_animation(EPOCH, prefix=prefix, path=p)
