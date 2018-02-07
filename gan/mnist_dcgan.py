import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn import DataParallel # multi-gpu
from time import time


GPU_MODE = True
DIR = 'MNIST_DCGAN_results'
utils.create_dir(DIR)


# G(z)
class generator(nn.Module):
    # init
    def __init__(self, nc=1, ngf=64, nz=100):
        super(generator, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.deconv1 = nn.ConvTranspose2d(nz, ngf*8, kernel_size=4, stride=1, padding=0)
        self.deconv1_bn = nn.BatchNorm2d(ngf*8)
        self.deconv2 = nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1)
        self.deconv2_bn = nn.BatchNorm2d(ngf*4)
        self.deconv3 = nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1)
        self.deconv3_bn = nn.BatchNorm2d(ngf*2)
        self.deconv4 = nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1)
        self.deconv4_bn = nn.BatchNorm2d(ngf)
        self.deconv5 = nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1)

    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    # forward
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x


# D(x)
class discriminator(nn.Module):
    # init
    def __init__(self, nc=1, ndf=64):
        super(discriminator, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv1 = nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(ndf*2)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(ndf*8)
        self.conv5 = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0)

    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    # forward
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))
        return x


# hyper parameters
BATCH_SIZE = 128
LR_G = 0.0002
LR_D = 0.0002
BETAS_G = (0.5, 0.999)
BETAS_D = (0.5, 0.999)
EPOCH = 20

# data_loader
img_size = 64
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)

# network
G = generator(nc=1, ngf=64, nz=100)
D = discriminator(nc=1, ndf=64)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G = DataParallel(G)
D = DataParallel(D)
if GPU_MODE:
    G.cuda()
    D.cuda()

print(G)
print(D)

# Binary Cross Entropy Loss
BCE_loss = nn.BCELoss()

# Adam Optimizer
G_optimizer = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS_G)
D_optimizer = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS_D)

# fixed input
fixed_z = torch.randn(5*5, 100).view(-1, 100, 1, 1)
if GPU_MODE:
    fixed_z = Variable(fixed_z.cuda(), volatile=True)
else:
    fixed_z = Variable(fixed_z, volatile=True)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_times'] = []

y_real = torch.ones(BATCH_SIZE)
y_fake = torch.zeros(BATCH_SIZE)
if GPU_MODE:
    y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())
else:
    y_real, y_fake = Variable(y_real), Variable(y_fake)

########## train ##########
print("Training start!!!...")
start_time = time()

for epoch in range(EPOCH):
    D_losses = []
    G_losses = []
    epoch_start_time = time()
    for step, (x, y) in enumerate(train_loader):
        if step == train_loader.dataset.__len__() // BATCH_SIZE:
            break

        z = torch.randn((BATCH_SIZE, 100)).view(-1, 100, 1, 1)
        if GPU_MODE:
            x, z = Variable(x.cuda()), Variable(z.cuda())
        else:
            x, z = Variable(x), Variable(z)

        ###### train D #####
        D.zero_grad()

        D_real = D(x).squeeze()  # squeeze
        D_real_loss = BCE_loss(D_real, y_real)

        G_ = G(z)
        D_fake = D(G_).squeeze()  # squeeze
        D_fake_loss = BCE_loss(D_fake, y_fake)

        # - (log(D(x)) + log(1 - D(G(z))))
        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()
        D_losses.append(D_train_loss.data[0])

        ###### train G ######
        G.zero_grad()

        G_ = G(z)
        D_fake = D(G_).squeeze()  # squeeze
        G_train_loss = BCE_loss(D_fake, y_real)

        G_train_loss.backward()
        G_optimizer.step()
        G_losses.append(G_train_loss.data[0])

    epoch_end_time = time()
    per_epoch_time = epoch_end_time - epoch_start_time
    ########## summary ##########
    print('[%d/%d] - time: %.2f, loss_d: %.3f, loss_g: %.3f' %
          ((epoch + 1), EPOCH, per_epoch_time,
           torch.mean(torch.FloatTensor(D_losses)),
           torch.mean(torch.FloatTensor(G_losses))))

    ########## test ##########
    G.eval()  # stop train and start test
    z = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    if GPU_MODE:
        z = Variable(z.cuda(), volatile=True)
    else:
        z = Variable(z, volatile=True)
    random_image = G(z)
    fixed_image = G(fixed_z)
    G.train()  # stop test and start train

    p = DIR + '/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    fixed_p = DIR + '/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    utils.save_result(random_image, (epoch+1), save=True, path=p)
    utils.save_result(fixed_image, (epoch+1), save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_times'].append(per_epoch_time)

end_time = time()
total_time = end_time - end_time
print("Avg per epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_times'])), EPOCH, total_time))
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
