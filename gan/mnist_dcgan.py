import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


GPU_MODE = False
DIR = 'MNIST_DCGAN_results'
utils.create_dir(DIR)


# G(z)
class generator(nn.Module):
    # init
    def __init__(self, d=128):
        super(generator, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d*1, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*1)
        self.deconv5 = nn.ConvTranspose2d(d*1, 1, 4, 2, 1)

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
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv1 = nn.Conv2d(1, d*1, 4, 2, 1)
        # self.conv1_bn = nn.BatchNorm2d(d*1)
        self.conv2 = nn.Conv2d(d*1, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 2, 0)

    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    # forward
    def forward(self, input):
        # x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
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
EPOCH = 10

# data_loader
img_size = 64
transform = transforms.Compose([
    transforms.Scale(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
if GPU_MODE:
    G.cuda()
    D.cuda()

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
