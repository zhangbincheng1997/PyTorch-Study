import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision

from vgg import *
from load import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            train_loss /= 100

            print('Train Epoch: %d [%d/%d (%.4f%%)]\tLoss: %.4f' % (
                epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                100. * (batch_idx + 1) * len(image) / len(train_loader.dataset), train_loss))
            train_loss = 0


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    total_true = 0
    total_loss = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)

            output = model(image)
            loss = criterion(output, label)

            pred = torch.max(output, 1)[1]  # get the index of the max log-probability
            total_true += (pred.view(label.size()).data == label.data).sum().item()
            total_loss += loss.item()

    accuracy = total_true / len(test_loader.dataset)
    loss = total_loss / len(test_loader.dataset)
    print('\nTest Epoch: %d ====> Accuracy: [%d/%d (%.4f%%)]\tAverage loss: %.4f\n' % (
        epoch, total_true, len(test_loader.dataset), 100. * accuracy, loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST FashionMNIST CIFAR10 CIFAR100')
    parser.add_argument('--dataset', type=str, default='mnist', help='enter the dataset (default: mnist)')
    parser.add_argument('--batch', type=int, default=64, help='batch (default: 64)')
    parser.add_argument('--epoch', type=int, default=20, help='epoch (default: 20)')
    args = parser.parse_args()

    train_loader, test_loader = load(dataset=args.dataset, batch_size=args.batch)

    if args.dataset.lower() == 'mnist':
        model = vgg16(channels=1, num_classes=10).to(device)
    elif args.dataset.lower() == 'fashionmnist':
        model = vgg16(channels=1, num_classes=10).to(device)
    elif args.dataset.lower() == 'cifar10':
        model = vgg16(channels=3, num_classes=10).to(device)
    elif args.dataset.lower() == 'cifar100':
        model = vgg16(channels=3, num_classes=100).to(device)
    else:
        print('[MNIST FashionMNIST CIFAR10 CIFAR100]')
        exit(0)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epoch + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader, criterion, epoch)
