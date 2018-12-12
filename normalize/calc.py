import argparse
import torch
import torchvision
import torchvision.transforms as transforms


def get_mean_std(dataset, download=False):
    if dataset.lower() == 'mnist':  # transforms.Normalize([0.1307,], [0.3015,])
        # 28*28
        dataset = torchvision.datasets.MNIST('./data/MNIST', train=True, download=download,
                                             transform=transforms.ToTensor())
        channels = 1
    elif dataset.lower() == 'fashionmnist':  # transforms.Normalize([0.2868,], [0.3201,])
        # 28*28
        dataset = torchvision.datasets.FashionMNIST('./data/FashionMNIST', train=download, download=True,
                                                    transform=transforms.ToTensor())
        channels = 1
    elif dataset.lower() == 'cifar10':  # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # 32*32
        dataset = torchvision.datasets.CIFAR10('./data/CIFAR10', train=True, download=download,
                                               transform=transforms.ToTensor())
        channels = 3
    elif dataset.lower() == 'cifar100':  # transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023])
        # 32*32
        dataset = torchvision.datasets.CIFAR100('./data/CIFAR100', train=True, download=download,
                                                transform=transforms.ToTensor())
        channels = 3
    else:
        print('[MNIST FashionMNIST CIFAR10 CIFAR100]')
        exit(0)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    mean, std = torch.zeros(channels), torch.zeros(channels)
    for image, _ in loader:
        for i in range(channels):
            mean[i] += image[:, i, :, :].mean()
            std[i] += image[:, i, :, :].std()

    mean.div_(len(loader))
    std.div_(len(loader))

    return mean, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST FashionMNIST CIFAR10 CIFAR100')
    parser.add_argument('--dataset', type=str, default='mnist', help='enter the dataset (default: mnist)')
    args = parser.parse_args()

    print(get_mean_std(args.dataset))
