import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def load(dataset='mnist', batch_size=64):
    if dataset.lower() == 'mnist':
        train_dataset = torchvision.datasets.MNIST('./data/MNIST', train=True, transform=
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            # transforms.Normalize([0.1307,], [0.3015,])
        ]))
        test_dataset = torchvision.datasets.MNIST('./data/MNIST', train=False, transform=
        transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.1307,], [0.3015,])
        ]))

    elif dataset.lower() == 'fashionmnist':
        train_dataset = torchvision.datasets.FashionMNIST('./data/FashionMNIST', train=True, transform=
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            # transforms.Normalize([0.2868,], [0.3201,])]))
        ]))
        test_dataset = torchvision.datasets.FashionMNIST('./data/FashionMNIST', train=False, transform=
        transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.2868,], [0.3201,])]))
        ]))

    elif dataset.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('./data/CIFAR10', train=True, transform=
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]))
        ]))
        test_dataset = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, transform=
        transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]))
        ]))

    elif dataset.lower() == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100('./data/CIFAR100', train=True, transform=
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            # transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023])]))
        ]))
        test_dataset = torchvision.datasets.CIFAR100('./data/CIFAR100', train=False, transform=
        transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023])]))
        ]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('Train size:', len(train_loader))
    print('Test size:', len(test_loader))

    return train_loader, test_loader
