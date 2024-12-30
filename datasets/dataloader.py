import os

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_transform(dataset):
    if dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.ToTensor()

    else:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    return train_transform, test_transform


def get_dataloader(dataset='cifar10', batch_size=128, num_workers=4, augment=True, data_path='data/'):
    dataset = dataset.lower()
    n_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'caltech': 101,
    }[dataset]

    train_transform, test_transform = get_transform(dataset)

    if dataset in ['cifar10', 'cifar100']:
        DatasetClass = {'cifar10': datasets.CIFAR10,
                    'cifar100': datasets.CIFAR100,
                    }[dataset]
        
        trainset = DatasetClass(root=data_path, train=True, download=True,
                                transform=train_transform if augment else test_transform)

        testset = DatasetClass(root=data_path, train=False, download=True, transform=test_transform)

    elif dataset == 'caltech':
        caltech_directory = os.path.join(data_path, 'caltech-101')
        trainset = datasets.ImageFolder(os.path.join(caltech_directory, 'train'), 
                                        transform=train_transform if augment else test_transform)
        testset = datasets.ImageFolder(os.path.join(caltech_directory, 'test'), transform=test_transform) 

    else:
        raise NotImplementedError('Dataset is not implemented.')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=augment, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainset, testset, trainloader, testloader, n_classes


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = Subset(dataset, indices)  
        self.transform = transform  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        
        return data, target


def get_train_transform(dataset):
    if dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    return train_transform


def get_dataloader_from_idx(dataset, batch_size, trainset, testset, train_indices, transform=False, num_workers=4):
    if transform:
        train_transform = get_train_transform(dataset)
        trainset = CustomDataset(trainset, train_indices, transform=train_transform)
    else:
        trainset = CustomDataset(trainset, train_indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
