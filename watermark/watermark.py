import os
import math
import random
import glob

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image

from datasets.dataloader import get_dataloader, CustomDataset
from PIL import Image, ImageFont, ImageDraw


default_texts = ['THE', 'ATTACKER', 'DOES', 'NOT', 'KNOW', 'TRIGGER', 'PATTERN', 'EITHER', 'COLOR', 'OR', 'WORD']
default_colors = ['black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'darkgray', 'orange', 'purple', 'brown', 'pink']

class TriggerSet(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, num_classes, transform=None, target=None):
        self.dataset = Subset(dataset, indices)  
        self.classes = num_classes - 1  
        self.transform = transform
        self.target = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, truth = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        
        if self.target is None:
            label = random.randint(0, self.classes)
            while label == truth:
                label = random.randint(0, self.classes)
        else:
            label = self.target
        
        return data, label
    

"""
    Content-based backdoor
"""
class EmbedText(object):
    def __init__(self, text, pos, color):
        self.text = text
        self.pos = pos
        self.color = color

    def __call__(self, tensor):
        img = transforms.ToPILImage()(tensor)
        draw = ImageDraw.Draw(img)
        font_path = '/path/to/font/sans_serif.ttf'
        font = ImageFont.truetype(font_path, 10)
        draw.text(self.pos, self.text, fill=self.color, font=font)
        tensor = transforms.ToTensor()(img)
        return tensor
    
class EmbedRandomText(object):
    def __init__(self, texts, colors):
        self.texts = texts
        self.colors = colors

    def __call__(self, tensor):
        img = transforms.ToPILImage()(tensor)
        draw = ImageDraw.Draw(img)
        font_path = '/path/to/font/sans_serif.ttf'
        font = ImageFont.truetype(font_path, 10)

        text = random.choice(self.texts)
        color = random.choice(self.colors)
        pos = (random.randint(0, 20), random.randint(0, 20))

        draw.text(pos, text, fill=color, font=font)
        tensor = transforms.ToTensor()(img)
        return tensor

'''
    Noise-based backdoor
'''
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
'''
    Utils for unrelated-based backdoor
'''
def to_3channels(img):  
    return img.repeat(3, 1, 1) 


'''
    Strong augmentation for MemEnc, 
    intentionally enhancing the robustness of members.
'''
def get_member_transform(dataset):
    if dataset in ['cifar10', 'cifar100']:
        member_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=45, translate=(0.4, 0.4), scale=(0.5, 1.5)),
            transforms.ToTensor(),
        ])

    else:
        member_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.4, 0.4)),
            transforms.ToTensor(),
        ])

    return member_transform


'''
    Out-of-Distribution data for MisFT
'''
class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, rootfile, transform=None):
        self.filenames = glob.glob(os.path.join(rootfile, 'tiny-imagenet-200/val/images/*.JPEG'))
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(os.path.join(rootfile, 'tiny-imagenet-200/val/val_annotations.txt'), 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path)
          image = image.repeat(3, 1, 1)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_ood_dataset(rootfile, num):
    ood_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    id_dict = {}
    for i, line in enumerate(open(os.path.join(rootfile, 'tiny-imagenet-200/wnids.txt'), 'r')):
        id_dict[line.replace('\n', '')] = i

    ood_set = TestTinyImageNetDataset(id=id_dict, rootfile=rootfile, transform=ood_transform)
    ood_set, _ = torch.utils.data.random_split(ood_set, [num, len(ood_set) - num])
    return ood_set


class Triggers(nn.Module):
    def __init__(self, dataset, data_path, carrier_indices, wm_type, wm_num, attack=False, target=None):
        super().__init__()
        
        trainset, testset, _, _, num_classes = get_dataloader(dataset, batch_size=100, augment=False, data_path=data_path)

        if wm_type == 'content':
            if attack:
                wm_transform = EmbedRandomText(texts=default_texts, colors=default_colors)
                self.wm_dataset = TriggerSet(trainset, carrier_indices, num_classes, wm_transform, target=target)
            else:
                wm_transform = EmbedText('TEST', (0, 0), 'white')
                self.wm_dataset = TriggerSet(trainset, carrier_indices, num_classes, wm_transform, target=target)

        elif wm_type == 'noise':
            wm_transform = AddGaussianNoise(mean=0.0, std=0.15)
            self.wm_dataset = TriggerSet(trainset, carrier_indices, num_classes, wm_transform, target=target)

        elif wm_type == 'unrelated':
            # unrelated dataset: MNIST
            mnist_transforms = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(to_3channels),
            ])
            mnist_dataset = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=mnist_transforms)
            indices = random.sample(list(range(len(mnist_dataset))), wm_num) 
            self.wm_dataset = TriggerSet(mnist_dataset, indices, num_classes, target=target)

        elif wm_type == 'normal':
            self.wm_dataset = TriggerSet(trainset, carrier_indices, num_classes, target=target)

        else:   # wm_type == 'MemEnc'
            member_transform = get_member_transform(dataset)
            self.wm_dataset = CustomDataset(trainset, carrier_indices, member_transform)

        wm_batch_size = 100 if wm_num > 100 else wm_num
        self.wm_loader = torch.utils.data.DataLoader(self.wm_dataset, batch_size=wm_batch_size, shuffle=False, num_workers=4)

        # fix the pattern
        self.images = torch.cat([i for i, _ in self.wm_loader], dim=0)
        self.targets = torch.cat([t for _, t in self.wm_loader], dim=0)