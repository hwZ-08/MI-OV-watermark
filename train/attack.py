import copy

import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms


def weight_prune(model, pruning_perc):
    new_model = copy.deepcopy(model)
    if pruning_perc == 0:
        return new_model

    all_weights = np.concatenate([p.abs().data.cpu().numpy().reshape(-1) for p in new_model.parameters()
                                  if len(p.data.size()) != 1])

    threshold = np.percentile(all_weights, pruning_perc)
    for p in new_model.parameters():
        mask = p.abs() > threshold
        p.data.mul_(mask.float())
    return new_model


def re_initializer_layer(model, num_classes, device, layer=None):
    """remove the last layer and add a new one"""
    original_last_layer = model.module.linear
    indim = original_last_layer.in_features

    if layer:
        model.module.linear = layer
    else:
        model.module.linear = nn.Linear(indim, num_classes).to(device)

    return model, original_last_layer



class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
RP_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.25)),
    transforms.ToTensor(),
])

blur_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.)),
    transforms.ToTensor(),
])

def attack_transfomation(type):
    if type == 'RP':
        transformation = RP_transform
    elif type == 'noise':
        transformation = AddGaussianNoise(mean=0.0, std=0.03)
    else:
        transformation = blur_transform
    return transformation