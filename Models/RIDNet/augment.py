import random

import torch
from torchvision.transforms import functional as F
from torchvision import transforms


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img



class ToTensor(object):
    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
        noisy = F.to_tensor(noisy)
        clean = F.to_tensor(clean)
        # noisy = torch.from_numpy(noisy)
        # clean = torch.from_numpy(clean)
        #
        # noisy = torch.unsqueeze(noisy, dim=0)
        # clean = torch.unsqueeze(clean, dim=0)

        data = {'noisy': noisy, 'clean': clean}

        return data


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
        noisy = pad_if_smaller(noisy, size=self.size)
        clean = pad_if_smaller(clean, size=self.size)
        crop_params = transforms.RandomCrop.get_params(clean, (self.size, self.size))
        noisy = F.crop(noisy, *crop_params)
        clean = F.crop(clean, *crop_params)
        data = {'noisy': noisy, 'clean': clean}
        return data


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
        if random.random() < self.p:
            noisy = F.hflip(noisy)
            clean = F.hflip(clean)
            data = {'noisy': noisy, 'clean': clean}
        return data


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
        if random.random() < self.p:
            noisy = F.vflip(noisy)
            clean = F.vflip(clean)
            data = {'noisy': noisy, 'clean': clean}
        return data


class RandomRotation(object):
    def __init__(self, angles=None):
        if angles is None:
            angles = [0, 90, 180, 270]
        self.angles = angles

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
        angle = random.choice(self.angles)
        noisy = F.rotate(noisy, angle)
        clean = F.rotate(clean, angle)
        data = {'noisy': noisy, 'clean': clean}
        return data

