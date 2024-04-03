import random

from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from PIL import Image


class DiffDataset(Dataset):
    def __init__(self, path, transform=None, ex=1):
        super(DiffDataset, self).__init__()
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path)] * ex)  # ex表示增广系数
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        clean = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        noisy = np.copy(clean)
        noisy = self.add_gaussian_noise(noisy)
        data = {'noisy': Image.fromarray(noisy), 'clean': Image.fromarray(clean)}
        if self.transform:
            data = self.transform(data)
        return data

    def add_gaussian_noise(self, img):
        noise_level = [15, 20, 50]
        sigma = random.choice(noise_level)
        noises = np.random.normal(scale=sigma, size=(img.shape[0], img.shape[1]))
        noises = noises.round()
        img_noise = img.astype(np.int16) + noises.astype(np.int16)
        img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)

        return img_noise
