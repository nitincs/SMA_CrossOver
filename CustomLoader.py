import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import cv2


class CustomDataset(Dataset):
    def __init__(self, train):
        y1 = glob.glob('img/y1/*.png', recursive=True)
        y2 = glob.glob('img/y2/*.png', recursive=True)
        y3 = glob.glob('img/y3/*.png', recursive=True)
        y4 = glob.glob('img/y4/*.png', recursive=True)

        self.data = []

        if train:
            for x in y1[:600]:
                self.data.append((x, 1))

            for x in y2[:600]:
                self.data.append((x, 2))

            for x in y3[:600]:
                self.data.append((x, 3))

            for x in y4[:600]:
                self.data.append((x, 4))

        else:
            for x in y1[600:700]:
                self.data.append((x, 1))

            for x in y2[600:700]:
                self.data.append((x, 2))

            for x in y3[600:700]:
                self.data.append((x, 3))

            for x in y4[600:700]:
                self.data.append((x, 4))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.data[idx][1]
        img = self.data[idx][0]

        img = Image.open(img)
        import torchvision.transforms as transforms

        t = transforms.ToTensor()
        img = t(img)
        return label, img
