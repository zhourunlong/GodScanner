import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
from PIL import ImageFont, ImageDraw, ImageOps
from random import randint
import matplotlib.font_manager as fm
import pandas as pd
from IPython import display

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class DE_GAN_dataset(Dataset):

    def __init__(self, file_names):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.originals, self.GTs = [], []
        for fn in file_names:
            original = np.asarray(Image.open(os.path.join("data/original", fn)).convert("L"))
            GT = np.asarray(Image.open(os.path.join("data/GT", fn)).convert("L"))
            original = np.pad(original, ((64, 64), (64, 64)))
            GT = np.pad(GT, ((64, 64), (64, 64)))

            h, w = GT.shape[: 2]

            for bh in range(0, h - 255, 128):
                for bw in range(0, w - 255, 128):
                    original_patch = original[bh : bh + 256, bw : bw + 256]
                    GT_patch = GT[bh : bh + 256, bw : bw + 256]
                    
                    original_patch = Image.fromarray(original_patch.astype(np.uint8))
                    GT_patch = Image.fromarray(GT_patch.astype(np.uint8))

                    original_patch = self.transform(original_patch)
                    GT_patch = self.transform(GT_patch)

                    self.originals.append(original_patch)
                    self.GTs.append(GT_patch)
        
    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        return self.originals[idx], self.GTs[idx]

def build_dataset():
    all_files = os.listdir("data/original")

    tot_num = len(all_files)
    valid_num = int(tot_num * 0.1)
    train_num = tot_num - valid_num

    valid_idx = np.random.choice(tot_num, valid_num, replace=False)
    
    valid_fn, train_fn = [], []

    for i in range(tot_num):
        if i in valid_idx:
            valid_fn.append(all_files[i])
        else:
            train_fn.append(all_files[i])
    
    return DE_GAN_dataset(train_fn), DE_GAN_dataset(valid_fn)
