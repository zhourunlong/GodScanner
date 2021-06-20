# code to train backward mapping regression from GT world coordinates
# models are saved in checkpoints-bm/

import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from torchvision import utils
from tqdm import tqdm

from models import get_model
from loaders import get_loader
from utils import show_unwarp_tnsboard, get_lr
import recon_lossc
import pytorch_ssim
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    train = "/data2/shenrui/Downloads/doc3d/train.txt"
    val = "/data2/shenrui/Downloads/doc3d/val.txt"
    file_dir = "/data2/shenrui/Downloads/doc3d/bm"

    train_n = 0
    val_n = 0
    with open(train, "w") as f_train, open(val, "w") as f_val:
        for files in os.listdir(file_dir):
            for file in os.listdir(os.path.join(file_dir, files)):
                if (random.randint(1, 10) > 1):
                    f_train.write(os.path.join(files, file[:-4]) + '\n')
                    train_n += 1
                else:
                    f_val.write(os.path.join(files, file[:-4]) + '\n')
                    val_n += 1

    print(train_n)
    print(val_n)