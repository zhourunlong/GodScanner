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

from nets import GeneratorNet
from dataset import build_dataset

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default=None, type=str)
parser.add_argument("--pic-dir", default=None, type=str, help="Input the folder containing pictures.")
parser.add_argument("--batch-size", default=50, type=int)
args = parser.parse_args()

state_dict = torch.load(args.model_dir)
model = GeneratorNet(in_channels=1, out_channels=1)
model.load_state_dict(state_dict["g"])
model.cuda()
model.eval()

def evaluate(pic_dir):
    unpadded_raw_input = np.asarray(Image.open(pic_dir).convert("L"))
    h, w = unpadded_raw_input.shape[: 2]
    nh, nw = ((h - 1) // 256 + 1) * 256, ((w - 1) // 256 + 1) * 256
    ph = ((nh - h) // 2, (nh - h + 1) // 2)
    pw = ((nw - w) // 2, (nw - w + 1) // 2)
    raw_input = np.pad(unpadded_raw_input, (ph, pw))
    predicted = np.zeros((nh, nw))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    batched = torch.zeros(((nh // 256) * (nw // 256), 256, 256), device="cuda")
    cnt = 0
    for i in range(0, nh, 256):
        for j in range(0, nw, 256):
            patch = raw_input[i:i+256, j:j+256]
            patch_pic = Image.fromarray(patch.astype(np.uint8))
            batched[cnt, :, :] = transform(patch_pic)
            cnt += 1
    batched.unsqueeze_(1)

    batch_queue = []
    for i in range(0, cnt, args.batch_size):
        batch_queue.append(batched[i:i+args.batch_size, :, :, :])

    i, j = 0, 0
    with tqdm(batch_queue, desc="Evaluating") as pbar:
        for batch in pbar:
            with torch.no_grad():
                output = model(batch.float())
            output = output.squeeze(1).cpu()
            for b in range(batch.shape[0]):
                predicted[i:i+256, j:j+256] = output[b, :, :]
                j += 256
                if j == nw:
                    j = 0
                    i += 256

    return predicted[ph[0]:nh-ph[1], pw[0]:nw-pw[1]]

all_files = os.listdir(args.pic_dir)

for fn in all_files:
    splitted = os.path.splitext(fn)
    if splitted[0].endswith("_predicted"):
        continue
    print(fn)
    predicted = evaluate(os.path.join(args.pic_dir, fn))
    outputImg = Image.fromarray(predicted * 255.0).convert("L")
    outputImg.save(os.path.join(args.pic_dir, splitted[0] + "_predicted" + splitted[1]))

    '''
    _, ax = plt.subplots(2, 1, figsize=(nh // 100, nw // 100))
    ax[0].imshow(raw_input, cmap='gray')
    ax[0].title.set_text('Degrade Image')
    ax[1].imshow(predicted, cmap='gray')
    ax[1].title.set_text('Predicted Image')
    plt.savefig("evaluation.png")

    plt.close()

    plt.figure(figsize = (nh // 100, nw // 100))
    fig = plt.imshow(predicted, cmap='gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig("predicted.png")
    '''
