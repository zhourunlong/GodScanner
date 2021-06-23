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

from de_gan.nets import GeneratorNet
from de_gan.dataset import build_dataset

from tqdm import tqdm
import argparse
import metrics

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default='de_gan/best.pt', type=str)
parser.add_argument("--pic-dir", default='de_gan/evaluate_pic', type=str, help="Input the folder containing pictures.")
parser.add_argument("--batch-size", default=50, type=int)
args = parser.parse_args()

state_dict = torch.load(args.model_dir)
model = GeneratorNet(in_channels=1, out_channels=1)
model.load_state_dict(state_dict["g"])
model.cuda()
model.eval()

def evaluate(pic_dir):
    unpadded_raw_input = pic_dir * 255.
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






import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt


from DewarpNet.models import get_model
from DewarpNet.utils import convert_state_dict
# from DewarpNet.loaders import get_loader


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwarp(img, bm):
    w,h=img.shape[0],img.shape[1]
    bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0,:,:,:]
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))
    bm0=cv2.resize(bm0,(h,w))
    bm1=cv2.resize(bm1,(h,w))
    bm=np.stack([bm0,bm1],axis=-1)
    bm=np.expand_dims(bm,0)
    bm=torch.from_numpy(bm).double()

    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()

    res = F.grid_sample(input=img, grid=bm)
    res = res[0].numpy().transpose((1, 2, 0))

    return res


def test(args,img_path,fname):
    wc_model_file_name = os.path.split(args.wc_model_path)[1]
    wc_model_name = wc_model_file_name[:wc_model_file_name.find('_')]

    bm_model_file_name = os.path.split(args.bm_model_path)[1]
    bm_model_name = bm_model_file_name[:bm_model_file_name.find('_')]

    wc_n_classes = 3
    bm_n_classes = 2

    wc_img_size=(256,256)
    bm_img_size=(128,128)

    # Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg0 = cv2.imread(img_path)
    imgorg = cv2.cvtColor(imgorg0, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, wc_img_size)
    img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Predict
    htan = nn.Hardtanh(0,1.0)
    wc_model = get_model(wc_model_name, wc_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        wc_state = convert_state_dict(torch.load(args.wc_model_path, map_location='cpu')['model_state'])
    else:
        wc_state = convert_state_dict(torch.load(args.wc_model_path)['model_state'])

    wc_model.load_state_dict(wc_state)
    wc_model.eval()
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        bm_state = convert_state_dict(torch.load(args.bm_model_path, map_location='cpu')['model_state'])
    else:
        bm_state = convert_state_dict(torch.load(args.bm_model_path)['model_state'])
    bm_model.load_state_dict(bm_state)
    bm_model.eval()

    if torch.cuda.is_available():
        wc_model.cuda()
        bm_model.cuda()
        images = Variable(img.cuda())
    else:
        images = Variable(img)

    with torch.no_grad():
        wc_outputs = wc_model(images)
        pred_wc = htan(wc_outputs)
        bm_input=F.interpolate(pred_wc, bm_img_size)
        outputs_bm = bm_model(bm_input)

    # call unwarp
    uwpred=unwarp(imgorg, outputs_bm)

    if args.show:
        f1, axarr1 = plt.subplots(1, 2)
        axarr1[0].imshow(imgorg)
        axarr1[1].imshow(uwpred)
        plt.show()

    # Save the output

    outp=os.path.join(args.out_path,fname)
    tuwpred = uwpred[:, :, ::-1] * 255.
    

    cv2.imwrite(outp, tuwpred)
    # t_uwpred = np.hstack((imgorg0, uwpred))
    # cv2.imwrite(os.path.join(args.out_path2, fname),t_uwpred)
    return uwpred[:, :, ::-1]

if __name__ == '__main__':
    parser.add_argument('--wc_model_path', nargs='?', type=str, default='DewarpNet/eval/models/unetnc_doc3d.pkl',
                        help='Path to the saved wc model')
    parser.add_argument('--bm_model_path', nargs='?', type=str, default='DewarpNet/eval/models/dnetccnl_doc3d.pkl',
                        help='Path to the saved bm model')
    parser.add_argument('--img_path', nargs='?', type=str, default='DewarpNet/eval/inp/',
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default='de_gan/evaluate_pic/',
                        help='Path of the output unwarped image')
    parser.add_argument('--out_path2', nargs='?', type=str, default='DewarpNet/eval/together/',
                        help='Path of the output unwarped image')
    parser.add_argument('--gt_path', nargs='?', type=str, default='gt/',
                        help='Path of the ground-truth images')
    parser.add_argument('--show', dest='show', action='store_true',
                        help='Show the input image and output unwarped')
    parser.add_argument('--generate', action='store_true', default=False)
    parser.set_defaults(show=False)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.out_path2, exist_ok=True)

    running_psnr, running_f, running_pf, running_drd = [], [], [], []
    for fname in os.listdir(args.img_path):
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path=os.path.join(args.img_path,fname)
            uwpred = test(args,img_path,fname)
            uwpred = .2126 * uwpred[:, :, 0] + .7152 * uwpred[:, :, 1] + 0.0722 * uwpred[:, :, 2]
            # print(uwpred.shape)
            outp=os.path.join(args.out_path,fname)
            splitted = os.path.splitext(outp)
            predicted = evaluate(uwpred)
            outputImg = Image.fromarray(predicted * 255.0).convert("L")
            outputImg.save(splitted[0] + "_predicted" + splitted[1])

            if not args.generate:
                # retrieve gt from somewhere
                tex_name = fname.split('-')[1]
                print(tex_name)
                gt = cv2.imread(os.path.join(args.gt_path, tex_name + '.jpg'))
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                psnr, fmeasure, pfmeasure, drd = metrics.evaluate_metrics(predicted, gt)
                running_psnr.append(psnr)
                running_f.append(fmeasure)
                running_pf.append(pfmeasure)
                running_drd.append(drd)
    
    if not args.generate:
        print("PSNR", np.mean(running_psnr), "F", np.mean(running_f), "Pseudo F", np.mean(running_pf), "DRD", np.mean(running_drd))

'''
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
