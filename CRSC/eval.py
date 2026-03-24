############### test KDHT ##############
from SC_model import SemCom_Model
import torch
import random
from torch import nn
from matplotlib import pyplot as plt
import torchvision
import numpy as np
import os
from torch.nn import functional as F
from data_utils import get_dataloader,get_testloader
from Config import Config
import json
import time
from torchvision.transforms import transforms
torch.cuda.set_device(0)

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# show image and save
def save_images(y, x_rec, save_path):
    max = y.data.max()
    min = y.data.min()
    imgs_sample = (y.data - min) / (max - min)
    filename = save_path.replace("scene1_light","raw")
    save_dir = os.path.sep.join(os.path.split(filename)[:-1])
    os.makedirs(save_dir,exist_ok=True)
    torchvision.utils.save_image(imgs_sample, filename, nrow=1)
    print(filename)

    imgs_sample = (x_rec.data - min) / (max - min)
    filename = save_path.replace("scene1_light","rec")
    save_dir = os.path.sep.join(os.path.split(filename)[:-1])
    os.makedirs(save_dir, exist_ok=True)
    torchvision.utils.save_image(imgs_sample, filename, nrow=1)
    print(filename)

import math
from ssim import SSIM
def evaluate(x,x_):
    def psnr_loss(mse,PIXEL_MAX=1):
        if mse < 1.0e-10:
           return 100
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    ssim_loss = SSIM()
    psnr = psnr_loss(F.mse_loss(x, x_))
    ssim = ssim_loss(x,x_)
    return np.mean(psnr), np.mean(ssim)


import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_msssim import ms_ssim


def calculate_ms_ssim(image_path1, image_path2):
    # 读取图像并转换为灰度图
    image1 = Image.open(image_path1).convert('L')
    image2 = Image.open(image_path2).convert('L')

    # 将图像转换为张量并归一化
    transform = transforms.ToTensor()
    image1 = transform(image1).unsqueeze(0)  # 增加一个维度以匹配批次格式
    image2 = transform(image2).unsqueeze(0)

    # 计算MS-SSIM
    msssim_value = ms_ssim(image1, image2, data_range=1.0)

    return msssim_value.item()


def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """
    计算两个图像张量之间的峰值信噪比（PSNR）。

    参数:
    img1, img2 -- Tensor，形状 (H, W) 或 (C, H, W) 或 (N, C, H, W)，值范围 [0, max_pixel_value]
    max_pixel_value -- float，像素值的最大可能值（例如对于8位图像，最大值为255；对于归一化图像，最大值为1）

    返回:
    PSNR值 (float)
    """
    # 确保两个输入张量形状相同
    assert img1.shape == img2.shape, "两个输入张量的形状必须相同"

    # 计算均方误差 (MSE)
    mse = torch.mean((img1 - img2) ** 2)

    # 如果 MSE 为零，意味着图像完全相同，返回无穷大的 PSNR
    if mse == 0:
        return float('inf')

    # 计算 PSNR
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


# Eval
@torch.no_grad()
def Eval(mentor_model, stu_model, dataloader, cfg):

    weights = torch.load(f"checkpoints/mentor.pth", map_location="cpu")
    mentor_model.load_state_dict(weights,strict=False)
    mentor_model.eval()

    weights = torch.load(f"checkpoints/student.pth", map_location="cpu")
    stu_model.load_state_dict(weights,strict=False)
    stu_model.eval()

    stu_model.to(cfg.device)
    mentor_model.to(cfg.device)

    snr = cfg.snr
    print("snr:", snr)
    PSNR_res = []
    MS_SSIM_res = []
    start = time.time()
    for x,label,img_path in dataloader:
        x = x.to(cfg.device)
        label = label.item()
        img_path = img_path[0]
        if label==0:
            x_r, se_s, de_s = mentor_model(x,snr=snr)
        else:
            x_r, se_m, de_m = mentor_model(x, snr=snr)
        save_images(x, x_r, img_path)
        # 计算MS-SSIM
        max_val = x.data.max()
        min_val = x.data.min()
        x_norm = (x.data-min_val)/(max_val-min_val)
        x_r_norm = (x_r.data-min_val)/(max_val-min_val)
        msssim_value = ms_ssim(x_norm, x_r_norm, data_range=1.0)
        psnr_value = calculate_psnr(x_norm,x_r_norm)
        MS_SSIM_res.append(msssim_value.item())
        PSNR_res.append(psnr_value)
        break
    print("PSNR:",np.mean(PSNR_res))
    print("MS-SSIM",np.mean(MS_SSIM_res))
    print("waste time:",time.time()-start)


if __name__ == '__main__':
    # hyparametes set
    same_seeds(2048)
    config = Config()
    # SCCVS w/o osms
    # dataloader = get_testloader(r"D:\pythonProject\PythonProj\sd_server\datasets\test\testdata")
    # config.channel = "fading"
    # config.snr = 25
    # mentor_model = SemCom_Model(CR="low")
    # stu_model = SemCom_Model(CR="high")
    # Eval(mentor_model,stu_model,dataloader,config)

    # SCCVS
    dataloader = get_testloader("CS_res1l.json")
    config.channel = "AWGN"
    mentor_model = SemCom_Model(CR="low", channel=config.channel)
    stu_model = SemCom_Model(CR="high", channel=config.channel)
    for snr in [5]:
        config.snr = snr
        Eval(mentor_model, stu_model, dataloader, config)















