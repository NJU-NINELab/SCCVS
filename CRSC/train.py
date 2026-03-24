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
from data_utils import get_dataloader
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
def save_images(y, x_rec, save_pth):
    max = y.data.max()
    min = y.data.min()
    os.makedirs(save_pth,exist_ok=True)
    imgs_sample = (y.data - min) / (max - min)
    filename = os.path.join(save_pth, "raw.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    imgs_sample = (x_rec.data -min) / (max - min)
    filename = os.path.join(save_pth, "rec.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)


    os.makedirs(os.path.join(save_pth, "all_imgs_raw"), exist_ok=True)
    plt.figure()
    imgs_sample = (y.data - min) / (max - min)
    for i in range(5):
        img = imgs_sample[i]
        torchvision.utils.save_image(img, os.path.join(save_pth, f"all_imgs_raw/{i}.jpg"), nrow=1)
    plt.close()


    os.makedirs(os.path.join(save_pth, "all_imgs_rec"), exist_ok=True)
    plt.figure()
    imgs_sample = (x_rec.data - min) / (max - min)
    for i in range(5):
        img = imgs_sample[i]
        torchvision.utils.save_image(img, os.path.join(save_pth, f"all_imgs_rec/{i}.jpg"), nrow=1)
    plt.close()

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

# training based on KD
def KD_train(mentor_model, stu_model, dataloader, cfg):
    start = time.time()
    checkpoint_path = os.path.join(cfg.checkpoints_dir)
    os.makedirs(checkpoint_path, exist_ok=True)

    stu_model.to(cfg.device)
    mentor_model.to(cfg.device)


    # define optimizer

    optimizer_mentor = torch.optim.AdamW(mentor_model.parameters(), lr=cfg.lr,
                                             weight_decay=cfg.weight_delay)

    scheduler_mentor = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mentor, mode='min',
                                                                       factor=0.1, patience=200,
                                                                       verbose=True, threshold=0.0001,
                                                                       threshold_mode='rel',
                                                                       cooldown=0, min_lr=0, eps=1e-08)
    optimizer_stu = torch.optim.AdamW(stu_model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_delay)
    scheduler_stu = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stu, mode='min',
                                                                  factor=0.1, patience=200,
                                                                  verbose=True, threshold=0.0001,
                                                                  threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)

    # define loss function
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    kl = nn.KLDivLoss()
    # training
    train_mentor_loss = []
    train_stu_loss = []
    for epoch in range(cfg.training_epoch):
        # stu_model.train()
        mentor_model.train()
        m_epoch_loss = []
        s_epoch_loss = []
        for x in dataloader:
            optimizer_stu.zero_grad()
            optimizer_mentor.zero_grad()
            x = x.to(cfg.device)
            snr = random.randint(cfg.SNR_MIN,cfg.SNR_MAX)
            # print(snr)
            y_m, se_m, de_m = mentor_model(x,snr=snr)
            y_s, se_s, de_s = stu_model(x,snr=snr)

            # mentor loss
            l_m_task = mse(y_m,x) * 100
            # l_m_ch = mse(se_m,de_m)
            l_m = l_m_task # + l_m_ch

            # student loss
            l_s_task = mse(y_s, x) * 100
            # l_s_ch = mse(se_s,de_s)
            # KD loss
            mentor_dis = F.softmax(de_m)
            stu_dis = F.log_softmax(de_s)
            l_s_kd = kl(stu_dis,mentor_dis.detach())/l_m_task.detach()
            l_s = l_s_task + l_s_kd # + l_s_ch

            total_loss = l_s + l_m
            total_loss.backward()
            optimizer_mentor.step()
            scheduler_mentor.step(l_m)
            optimizer_stu.step()
            scheduler_stu.step(l_s)

            print(f"epoch {epoch} | mentor loss:{l_m} | student loss:{l_s}")

            m_epoch_loss.append(l_m.item())
            s_epoch_loss.append(l_s.item())
        train_mentor_loss.append(np.mean(m_epoch_loss))
        train_stu_loss.append(np.mean(s_epoch_loss))
        save_images(x, y_m, os.path.join(cfg.logs_dir,"mentor"))
        save_images(x,y_s,os.path.join(cfg.logs_dir,"student"))
        # save_weights
        torch.save(mentor_model.state_dict(),os.path.join(checkpoint_path,"mentor.pth"))
        torch.save(stu_model.state_dict(),os.path.join(checkpoint_path,"student.pth"))
        # save_images(x, y_s, cfg.logs_dir)
        records = {"stu_train_loss":train_stu_loss,"mentor_train_loss":train_mentor_loss}
        with open(os.path.join(cfg.logs_dir, "loss.json"), "w",
                  encoding="utf-8")as f:
            f.write(json.dumps(records, ensure_ascii=False, indent=4))
    print("waste time:",time.time()-start)


if __name__ == '__main__':
    # hyparametes set
    same_seeds(2048)
    config = Config()
    # prepare data
    dataloader = get_dataloader(config)
    mentor_model = SemCom_Model(CR="low",channel=config.channel)
    stu_model = SemCom_Model(CR="high",channel=config.channel)
    checkpoint = torch.load("checkpoints/mentor.pth", map_location='cpu')
    mentor_model.load_state_dict(checkpoint, strict=False)

    checkpoint = torch.load("checkpoints/student.pth", map_location='cpu')
    stu_model.load_state_dict(checkpoint, strict=False)

    KD_train(mentor_model,stu_model,dataloader,config)















