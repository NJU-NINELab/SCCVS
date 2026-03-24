import json
import os
import torch, torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import random
from PIL import Image
from matplotlib import pyplot as plt
from Config import Config
# -------------------------------------------------------------------------------------------------------
# DATASETS
# -------------------------------------------------------------------------------------------------------


import torchvision.datasets as dset

class train_datasets(Dataset):
    def __init__(self, data):
        self.data = data
        self.img_transform = self.transform()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        img = Image.open(self.data[item]).convert('RGB')
        img = self.img_transform(img)
        return img

    def transform(self):
        imagenet_mean = np.array([0.4802, 0.4481, 0.3975])
        imagenet_std = np.array([0.2770, 0.2691, 0.2821])
        compose = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
        return transforms.Compose(compose)

def get_dataloader(cfg):
    data_path = cfg.dataset_path
    data_train = dset.ImageFolder(root=data_path).imgs
    random.shuffle(data_train)
    x_train = np.array([x[0] for x in data_train])

    train_loader = torch.utils.data.DataLoader(train_datasets(x_train), batch_size=cfg.batch_size,
                                               shuffle=True)


    return train_loader

class test_datasets(Dataset):
    def __init__(self, sensing_res):
        self.dynamic_frames = sensing_res["dynamic"]
        self.redundant_frames = sensing_res['redundant']
        self.data = self.dynamic_frames + self.redundant_frames
        self.img_transform = self.transform()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        img = Image.open(self.data[item]).convert('RGB')
        if self.data[item] in self.dynamic_frames:
            label = 0
        else:
            label = 1
        img = self.img_transform(img)
        return img, label, self.data[item]

    def transform(self):
        imagenet_mean = np.array([0.4802, 0.4481, 0.3975])
        imagenet_std = np.array([0.2770, 0.2691, 0.2821])
        compose = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
        return transforms.Compose(compose)

def get_testloader(sensing_res="CS_res1l.json"):
    if sensing_res.endswith(".json"):
        with open(sensing_res,"r",encoding="utf-8")as f:
            content = json.load(f)
    else:
        data = [os.path.join(sensing_res,img) for img in os.listdir(sensing_res)]
        content = {"dynamic":[],"redundant":data}
    data_loader = torch.utils.data.DataLoader(test_datasets(content), batch_size=1,
                                               shuffle=False)
    return data_loader

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
if __name__ == "__main__":
    same_seeds(2048)
    cfg = Config()
    # cfg.dataset = "tiny"
    dataloader = get_dataloader(cfg)
    print(dataloader.__len__())

