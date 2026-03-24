import os.path
import cv2
import numpy as np
import json
from ultralytics import YOLOv10
import torch
torch.backends.cudnn.enabled = False

import sys

detmodel = YOLOv10('yolov10n.pt')
total_params = sum(p.numel() for p in detmodel.parameters())
print(total_params)
detmodel.model.fuse = lambda verbose=False: detmodel.model  # 禁用 fuse 操作


def GetLocation(img_path):
    results = detmodel.predict(img_path,save=True)
    # results = detmodel.predict(img_path, save=True, fuse=False, device='cuda:0')

    for r in results:
        boxes = r.boxes.xyxy
        break
    boxes = boxes.cpu().numpy().tolist()
    save_path = img_path.replace(".jpg",".json")
    with open(save_path,"w",encoding="utf-8")as f:
        f.write(json.dumps(boxes,indent=4,ensure_ascii=False))
    print(boxes)


import torch
if __name__ == '__main__':
    # source = r"D:\pythonProject\PythonProj\SCCVS\baselines\input3"
    source = "/workspace/PythonProj/SCCVS/baselines/input2/1"
    for img in os.listdir(source):
        if img.endswith(".jpg"):
            img_path = os.path.join(source,img)
            GetLocation(img_path)




