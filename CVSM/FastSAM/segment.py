import os
import json
import re
import torch
import numpy as np
from fastsam import FastSAM, FastSAMPrompt
import ast
from PIL import Image

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def segment(img_path,box_prompt):
    # load model
    input = Image.open(img_path)
    input = input.convert("RGB")
    everything_results = segmodel(
        input,
        device="cuda",
        retina_masks=True,
        imgsz=640,
        conf=0.4,
        iou=0.9
    )
    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device="cuda")
    if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
        ann = prompt_process.box_prompt(bboxes=box_prompt)
        bboxes = box_prompt
    else:
        ann = prompt_process.everything_prompt()
    # prompt_process.plot(
    #     annotations=ann,
    #     output_path="output/" + img_path.split("\\")[-1],
    #     bboxes=bboxes,
    #     points=points,
    #     point_label=point_label,
    #     withContours=False,
    #     better_quality=False,
    # )
    return ann

if __name__ == '__main__':

    segmodel = FastSAM("FastSAM-s.pt")

    # total_params = sum(p.numel() for p in segmodel.parameters())
    # print(total_params)

    img_dir = r"/workspace/PythonProj/SCCVS/baselines/input2/1"
    imgs = sorted_aphanumeric([file for file in os.listdir(img_dir) if file.endswith(".jpg")])
    dynamic_frames = []
    redundant_frames = []
    for i,img in enumerate(imgs):
        img_path = os.path.join(img_dir,img)
        print(img_path)
        json_path = img_path.replace(".jpg",".json")
        with open(json_path,"r",encoding="utf-8")as f:
            boxes = json.load(f)
        if boxes == []:
            redundant_frames.append(img_path)
            continue

        ann = segment(img_path,boxes)
        if i>0:
            if (ann.shape != temp.shape) or (np.mean(np.abs(ann-temp))) > 0.5:
                dynamic_frames.append(img_path)
            else:
                redundant_frames.append(img_path)
        temp = ann
    print("dynamic:", len(dynamic_frames))
    print("redundant:", len(redundant_frames))
    # with open("input3_SCCVS_x_sensing_res.json","w",encoding="utf-8")as f:
    #     f.write(json.dumps({"dynamic": dynamic_frames, "redundant": redundant_frames},indent=4,ensure_ascii=False))


