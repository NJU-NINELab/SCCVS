import os.path
import sys
sys.path.append("yolov10")
# We just upload the key files of yolov10 for detection, so you need to download the full code of yolov10 to run this code successfully. Download from https://github.com/ultralytics/ultralytics,
sys.path.append("FastSAM")
# We just upload the key files of FastSAM for detection, so you need to download the full code of FastSAM to run this code successfully. Download from https://github.com/CASIA-IVA-Lab/FastSAM.git
from yolov10.Detection import GetLocation
from FastSAM.segment import segment

def spiltframes(source_path):
    for img in os.listdir(source_path):
        img_path = os.path.join(source_path,img)
        # get location via yolov10
        boxes = GetLocation(img_path)
        print(boxes)
        # segment based on location
        ann = segment(img_path,boxes)
        print(ann)


if __name__ == '__main__':
    source = r"D:\pythonProject\PythonProj\sd_server\SCCVS\OSMS\yolov10\ultralytics\assets"
    spiltframes(source)


