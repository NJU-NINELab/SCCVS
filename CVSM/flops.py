from ultralytics import YOLOv10
import time
model = YOLOv10('yolov10n.yaml')
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
del model.model.model[-1].cv2
del model.model.model[-1].cv3
start_time = time.time()
model.fuse()
print(time.time() - start_time)