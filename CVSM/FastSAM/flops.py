
from fastsam import FastSAM, FastSAMPrompt
model = FastSAM("FastSAM-s.pt")
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
# del model.model.model[-1].cv2
# del model.model.model[-1].cv3
model.fuse()