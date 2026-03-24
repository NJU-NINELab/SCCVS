import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import torch


def export_onnx(example_shape, opset=12):
    module = detmodel.model
    module.eval().to("cpu")   # 导出通常在 cpu 上更稳健
    dummy = torch.randn(example_shape)
    print("Exporting to ONNX:", ONNX_PATH)
    torch.onnx.export(
        module,
        dummy,
        ONNX_PATH,
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}},
        verbose=False
    )
    print("ONNX export done.")
    print("You can build TRT engine with trtexec:")
    print(f"  trtexec --onnx={ONNX_PATH} --saveEngine={TRT_ENGINE_PATH} --fp16 --workspace=1024")
    return ONNX_PATH

# ======================
# TensorRT 模型类
# ======================
class TRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        # 读取 TensorRT engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # -------- 获取输入输出的名称 --------
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # -------- 获取形状和类型 --------
        self.input_shape = list(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = list(self.engine.get_tensor_shape(self.output_name))
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        # -------- 分配显存 --------
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(self.input_dtype).itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(self.output_dtype).itemsize)
        self.stream = cuda.Stream()

        print(f"[INFO] Engine loaded: {engine_path}")
        print(f"[INFO] Input name:  {self.input_name}, shape: {self.input_shape}")
        print(f"[INFO] Output name: {self.output_name}, shape: {self.output_shape}")

    def infer(self, inputs):
        inputs = np.ascontiguousarray(inputs, dtype=np.float32)

        # 如果是动态 batch，先设置输入 shape
        self.context.set_input_shape(self.input_name, inputs.shape)

        # 重新分配 GPU buffer（动态 batch）
        self.d_input = cuda.mem_alloc(inputs.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.d_input))

        # 输出 buffer
        output_size = trt.volume(self.context.get_tensor_shape(self.output_name)) * np.dtype(self.output_dtype).itemsize
        self.d_output = cuda.mem_alloc(output_size)
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        # 执行
        cuda.memcpy_htod_async(self.d_input, inputs, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        outputs = np.empty(self.context.get_tensor_shape(self.output_name), dtype=self.output_dtype)
        cuda.memcpy_dtoh_async(outputs, self.d_output, self.stream)
        self.stream.synchronize()

        return outputs


# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
    # export ONNX
    from ultralytics import YOLOv10
    ONNX_PATH = "yolov10n.onnx"
    TRT_ENGINE_PATH = "yolov10n.trt"
    detmodel = YOLOv10('yolov10n.pt')
    IMAGE_SIZE = 640
    export_onnx(example_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE), opset=12)
    # trtexec --onnx=yolov10n.onnx --saveEngine=yolov10n.trt --fp16

    engine_path = "yolov10n.trt"

    # 初始化模型
    trt_model = TRTModel(engine_path)

    # 随机输入 (batch, 3, 640, 640)
    batch = 256
    C, H, W = 3, 640, 640
    dummy_input = np.random.rand(batch, C, H, W).astype(np.float32)
    for i in range(10):
        print("[INFO] Start inference ...")
        t1 = time.time()
        output = trt_model.infer(dummy_input)
        t2 = time.time()

        print(f"[INFO] Output shape: {output.shape}")
        print(f"[INFO] Inference done, time cost: {(t2 - t1):.3f}s")
