import ctypes
import tensorrt as trt
import os
import torchvision.transforms as T
from PIL import Image

plugin_path = "/root/TensorRT/build/out/libnvinfer_plugin.so"
print(f"Loading TensorRT plugin library: {plugin_path}")
ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)

# This line explicitly loads all available plugins
trt.init_libnvinfer_plugins(None, "")

import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os

class TRTFeatureExtractor:
    def __init__(
        self,
        engine_path,
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        output_dim=512,
        verbose=True
    ):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.image_size = image_size
        self.pixel_norm = pixel_norm
        self.pixel_mean = np.array(pixel_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.pixel_std = np.array(pixel_std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.input_dims = (3, image_size[0], image_size[1])
        self.output_dim = output_dim
        self.stream = cuda.Stream()

        # Get input/output tensor names
        self.input_name = next(name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.output_name = next(name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)

        # Buffer
        self.d_input = None
        self.d_output = None
        self.h_output = None

        if verbose:
            print(f"[TRT] Engine loaded from: {engine_path}")
            print(f"[TRT] Input: {self.input_name}, Output: {self.output_name}")
            print(f"[TRT] Input shape: {self.input_dims}, Output dim: {self.output_dim}")

    def load_engine(self, engine_path):
        try:
            with open(engine_path, "rb") as f:
                engine = self.runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    raise RuntimeError(f"TensorRT engine 無法反序列化: {engine_path}")
                return engine
        except Exception as e:
            raise RuntimeError(f"讀取 TensorRT engine 時發生錯誤: {e}")


    def preprocess(self, batch_imgs):
        if not isinstance(batch_imgs, np.ndarray):
            raise TypeError("輸入必須為 numpy.ndarray")
        if batch_imgs.ndim != 4 or batch_imgs.shape[-1] != 3:
            raise ValueError(f"輸入 shape 必須為 (N, H, W, 3)，目前為 {batch_imgs.shape}")

        N = batch_imgs.shape[0]
        out = np.empty((N, 3, self.image_size[0], self.image_size[1]), dtype=np.float32)

        for i in range(N):
            img = cv2.resize(batch_imgs[i], (self.image_size[1], self.image_size[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            out[i] = img

        # normalization
        if self.pixel_norm:
            out = (out - self.pixel_mean) / self.pixel_std

        return out

    def infer(self, input_data):
        assert input_data.dtype == np.float32
        assert input_data.shape[1:] == self.input_dims

        batch_size = input_data.shape[0]
        output_shape = (batch_size, self.output_dim)

        input_bytes = input_data.nbytes
        output_bytes = batch_size * self.output_dim * np.float32().itemsize

        if self.d_input is None or self.h_output is None or self.h_output.shape != output_shape:
            self.d_input = cuda.mem_alloc(input_bytes)
            self.d_output = cuda.mem_alloc(output_bytes)
            self.h_output = cuda.pagelocked_empty(output_shape, dtype=np.float32)

        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)

        if self.engine.binding_is_input(self.engine.get_binding_index(self.input_name)):
            self.context.set_input_shape(self.input_name, input_data.shape)

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.copy()

    def __call__(self, batch_imgs):
        input_data = self.preprocess(batch_imgs)
        
        # inference
        raw_feats = self.infer(input_data)

        # L2 normalization
        norms = np.linalg.norm(raw_feats, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1e-6, norms)
        return raw_feats / safe_norms
