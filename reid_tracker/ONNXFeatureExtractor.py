import onnxruntime as ort
import numpy as np
import cv2

class ONNXFeatureExtractor:
    def __init__(
        self,
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):
        assert model_path, "model_path 不能為空"
        
        # 選擇裝置
        providers = ['CUDAExecutionProvider'] if device.startswith('cuda') else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 設定參數
        self.image_size = image_size
        self.pixel_norm = pixel_norm
        self.pixel_mean = np.array(pixel_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.pixel_std = np.array(pixel_std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.input_dims = (3, image_size[0], image_size[1])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        if verbose:
            print(f"Loaded ONNX model from {model_path}")
            print(f"Input shape: {self.input_dims}")
            print(f"Providers: {self.session.get_providers()}")

    def preprocess(self, batch_imgs):
        """
        batch_imgs: np.ndarray with shape (N, H, W, 3), dtype=uint8 or float32
        Returns: np.ndarray with shape (N, 3, H, W), dtype=float32
        """
        if not isinstance(batch_imgs, np.ndarray):
            raise TypeError("輸入必須為 numpy.ndarray")
        if batch_imgs.ndim != 4 or batch_imgs.shape[-1] != 3:
            raise ValueError(f"輸入 shape 必須為 (N, H, W, 3)，目前為 {batch_imgs.shape}")

        N = batch_imgs.shape[0]
        out = np.empty((N, 3, self.image_size[0], self.image_size[1]), dtype=np.float32)

        for i in range(N):
            img = batch_imgs[i]
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            out[i] = img

        if self.pixel_norm:
            out = (out - self.pixel_mean) / self.pixel_std

        return out

    def infer(self, batch_imgs):
        input_data = self.preprocess(batch_imgs)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})[0]

        # L2 normalize
        norms = np.linalg.norm(outputs, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1e-8, norms)
        outputs /= safe_norms
        
        return outputs
        
    def __call__(self, batch_imgs):
        return self.infer(batch_imgs)
