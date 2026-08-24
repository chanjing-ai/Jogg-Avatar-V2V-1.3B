
"""
Ref https://github.com/hanson-young/nniefacelib/tree/master/PFPLD/models/onnx
"""
import os
import cv2
import onnxruntime as ort
import numpy as np

from cv2box import CVImage
from apstone import ModelBase
from face_landmark.utils import convert98to68

MODEL_ZOO = {
    'pfpld': {
        'model_path': 'models/pfpld.onnx',
        'model_input_size': (112, 112), },
}


class PFPLD(ModelBase):
    def __init__(self, model_name='pfpld', provider='gpu', cpu=False, device_id=0):

        # 如果使用GPU，重新初始化ONNX会话以指定设备
        if provider == 'gpu' and not cpu:
            self._init_gpu_session(device_id)
        else:
            super().__init__(MODEL_ZOO[model_name], provider)

    def _init_gpu_session(self, device_id):
        """
        重新初始化ONNX会话以使用指定的GPU设备
        """
        try:
            # 获取模型路径
            model_path = MODEL_ZOO['pfpld']['model_path']

            # 创建新的ONNX会话，指定GPU设备
            self.model = ort.InferenceSession(
                model_path,
                providers=[
                    ('CUDAExecutionProvider', {'device_id': device_id})
                ]
            )
            print(f"PFPLD模型已加载到GPU {device_id}")
        except Exception as e:
            print(f"PFPLD GPU初始化失败: {e}")
            # 回退到CPU
            self.model = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            print("PFPLD模型回退到CPU")

    def forward(self, face_image):
        """
        Args:
            face_image: RGB
        Returns:
        """
        input_image_shape = face_image.shape
        face_image = CVImage(face_image).resize((112, 112)).bgr
        face_image = (face_image / 255).astype(np.float32)
        pred = self.model.forward(face_image, trans=True)
        pred = convert98to68(pred[1])
        pred = pred.reshape(-1, 68, 2) * input_image_shape[:2][::-1]
        return pred
