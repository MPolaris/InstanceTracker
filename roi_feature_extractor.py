'''
基于CNN的特征提取器
author: MPolaris
time: 2022/05/31
'''
import numpy as np
import tensorflow as tf
from utils import preprocess_img

class feature_extracor_network(object):
    def __init__(self, model_path:str, means:list=[0.485, 0.456, 0.406], stds:list=[0.229, 0.224, 0.225]) -> None:
        self.means = means
        self.stds = stds

        self.model = tf.lite.Interpreter(model_path=model_path, num_threads=4)
        self.model.allocate_tensors()
        self.input_details, self.output_details  = self.model.get_input_details(), self.model.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']
        
    def __call__(self, imgs:list)->np.ndarray:
        self.output_shape[0] = len(imgs)
        features = np.zeros(shape=self.output_shape, dtype=np.float32)
        for i in range(len(imgs)):
            _input = preprocess_img(imgs[i], keep_ratio=False, 
                                            target_shape=(self.input_shape[1], self.input_shape[2]),
                                            div_num=255,
                                            means=self.means,
                                            stds=self.stds)
            self.model.set_tensor(self.input_details[0]['index'], _input)
            self.model.invoke()
            features[i] = self.model.get_tensor(self.output_details[0]['index'])
        return features

if __name__ == "__main__":
    model = feature_extracor_network("./models/feature_extractor_osnet.tflite")
    imgs = []
    for i in range(10):
        imgs.append(np.random.randn(128, 128, 3))
    out = model(imgs)
    print(out.shape)