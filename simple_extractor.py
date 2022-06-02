'''
简单的特征提取器
author: MPolaris
time: 2022/05/31
'''
import cv2
import numpy as np

class simple_extractor(object):
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, imgs:list)->np.ndarray:
        features = np.zeros(shape=(len(imgs), 1024), dtype=np.float32)
        for i in range(len(imgs)):
            _img = cv2.resize(imgs[i], (32, 32))
            _img = np.mean(_img, axis=-1)
            features[i] = _img.reshape(-1)

        return features
    

if __name__ == "__main__":
    m = simple_extractor()
    img =[np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)]
    o = m(img)
    print(o.shape)