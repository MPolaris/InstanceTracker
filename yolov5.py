'''
yolov5包装类
author: MPolaris
time: 2022/05/31
'''
import os
import cv2
import numpy as np
import tensorflow as tf

from utils import preprocess_img, detect_postprocess

class Yolov5(object):
    coco_class = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

    def __init__(self, model_path:str, conf_thres:float=0.5, iou_thres:float=0.4, interested_classid:list=None) -> None:
        """
        Yolov5 包装类
        :param model_path: 模型路径
        :param conf_thres: 置信度阈值
        :param iou_thres: nms iou阈值
        :param interested_classid: 输出类别的id,应为list,为None时则输出所有
        :return None
        """
        assert os.path.exists(model_path), f"{model_path} 不存在"
        if interested_classid is not None:
            assert isinstance(interested_classid, list), f"interested_classid 应为list或None"
        self.interested_classid = interested_classid
        self.model = tf.lite.Interpreter(model_path=model_path, num_threads=4)
        self.model.allocate_tensors()
        self.input_details, self.output_details  = self.model.get_input_details(), self.model.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def invoke(self, ori_img:np.ndarray) -> np.ndarray:
        '''
        推理函数
        :param ori_img: 预测图像
        :return: 预测坐标框信息, np.ndarray(N, 6), xywh、conf、cls_id
        '''
        input_img = preprocess_img(ori_img, 
                                    target_shape=(self.input_shape[1], self.input_shape[2]),
                                    div_num=255,
                                    means=None,
                                    stds=None)
        self.model.set_tensor(self.input_details[0]['index'], input_img)
        self.model.invoke()
        det_pred = self.model.get_tensor(self.output_details[0]['index'])[0]
        det_pred = detect_postprocess(det_pred, ori_img.shape, [self.input_shape[1], self.input_shape[2], 3], 
                                        conf_thres=0.5, iou_thres=0.4, interested_classid=self.interested_classid)
        return det_pred

    def draw_pic(self, img:np.ndarray, det_pred:np.ndarray) -> np.ndarray:
        '''
        绘图函数
        :param img: 图像
        :param det_pred: 预测结果 np.ndarray(N, 6), 坐标框信息, x1y1x2y2、conf、cls_id
        :return: 完成预测结果绘制的图像
        '''
        img = img.astype(np.uint8)
        color_step = int(255/len(det_pred))
        for i in range(len(det_pred)):
            x1, y1, x2, y2, conf, cls_id  = [int(t) for t in det_pred[i]]
            cv2.putText(img, f'{self.coco_class[cls_id]}', (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, cls_id*color_step, 255-cls_id*color_step),thickness = 2)
        return img