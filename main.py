'''
YOLOV5 实例追踪算法
author: MPolaris
time: 2022/05/31
'''
import cv2
import time
import argparse

from yolov5 import Yolov5
from tracker import InstanceTracker
from utils import draw_track_res

def main(video_path:str="./sample.mp4", 
            det_model_path:str="./models/yolov5s-640.tflite", 
            ext_model_path:str="./models/feature_extractor_osnet.tflite",
            class_ids=[0],
            simple_extractor:bool=False):
    '''
    :param video_path:str 需检测的视频地址
    :param det_model_path:str yolov5模型地址
    :param ext_model_path:str 特征提取cnn网络模型地址, simple_extractor=True时不生效
    :param class_ids:list 需要追踪显示的id号, person=0
    :param simple_extractor:bool 是否使用简单特征提取, False时使用CNN进行特征提取
    '''
    detector = Yolov5(model_path=det_model_path, 
                        conf_thres=0.4, 
                        iou_thres=0.4, 
                        interested_classid=class_ids)

    extractor = None
    if simple_extractor:
        from simple_extractor import simple_extractor
        extractor = simple_extractor()
    else:
        from roi_feature_extractor import feature_extracor_network
        extractor = feature_extracor_network(model_path=ext_model_path)

    tracktor = InstanceTracker(extractor=extractor,
                                score_threshold=0.3,
                                max_age=24,
                                momentum=0.8)

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"video_path:{video_path} is done~")
            break
        det_time = time.time()
        det_res = detector.invoke(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det_time = time.time() - det_time

        x1y1x2y2, confs, clss = det_res[:, 0:4], det_res[:, 4], det_res[:, 5]
        
        track_time = time.time()
        track_res = tracktor(frame, x1y1x2y2, confs, clss)
        track_time = time.time() - track_time
        track_time = max(track_time, 0.0002)

        frame = draw_track_res(frame, track_res)
        cv2.putText(frame, "Track Time:{:^2.0f}ms, FPS:{:^2.0f}".format(track_time*1000, 1/track_time), (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        cv2.putText(frame, "Detect Time:{:^2.0f}ms, FPS:{:^2.0f}".format(det_time*1000, 1/det_time), (2, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow("show time", frame)
        cv2.waitKey(5)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default="./sample.mp4", help='视频地址')
    parser.add_argument('--det_model_path', type=str, default="./models/yolov5s-640.tflite", help='yolov5模型地址')
    parser.add_argument('--ext_model_path', type=str, default="./models/feature_extractor_osnet.tflite", help='特征提取cnn网络模型地址')
    parser.add_argument('--class_ids', type=int, nargs='+', default=[], help='需要追踪显示的id号, person=0')
    parser.add_argument('--simple_extractor', default=False, action='store_true', help='是否使用简单特征提取, False时使用CNN进行特征提取')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(video_path=opt.video_path,
            det_model_path=opt.det_model_path,
            ext_model_path=opt.ext_model_path,
            class_ids=opt.class_ids,
            simple_extractor=opt.simple_extractor)