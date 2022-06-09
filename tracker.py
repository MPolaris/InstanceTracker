'''
instance跟踪算法
author: MPolaris
time: 2022/05/31
'''
import numpy as np

class InstanceTracker(object):
    '''
    追踪器类
    第一步:使用CNN进行区域特征提取
    第二步:进行检测区域与历史区域匹配
    第三步:更新检测器,删除无效历史特征
    第四步:给每一个检测框分配instance id
    '''
    def __init__(self, extractor:object, score_threshold=0.5, max_id=200, max_age=5, momentum=0.8, **kwargs) -> None:
        '''
        初始化
        :param extractor: 特征提取器
        :param score_threshold: 框匹配分数阈值
        :param max_id: instance_id最大值
        :param max_age: 特征最大漏检帧
        :param momentum: 特征更新动量
        '''
        self.max_id = max_id
        self.max_age = max_age
        self.score_th = score_threshold
        self.model = extractor
        
        # 历史有效的特征
        self.exist_feature_vetors = None
        # 历史有效的坐标
        self.exist_boxes_cxcy = None
        # 追踪框生命计数, <=0 时将被移除
        self.tracker_life = []
        # 每一个框的追踪id
        self.trace_ids = []
        # 变量更新momentum
        self.momentum = momentum
        # 追踪计数变量
        self.cur_trace_id = 1
  
    def __call__(self, image, bboxes, scores, class_ids):
        if len(bboxes) == 0 and self.exist_feature_vetors is None:
            # 初始化不成功
            return []
        
        cxcy = self.get_cxcy(bboxes)
        # 抽特征
        features = self.get_features(image, bboxes)
        if self.exist_feature_vetors is None:
            # 初始化
            self.exist_feature_vetors = features
            self.exist_boxes_cxcy = cxcy
            self.tracker_life = np.array([self.max_age]*len(features))
            self.trace_ids = np.array([i for i in range(len(features))])
            self.cur_trace_id = len(self.trace_ids)
        
        pair_list = [-1]*len(self.exist_feature_vetors)
        if len(features) > 0:
            # 计算特征相似度分数
            similarity_scores = self.compute_similarity(features, self.exist_feature_vetors)
            # 计算距离分数
            distance_scores = self.compute_similarity(cxcy, self.exist_boxes_cxcy)
            final_score = similarity_scores*distance_scores
            
            # 进行配对
            pair_list = self.get_match_pair(final_score)
            
        # 更新tracker状态
        _pair_list = self.update_tracker_state(pair_list, features, cxcy)

        # 分配追踪id号
        instance_ids = [-1]*len(bboxes)
        instance_ids = self.assign_instance_id(_pair_list, instance_ids)

        outputs = np.hstack([bboxes, instance_ids[:, np.newaxis], 
                            class_ids[:, np.newaxis], scores[:, np.newaxis]])
        return outputs

    def get_cxcy(self, x1y1x2y2):
        '''
        得到中心点坐标
        :param x1y1x2y2: ndnarray
        '''
        cxcy = np.zeros((len(x1y1x2y2), 2), dtype=x1y1x2y2.dtype)
        cxcy[:, 0] = x1y1x2y2[:, 2] - x1y1x2y2[:, 0]
        cxcy[:, 1] = x1y1x2y2[:, 3] - x1y1x2y2[:, 1]
        return cxcy

    def assign_instance_id(self, match_pair, instance_ids):
        '''
        分配instance id
        :param match_pair: list, b2a信息
        :param instance_ids: list, 对应的实例ID
        '''
        for i in range(len(match_pair)):
            if match_pair[i] == -1:
                continue
            instance_ids[match_pair[i]] = self.trace_ids[i]
        return np.array(instance_ids, dtype=np.int32)

    def update_tracker_state(self, match_pair, features, cxcy):
        '''
        更新追踪器状态
        :param match_pair: list, b2a信息
        :param features: 2D matrix, 特征矩阵
        :param cxcy: (N, 2), 中心点坐标
        '''
        match_pair = np.array(match_pair, dtype=np.int32)
        # 修正tracker生命周期
        active_indexes = []
        for i in range(len(match_pair)):
            if match_pair[i] == -1:
                # 如果该历史feature此次没有匹配到框
                self.tracker_life[i] -= 1
                if self.tracker_life[i] > 0:
                    active_indexes.append(i)
            else:
                # 如果该历史feature此次匹配到了框
                self.tracker_life[i] = self.max_age
                active_indexes.append(i)
                # 修正feature, 动量更新
                self.exist_feature_vetors[i] = self.momentum*self.exist_feature_vetors[i] + (1-self.momentum)*features[match_pair[i]]
                self.exist_boxes_cxcy[i] = cxcy[match_pair[i]]

        # 去除不活跃的历史tracker
        self.exist_feature_vetors = self.exist_feature_vetors[active_indexes]
        self.exist_boxes_cxcy = self.exist_boxes_cxcy[active_indexes]
        self.tracker_life = self.tracker_life[active_indexes]
        self.trace_ids = self.trace_ids[active_indexes]
        match_pair = match_pair[active_indexes]
        
        # 新增加的
        for i in range(len(features)):
            if i in match_pair:
                # 如果已经匹配到则跳过
                continue
            match_pair = np.hstack([match_pair, i])
            self.exist_feature_vetors = np.vstack([self.exist_feature_vetors, [features[i]]])
            self.exist_boxes_cxcy = np.vstack([self.exist_boxes_cxcy, [cxcy[i]]])
            self.tracker_life = np.hstack([self.tracker_life, self.max_age])
            self.trace_ids = np.hstack([self.trace_ids, self.cur_trace_id])
            self.cur_trace_id = (self.cur_trace_id + 1)%self.max_id

        return match_pair

    def get_match_pair(self, similarity_scores_matrix):
        '''
        匈牙利算法配对
        :param similarity_scores_matrix: (N1, N2), N1为检测框数量,N2为已经存在的检测框数量
        :return map_list
        '''
        # 本轮检测到的(A), 已经检测到的(B)
        detect_num, exist_num = similarity_scores_matrix.shape[0], similarity_scores_matrix.shape[1]
        
        # B被分配到哪一个A, -1代表没有分配对象
        b2a = [-1]*exist_num

        # B元素是否已经被访问过
        b_visited = [False]*exist_num

        sort_keys = [i for i in range(exist_num)]
        def __match(a_index:int):
            scores = similarity_scores_matrix[a_index]
            key = sorted(sort_keys, key=lambda x:scores[x], reverse=True)
            for b_index in key:
                if(scores[b_index] > self.score_th and (not b_visited[b_index])):
                    b_visited[b_index] = True
                    if(b2a[b_index] == -1 or (scores[b_index] > np.max(similarity_scores_matrix[b2a[b_index]]) and __match(b2a[b_index]))):
                        b2a[b_index] = a_index
                        return True
            return False

        for i in range(detect_num):
            b_visited = [False]*exist_num
            __match(i)

        return b2a
    
    def compute_similarity(self, A, B):
        '''
        计算余弦相似度
        :param Matrix A (N1, M)
        :param Matrix B (N2, M)
        :return Matrix Y (N1, N2)
        '''
        Y = np.dot(A, B.T) \
            / np.dot(np.linalg.norm(A, axis=1, keepdims=True), 
                      np.linalg.norm(B, axis=1, keepdims=True).T)
        return Y

    def compute_distance(self, A, B):
        '''
        计算欧式距离
        :param Matrix A (N1, 2)
        :param Matrix B (N2, 2)
        :return Matrix Y (N1, N2)
        '''
        Y = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
        for i in range(A.shape[0]):
            Y[i] = np.sqrt(np.sum(np.power(A[i] - B, 2), axis=-1))
        return Y

    def get_features(self, ori_img, bboxes):
        '''
        提取区域特征
        :param ori_img: 原始图像
        :param bboxes: 预测框
        :return features list(len(bboxes), M)
        '''
        im_crops = []
        for box in bboxes:
            x1, y1, x2, y2 = [int(i) for i in box]
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features