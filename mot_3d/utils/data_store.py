from collections import deque
import numpy as np
from mot_3d.utils.data_process import global_to_lidar

class SequenceDatabase:
    """
    多帧信息存储器, 用来存储多帧的信息，结构是队列的类型
    需要存储的内容包括: bboxes, ids, states, gt_bboxes, pc, dets
    """
    def __init__(self, space_num=5):
        """
        Params:
        """
        self.track_boxes = deque()
        self.track_ids = deque()
        self.track_states = deque()
        self.gt_boxes = deque()
        self.frame_datas = deque()
        # self.pcs = deque()
        # self.det_boxes = deque()

    def __len__(self):
        return len(self.track_boxes)

    def pop(self):
        """
            出队
        """
        self.track_boxes.pop()
        self.track_ids.pop()
        self.track_states.pop()
        self.gt_boxes.pop()
        self.frame_datas.pop()

    def append(self, seq_data):
        """
            入队, 先进先出
            seq_data = {
                'track_boxes': result_pred_bboxes,
                'track_ids': result_pred_ids,
                'track_stat': result_pred_states,
                'gt': gt_bboxes[frame_index],
                'frame_data': frame_data
            }
        """
        self.track_boxes.appendleft(seq_data['track_boxes'])
        self.track_ids.appendleft(seq_data['track_ids'])
        self.track_states.appendleft(seq_data['track_stat'])
        self.gt_boxes.appendleft(seq_data['gt'])
        self.frame_datas.appendleft(seq_data['frame_data'])


class SeqData:
    def __init__(self, track_boxes, track_ids, track_stat, gt, pc, frame_data):
        self.track_boxes = track_boxes
        self.track_ids = track_ids
        self.track_stat = track_stat
        self.gt = gt        # without score
        self.frame_data = frame_data
        self.pc = pc    # 激光雷达坐标系
        self.dets = list()

    def trans_world_to_pc(self, calib_data, ego_data):
        # mot to ndarray: [xyz, lwh, heading, score]
        # 可以使用类方法： Bbox.bbox2array
        gt = [np.array([bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o])
                   for bbox in self.gt]
        track_boxes = [np.array([bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o, bbox.s])
                            for bbox in self.track_boxes]
        dets = [np.array([bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o, bbox.s])
                            for bbox in self.frame_data.dets]

        # ndarray to lidar
        self.gt = [global_to_lidar(bbox, calib_data=calib_data, ego_data=ego_data) for bbox in gt]
        self.track_boxes = [np.hstack([global_to_lidar(bbox[:7], calib_data=calib_data, ego_data=ego_data), bbox[-1]])
                            for bbox in track_boxes]
        self.dets = [np.hstack([global_to_lidar(bbox[:7], calib_data=calib_data, ego_data=ego_data), bbox[-1]])
                     for bbox in dets]

    def trans_pc_to_world(self, calib_data, ego_data):
        pass

