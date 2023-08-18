from collections import deque


class SequenceData:
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



