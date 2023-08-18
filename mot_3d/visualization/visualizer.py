import matplotlib.pyplot as plt
import numpy as np
import torch

from mot_3d.utils import SequenceData
from mot_3d.data_protos import BBox, Validity

try:
    import mayavi.mlab as mlab
    from mot_3d.visualization.visual_utils import visualize_utils as V
    OPEN3D_FLAG = False
except:
    import open3d
    from mot_3d.visualization.visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True


class VisualizerSequence:
    def __init__(self, figsize=(8, 8), figsize3d=(1000, 500), name=''):
        # self.database = database
        self.figsize = figsize
        self.figsize3d = figsize3d
        self.COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,
            'red': np.array([191, 4, 54]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256,
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 115, 67]) / 256
        }
        plt.axis('equal')

    def show2d(self, database: SequenceData):
        """
        database = {
            'track_boxes': result_pred_bboxes,
            'track_ids': result_pred_ids,
            'track_stat': result_pred_states,
            'gt': gt_bboxes[frame_index],
            'frame_data': frame_data
        }
        """
        fig = plt.figure(figsize=self.figsize)

        # 将点云的xy坐标绘制成散点图
        pc_list = [frame_data.pc for frame_data in database.frame_datas]
        if pc_list is not None:
            self.draw_pc_2d(pc_list)

        # 绘制gt框(gt无需显示text, 黑色, 默认实线)
        gts_list = database.gt_boxes
        if gts_list is not None:
            [self.draw_box_2d(gt, message='', color='black') for gts in gts_list for gt in gts]

        # 绘制pred框(显示score, 灰色, 虚线)
        dets_list = [frame_data.dets for frame_data in database.frame_datas]
        if dets_list is not None:
            [self.draw_box_2d(d, message='%.2f' % d.s, color='gray', linestyle='dashed')
                for dets in dets_list for d in dets if d.s >= 0.01]

        # 绘制目标追踪框
        track_boxes, track_ids, track_states = database.track_boxes, database.track_ids, database.track_states
        for track_box, track_id, track_state in zip(track_boxes, track_ids, track_states):
            for bbox, id, state in zip(track_box, track_id, track_state):
                # 追踪id存在(显示'score id'， 红色， 实线)
                if Validity.valid(state):
                    self.draw_box_2d(bbox, message='%.2f %s' % (bbox.s, id), color='red')
                # 追踪id置信度低(显示'score id'， 蓝色， 实线)
                else:
                    self.draw_box_2d(bbox, message='%.2f %s' % (bbox.s, id), color='light_blue')

        # 图像展示
        plt.show()
        plt.close()

    def draw_pc_2d(self, pc_list, color='gray'):
        assert pc_list is not None
        pc_list = np.concatenate(pc_list, axis=0)
        plt.scatter(pc_list[:, 0], pc_list[:, 1], marker='o', color=self.COLOR_MAP[color], s=0.01)

    def draw_box_2d(self, box: BBox, message: str='', color='red', linestyle='solid'):
        """
        Params:
            linestyle:["solid", "dotted", "dashed" or "dashdot"], 第一种'solid'是实线，后面三种都是虚线
        Func:
            按照颜色和线条设置绘制2d框, 默认是实线红色框
        """
        assert box is not None
        corners = np.array(BBox.box2corners2d(box))[:, :2]      # 获得2d上的box坐标
        corners = np.concatenate([corners, corners[0:1, :2]])   # 拼接起始点构造成一个环状结果
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)    # 绘线
        corner_index = np.random.randint(0, 4, 1)               # 随机一个角点位置显示文字
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[color])

    def show3d(self, database: SequenceData):
        """
        database = {
            'track_boxes': result_pred_bboxes,
            'track_ids': result_pred_ids,
            'track_stat': result_pred_states,
            'gt': gt_bboxes[frame_index],
            'frame_data': frame_data
        }
        """

        pc_list = [frame_data.pc for frame_data in database.frame_datas]
        pc = np.concatenate(pc_list, axis=0)

        V.draw_scenes(
            points=pc, ref_boxes=None,
            ref_scores=None, ref_labels=None
        )

        if not OPEN3D_FLAG:
            mlab.show(stop=True)



    def draw_pc_3d(self, pc_list, figure=None):
        assert pc_list is not None
        pc = np.concatenate(pc_list, axis=0)

        # fig = figure
        # print("====================", pc.shape)
        # if fig is None:  # 尺度颜色等设置
        #     fig = mlab.figure(
        #         figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        #     )

        # 直接显示原始点
        # mlab.points3d(
        #     pc[:, 0],
        #     pc[:, 1],
        #     pc[:, 2],
        #     pc[:, 2],
        #     color=None,
        #     mode="point",
        #     colormap="gnuplot",
        #     scale_factor=0.3,
        #     figure=fig,
        # )

    def draw_box_3d(self, box: BBox, message: str='', color='red', linestyle='solid'):
        pass