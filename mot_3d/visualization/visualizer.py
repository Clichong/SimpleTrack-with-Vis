import matplotlib.pyplot as plt
import numpy as np

from mot_3d.utils.data_store import SequenceDatabase
from mot_3d.data_protos import BBox, Validity
import mayavi.mlab as mlab

from mot_3d.visualization.visual_utils import visualize_utils

# try:
#     import mayavi.mlab as mlab
#     from mot_3d.visualization.visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False
# except:
#     import open3d
#     from mot_3d.visualization.visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True


class VisualizerSequence:
    def __init__(self, figsize=(12, 12), figsize3d=(1100, 700), name=''):
        # self.database = database
        self.figsize = figsize
        self.figsize3d = figsize3d
        self.COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,    # 蓝色表示追踪框(低置信度)
            'red': np.array([191, 4, 54]) / 256,            # 红色表示追踪框
            'black': np.array([0, 0, 0]) / 256,             # 黑色表示gt
            'purple': np.array([224, 133, 250]) / 256,
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 115, 67]) / 256          # 绿色表示绿色框
        }
        plt.axis('equal')

    def show2d(self, database):
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
        pc_list = [seq.frame_data.pc for seq in database]
        if pc_list is not None:
            self.draw_pc_2d(pc_list)

        # 绘制gt框(gt无需显示text, 黑色, 默认实线)
        gts_list = [seq.gt for seq in database]
        if gts_list is not None:
            [self.draw_box_2d(gt, message='', color='black') for gts in gts_list for gt in gts]

        # 绘制pred框(显示score, 灰色, 虚线)
        dets_list = [seq.frame_data.dets for seq in database]
        if dets_list is not None:
            [self.draw_box_2d(d, message='%.2f' % d.s, color='gray', linestyle='dashed')
                for dets in dets_list for d in dets if d.s >= 0.01]

        # 绘制目标追踪框
        track_boxes, track_ids, track_states = [seq.track_boxes for seq in database], \
                [seq.track_ids for seq in database], [seq.track_stat for seq in database]
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

    def show3d(self, database):
        """
        database = {
            'track_boxes': result_pred_bboxes,
            'track_ids': result_pred_ids,
            'track_stat': result_pred_states,
            'gt': gt_bboxes[frame_index],
            'pc': lidar coor point cloud,
            'det': lidar coor pred boxes
        }
        """

        figsize = self.figsize3d
        fig = mlab.figure(figure=None, bgcolor=(1.0, 1.0, 1.0), fgcolor=(0, 0, 0), engine=None, size=figsize)

        # 绘制3d点云
        pc_list = [seq.pc for seq in database]
        pc = np.concatenate(pc_list, axis=0)
        fig = self.draw_pc_3d(pc, fig=fig)
        # fig = visualize_utils.draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))       # draw grid

        # 绘制3d gt框(黑框 / without score)
        gt_boxes = np.vstack([seq.gt for seq in database])      # (N, 7)
        if gt_boxes is not None:
            corners3d = visualize_utils.boxes_to_corners_3d(gt_boxes)
            color = tuple(self.COLOR_MAP['black'])
            fig = self.draw_box_3d(corners3d, fig=fig, color=color, max_num=100)

        # 绘制pred框(显示score, 灰色, 虚线)
        det_boxes = np.vstack([seq.dets for seq in database])   # (N, 8)
        if det_boxes is not None:
            corners3d = visualize_utils.boxes_to_corners_3d(det_boxes[:, :7])
            color = tuple(self.COLOR_MAP['gray'])
            score = det_boxes[:, -1]
            fig = self.draw_box_3d(corners3d, fig=fig, color=color, max_num=100, message=score)

        # 绘制目标追踪框
        track_boxes = np.vstack([seq.track_boxes for seq in database])  # (N, 8)
        track_ids = np.hstack([seq.track_ids for seq in database])
        track_states = np.hstack([seq.track_stat for seq in database])
        assert len(track_states) == len(track_ids) == len(track_boxes)

        score = track_boxes[:, -1]
        corners3d = visualize_utils.boxes_to_corners_3d(track_boxes[:, :7])
        for i in range(len(corners3d)):
            bbox, id, state = corners3d[i], track_ids[i], track_states[i]
            message = '%.2f %s' % (score[i], id)
            # 追踪id存在(显示'score id'， 红色， 实线)
            if Validity.valid(state):
                fig = self.draw_box_3d(bbox[None, :], fig=fig,
                                       color=tuple(self.COLOR_MAP['red']), message=[message])
            # 追踪id置信度低(显示'score id'， 蓝色， 实线)
            else:
                fig = self.draw_box_3d(bbox[None, :], fig=fig,
                                       color=tuple(self.COLOR_MAP['light_blue']), message=[message])

        # 指定角度查看点云
        # elevation: 沿x轴旋转
        # azimuth:  沿y轴旋转
        # roll: 沿z轴旋转
        # mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
        mlab.view(azimuth=-179, elevation=-54.0, distance=120.0, roll=-90.0, figure=fig)

        mlab.show()
        # mlab.close()
        return fig


    def draw_pc_3d(self, pts, fig, show_intensity=False, draw_origin=True):
        if show_intensity:
            G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                              colormap='gnuplot', scale_factor=1, figure=fig)
        else:
            G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                              colormap='gnuplot', scale_factor=1, figure=fig)
        if draw_origin:
            mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
            mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
            mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
            mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

        return fig

    def draw_box_3d(self, corners3d, fig, color=(1, 1, 1), line_width=2, message=None, max_num=500, tube_radius=None):
        """
        :param corners3d: (N, 8, 3)
        :param fig:
        :param color:
        :param line_width:
        :param message:
        :param max_num:
        :return:
        """
        num = min(max_num, len(corners3d))    # 最多显示500个框
        for n in range(num):
            b = corners3d[n]  # (8, 3)

            # 绘制字符串信息
            if message is not None:
                text = '%.2f' % message[n] if isinstance(message, np.ndarray) else '%s' % message[n]
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], text, scale=(1.0, 1.0, 1.0),
                            color=color, figure=fig, line_width=4)

            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                            tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

                i, j = k + 4, (k + 1) % 4 + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                            tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

                i, j = k, k + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                            tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

            i, j = 0, 5
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)
            i, j = 1, 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        return fig
