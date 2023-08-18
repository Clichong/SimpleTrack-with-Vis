""" inference on the nuscenes dataset
"""
import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from data_loader import NuScenesLoader
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import pickle

from mot_3d.utils import SequenceData
from mot_3d.visualization import seq_frame_visualization, VisualizerSequence


parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--name', type=str, default='SimpleTrack2Hz')
parser.add_argument('--det_name', type=str, default='swim')
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--start_frame', type=int, default=0, help='start at a middle frame for debug')
parser.add_argument('--obj_types', default='car,bus,trailer,truck,pedestrian,bicycle,motorcycle')
# paths
parser.add_argument('--config_path', type=str, default='../configs/nu_configs/giou.yaml',
                    help='config file path, follow the path in the documentation')
parser.add_argument('--result_folder', type=str, default='../output/')
parser.add_argument('--data_folder', type=str, default='../data/data_dir_2hz/')
args = parser.parse_args()


def nu_array2mot_bbox(b):
    """
    格式的转化作用
    """
    nu_box = Box(b[:3], b[3:6], Quaternion(b[6:10]))
    mot_bbox = BBox(
        x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2],
        w=nu_box.wlh[0], l=nu_box.wlh[1], h=nu_box.wlh[2],
        o=nu_box.orientation.yaw_pitch_roll[0]
    )
    if len(b) == 11:
        mot_bbox.s = b[-1]
    return mot_bbox


def load_gt_bboxes(data_folder, type_token, segment_name):
    """
    Params：
        data_folder: 存储gt信息目录
        type_token: 需要处理的类别
        segment_name: 需要处理的名称，根据segment_name来在路径中找到对应的序列信息
    Func:
        首先根据路径读取对应的scenes的gt信息，根据类别设置来对scens的每一帧进行筛选，最后见box转化为mot_bbox的形式，
        每一个box的形式都是一个mot_bbox类信息
    """
    with open(os.path.join(data_folder, 'gt_info', '{:}.npz'.format(segment_name)), 'r') as f:
        gt_info = json.load(f)
    # gt_info = np.load(os.path.join(data_folder, 'gt_info', '{:}.npz'.format(segment_name)), allow_pickle=True)
    ids, inst_types, bboxes = gt_info['ids'], gt_info['types'], gt_info['bboxes']
    
    mot_bboxes = list()
    for _, frame_bboxes in enumerate(bboxes):
        mot_bboxes.append([])
        # chance the boxes format from 10dim to 7dim(Box)
        for _, b in enumerate(frame_bboxes):
            mot_bboxes[-1].append(BBox.bbox2array(nu_array2mot_bbox(b)))

    # 筛选出选择好的类别gt
    gt_ids, gt_bboxes = utils.inst_filter(ids, mot_bboxes, inst_types, 
        type_field=type_token, id_trans=True)
    return gt_bboxes, gt_ids


def frame_visualization(bboxes, ids, states, gt_bboxes=None, gt_ids=None, pc=None, dets=None, name=''):
    # 初始化图像大小以及颜色设置
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))

    # 将点云的xy坐标绘制成散点图
    if pc is not None:
        visualizer.handler_pc(pc)

    # 绘制gt框(gt无需显示text, 黑色, 默认实线)
    if gt_bboxes is not None:
        for _, bbox in enumerate(gt_bboxes):
            visualizer.handler_box(bbox, message='', color='black')

    # 绘制pred框(显示score, 灰色, 虚线)
    dets = [d for d in dets if d.s >= 0.01]     # 筛选低于0.01的检测结果
    for det in dets:
        visualizer.handler_box(det, message='%.2f' % det.s, color='gray', linestyle='dashed')

    # 绘制目标追踪框
    for _, (bbox, id, state_string) in enumerate(zip(bboxes, ids, states)):
        # 追踪id存在(显示'score id'， 红色， 实线)
        if Validity.valid(state_string):
            visualizer.handler_box(bbox, message='%.2f %s'%(bbox.s, id), color='red')
        # 追踪id置信度低(显示'score id'， 蓝色， 实线)
        else:
            visualizer.handler_box(bbox, message='%.2f %s'%(bbox.s, id), color='light_blue')

    # 图像展示
    visualizer.show()
    visualizer.close()


def sequence_mot(configs, data_loader, obj_type, sequence_id, gt_bboxes=None, gt_ids=None,
                 visualize=False, visualize_seq=True):
    tracker = MOTModel(configs)
    frame_num = len(data_loader)
    IDs, bboxes, states, types = list(), list(), list(), list()

    seq_database = SequenceData(space_num=5)
    visualizer = VisualizerSequence(figsize=(12, 12))

    # process frame from cur_frame to frame_num
    for frame_index in range(data_loader.cur_frame, frame_num):
        if frame_index % 10 == 0:
            print('TYPE {:} SEQ {:} Frame {:} / {:}'.format(obj_type, sequence_id, frame_index + 1, frame_num))
        
        # input data
        frame_data = next(data_loader)
        frame_data = FrameData(dets=frame_data['dets'], ego=frame_data['ego'], pc=frame_data['pc'], 
            det_types=frame_data['det_types'], aux_info=frame_data['aux_info'], time_stamp=frame_data['time_stamp'])

        # mot
        results = tracker.frame_mot(frame_data)
        result_pred_bboxes = [trk[0] for trk in results]
        result_pred_ids = [trk[1] for trk in results]
        result_pred_states = [trk[2] for trk in results]
        result_types = [trk[3] for trk in results]

        # visualization
        if visualize:
            frame_visualization(result_pred_bboxes, result_pred_ids, result_pred_states,
                gt_bboxes[frame_index], gt_ids[frame_index], frame_data.pc, dets=frame_data.dets, name='{:}_{:}'.format(args.name, frame_index))

        if visualize_seq:
            seq_data = {'track_boxes': result_pred_bboxes,
                        'track_ids': result_pred_ids,
                        'track_stat': result_pred_states,
                        'gt': gt_bboxes[frame_index],
                        'frame_data': frame_data
            }
            seq_database.append(seq_data)
            if len(seq_database) == 3:
                visualizer.show3d(seq_database)
                seq_database.pop()

        # wrap for output
        IDs.append(result_pred_ids)
        result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
        bboxes.append(result_pred_bboxes)
        states.append(result_pred_states)
        types.append(result_types)

    return IDs, bboxes, states, types


def main(name, obj_types, config_path, data_folder, det_data_folder, result_folder, start_frame=0, token=0, process=1):
    for obj_type in obj_types:
        summary_folder = os.path.join(result_folder, 'summary', obj_type)

        # simply knowing about all the segments
        file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
        
        # load model configs
        configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

        # process every scene in prepare dataset
        for file_index, file_name in enumerate(file_names[:]):
            if file_index % process != token:
                continue
            print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, file_index + 1, len(file_names)))

            # For every scene to create a scene loader
            segment_name = file_name.split('.')[0]
            data_loader = NuScenesLoader(configs, [obj_type], segment_name, data_folder, det_data_folder, start_frame)

            # load the gt information and filter by the object type in need
            gt_bboxes, gt_ids = load_gt_bboxes(data_folder, [obj_type], segment_name)

            # get every frame data in a scene to tracking and visualization
            ids, bboxes, states, types = sequence_mot(configs, data_loader, obj_type, file_index, gt_bboxes, gt_ids, args.visualize)

            #
            frame_num = len(ids)
            for frame_index in range(frame_num):
                id_num = len(ids[frame_index])
                for i in range(id_num):
                    ids[frame_index][i] = '{:}_{:}'.format(file_index, ids[frame_index][i])

            # chance the way to save
            with open(os.path.join(summary_folder, '{}.npz'.format(segment_name)), 'wb') as f:
                pickle.dump(
                    {'ids': ids, 'bboxes': bboxes, 'states': states, 'types': types}, f
                )
            # np.savez_compressed(os.path.join(summary_folder, '{}.npz'.format(segment_name)),
            #     ids=ids, bboxes=bboxes, states=states, types=types)


if __name__ == '__main__':
    result_folder = os.path.join(args.result_folder, args.name)
    os.makedirs(result_folder, exist_ok=True)
    summary_folder = os.path.join(result_folder, 'summary')
    os.makedirs(summary_folder, exist_ok=True)
    det_data_folder = os.path.join(args.data_folder, 'detection', args.det_name)

    # create folder for every class
    obj_types = args.obj_types.split(',')
    for obj_type in obj_types:
        tmp_summary_folder = os.path.join(summary_folder, obj_type)
        os.makedirs(tmp_summary_folder, exist_ok=True)

    # mutilprocess
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.name, obj_types, args.config_path, args.data_folder, det_data_folder, 
                result_folder, 0, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.name, obj_types, args.config_path, args.data_folder, det_data_folder, 
            result_folder, args.start_frame, 0, 1)