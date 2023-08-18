import os, numpy as np, nuscenes, argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
import matplotlib.pyplot as plt
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../raw/nuscenes/data/sets/nuscenes/')
parser.add_argument('--data_folder', type=str, default='../../../datasets/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()


def load_pc(path):
    pc = np.fromfile(path, dtype=np.float32)
    pc = pc.reshape((-1, 5))[:, :4]
    return pc


def main(nusc, scene_names, root_path, pc_folder, mode, pid=0, process=1):
    for scene_index, scene_info in enumerate(nusc.scene):
        if scene_index % process != pid:
            continue
        scene_name = scene_info['name']
        if scene_name not in scene_names:
            continue
        print('PROCESSING {:} / {:}'.format(scene_index + 1, len(nusc.scene)))

        first_sample_token = scene_info['first_sample_token']
        frame_data = nusc.get('sample', first_sample_token)
        if mode == '20hz':
            cur_sample_token = frame_data['data']['LIDAR_TOP']
        elif mode == '2hz':
            cur_sample_token = deepcopy(first_sample_token)
        frame_index = 0
        pc_data = dict()
        while True:
            # find the path to lidar data
            if mode == '2hz':
                lidar_data = nusc.get('sample', cur_sample_token)
                lidar_path = nusc.get_sample_data_path(lidar_data['data']['LIDAR_TOP'])
            elif args.mode == '20hz':
                lidar_data = nusc.get('sample_data', cur_sample_token)
                lidar_path = lidar_data['filename']

            # load and store the data (fix some problems)
            if mode == '2hz':
                point_cloud = np.fromfile(lidar_path, dtype=np.float32)
            elif mode == '20hz':
                point_cloud = np.fromfile(os.path.join(root_path, lidar_path), dtype=np.float32)
            point_cloud = np.reshape(point_cloud, (-1, 5))[:, :4]
            pc_data[str(frame_index)] = point_cloud

            # clean up and prepare for the next
            cur_sample_token = lidar_data['next']
            if cur_sample_token == '':
                break
            frame_index += 1

            if frame_index % 10 == 0:
                print('PROCESSING ', scene_index, ' , ', frame_index)
        
        np.savez_compressed(os.path.join(pc_folder, '{:}.npz'.format(scene_name)), **pc_data)
    return


if __name__ == '__main__':
    print('point cloud')
    pc_folder = os.path.join(args.data_folder, 'pc', 'raw_pc')
    os.makedirs(pc_folder, exist_ok=True)

    from version import version, split_mode
    nusc = NuScenes(version=version, dataroot=args.raw_data_folder, verbose=True)
    val_scene_names = splits.create_splits_scenes()[split_mode]
    main(nusc, val_scene_names, args.raw_data_folder, pc_folder, args.mode, 0, 1)
