import matplotlib.pyplot as plt, numpy as np
from ..data_protos import BBox


class Visualizer2D:
    def __init__(self, name='', figsize=(8, 8)):
        self.figure = plt.figure(name, figsize=figsize)
        plt.axis('equal')
        self.COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,
            'red': np.array([191, 4, 54]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 115, 67]) / 256
        }
    
    def show(self):
        plt.show()
    
    def close(self):
        plt.close()
    
    def save(self, path):
        plt.savefig(path)
    
    def handler_pc(self, pc, color='gray'):
        vis_pc = np.asarray(pc)
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], marker='o', color=self.COLOR_MAP[color], s=0.01)
    
    def handler_box(self, box: BBox, message: str='', color='red', linestyle='solid'):
        """
        Params:
            linestyle:["solid", "dotted", "dashed" or "dashdot"], 第一种'solid'是实线，后面三种都是虚线
        Func:
            按照颜色和线条设置绘制2d框, 默认是实线红色框
        """
        corners = np.array(BBox.box2corners2d(box))[:, :2]      # 获得2d上的box坐标
        corners = np.concatenate([corners, corners[0:1, :2]])   # 拼接起始点构造成一个环状结果
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)    # 绘线
        corner_index = np.random.randint(0, 4, 1)               # 随机一个角点位置显示文字
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[color])