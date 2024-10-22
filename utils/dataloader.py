# 该文件为数据加载器文件
# 数据输入为：data_root, split_mode  即  数据根目录   数据集类别（训练集or测试集）
# 数据输出为：返回两个值，以PIL中Image的格式返回 数据原图 和 其对应的掩码图
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import os.path
import random
import numpy as np
import torch
from os import path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.datasets.utils import list_files
import torchvision.transforms as transforms


def get_transform():
    # ImageNet中均值和方差的经验值
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # 加载钢材表面缺陷数据集


class Crack(Dataset):
    def __init__(self, data_root, split_mode):
        self.data_root = data_root  # data文件
        self.split_mode = split_mode  # 是训练还是测试
        self.transforms = transforms
        self.num_return = 2
        if self.split_mode == 'train':
            train_data_root = path.join(self.data_root, 'train')
            self.dataset = CrackPILDataset(data_root=train_data_root)
        elif self.split_mode == 'valid':
            valid_data_root = path.join(self.data_root, 'valid')
            self.dataset = CrackPILDataset(data_root=valid_data_root)
        elif self.split_mode == 'test':
            test_data_root = path.join(self.data_root, 'test')
            self.dataset = CrackPILDataset(data_root=test_data_root)
        else:
            self.logger.error('split_mode must be either "train" or "valid"')
            raise NotImplementedError

    def __getitem__(self, index):
        image, annot = self.dataset[index]
        # 增加数据的翻转
        if self.split_mode == 'train':
            if random.random() > 0.5:  # 百分之五十的概率
                image = TF.hflip(image)  # 水平翻转
                annot = TF.hflip(annot)
            if random.random() > 0.5:
                image = TF.vflip(image)  # 百分之五十的概率
                annot = TF.vflip(annot)  # 垂直翻转
        image = TF.to_tensor(image)
        annot = np.array(annot)
        annot = torch.tensor(annot, dtype=torch.float32)
        return image, annot

    def __len__(self):
        return len(self.dataset)


class CrackPILDataset(Dataset):  # 原始PIL图像的CFD数据集
    def __init__(self, data_root):
        self.data_root = path.expanduser(data_root)  # 使用Path对象处理路径
        self.image_dir = os.path.join(data_root, 'image')
        self.annot_dir = os.path.join(data_root, 'label')

        # print(f'Image Directory: {self.image_dir}')  # 调试输出
        # print(f'Annotation Directory: {self.annot_dir}')  # 调试输出

        self._image_paths = sorted(list_files(self.image_dir, suffix=('.jpg'), prefix=True))  # 导入原始文件并进行排序（001.jpg）
        self._annot_paths = sorted(list_files(self.annot_dir, suffix=('.png'), prefix=True))  # 导入分割文件并进行排序（001.png）

        # 确保图像和标注文件数量相同
        assert len(self._image_paths) == len(self._annot_paths), 'Crack dataset corrupted'  # 断言触发，防止数目不对等

    def __getitem__(self, index):  # 打开图片
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        annot = Image.open(self._annot_paths[index], mode='r').convert('L')
        return image, annot

    def __len__(self):
        return len(self._image_paths)


# if __name__ == '__main__':
#     Crack_dataset = Crack('./Dataset/', 'train')
#     print(Crack_dataset[0][1].shape)
#     # 训练集 3630
#     # 测试集 840
#     # 验证集 480
#     # 可视化加载出来的数据集中的标签集合
#     # plt.ion()
#     # for i in range(len(Crack_dataset)):
#     #     plt.imshow(torch.squeeze(Crack_dataset[i][1]).numpy())
#     #     plt.pause(0.1)
#     # plt.show()
