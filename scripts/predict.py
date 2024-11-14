import torch
import numpy as np
import os
import sys
import time
from os import path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
from torchvision.datasets.utils import list_files
import sys
import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from model import self_net  # 导入模型结构


class Predict:
    """
    Predict类，用于加载训练好的模型，并对数据集进行预测和评估。
    """

    def __init__(self, model_path=None, data_root='./Dataset', device=None):
        """
        初始化Predict类。

        参数:
        model_path : str, 训练好的模型路径
        data_root : str, 数据集路径
        device : str, 设备选择
        """
        if not model_path:
            raise ValueError("Model path must be provided to load the model.")

        self.model_path = model_path
        self.data_root = data_root
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型权重
        self.net = self_net().to(self.device)
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net.eval()

    def evaluate(self, num_classes=4):
        """
        执行模型预测和评估。

        参数:
        num_classes : int, 类别数量
        """
        test_loader = torch.utils.data.DataLoader(
            Crack(self.data_root, 'test'),
            batch_size=1, shuffle=False,
            num_workers=4, pin_memory=True)

        # 确保输出文件夹存在
        output_dirs = [
            'c_test_predictions'
        ]
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)

        with torch.no_grad():

            for i, img in enumerate(test_loader, start=1):
                img = img.to(self.device)

                # 模型预测
                pred = torch.argmax(self.net(img).squeeze(0), dim=0, keepdim=True).cpu().numpy()
                # 保存预测结果为.npy格式
                np.save(f'c_test_predictions/c_prediction_{i:06d}.npy', pred)


class Crack(Dataset):
    def __init__(self, data_root, split_mode):
        self.data_root = data_root  # data文件
        self.split_mode = split_mode  # 是训练还是测试
        if self.split_mode == 'test':
            self.dataset = CrackPILDataset(data_root=self.data_root)
        else:
            raise NotImplementedError('split_mode must be "test"')

    def __getitem__(self, index):
        image = self.dataset[index]
        image = TF.to_tensor(image)
        return image

    def __len__(self):
        return len(self.dataset)


class CrackPILDataset(Dataset):  # 原始PIL图像的CFD数据集
    def __init__(self, data_root):
        self.data_root = path.expanduser(data_root)  # 使用Path对象处理路径
        self._image_paths = sorted(list_files(self.data_root, suffix='.jpg'))

    def __getitem__(self, index):  # 打开图片
        image = Image.open(os.path.join(self.data_root, self._image_paths[index]), mode='r').convert('RGB')
        return image

    def __len__(self):
        return len(self._image_paths)


if __name__ == "__main__":
    model_path = 'model.pth'  # 模型路径
    data_root = 'npy 生成文件材料/datasets_C'  # 数据集路径
    print(os.path.abspath(data_root))

    # 实例化 Predict 类并执行预测
    predictor = Predict(model_path=model_path, data_root=data_root)
    predictor.evaluate(num_classes=4)