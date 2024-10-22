import torch
import numpy as np
import os
import time
from PIL import Image
import cv2  # 用于图像叠加
from utils.dataloader import Crack
from utils.tools import *

class Predict():
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
        self.model_path = model_path
        self.data_root ='datasets' + data_root
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载完整的模型
        if self.model_path:
            self.net = torch.load(self.model_path, map_location=self.device)
        else:
            raise ValueError("Model path must be provided to load the model.")
        
        self.net = self.net.to(self.device)
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
        os.makedirs('./mask/test_predictions', exist_ok=True)
        os.makedirs('./mask/test_ground_truths', exist_ok=True)
        os.makedirs('./mask/baseline_predictions', exist_ok=True)

        os.makedirs('./submission/test_predictions', exist_ok=True)
        os.makedirs('./submission/test_ground_truths', exist_ok=True)
        os.makedirs('./submission/baseline_predictions', exist_ok=True)

        os.makedirs('./markadd/test_predictions', exist_ok=True)
        os.makedirs('./markadd/test_ground_truths', exist_ok=True)
        os.makedirs('./markadd/baseline_predictions', exist_ok=True)

        with torch.no_grad():
            pred_time = 0
            pred_img_num = 0
            iou = [[], [], []]  # 存储 IOU 数据

            for i, (img, lab) in enumerate(test_loader, start=1):
                img, lab = img.to(self.device), lab.to(self.device)
                lab = lab.type(torch.LongTensor)

                # 模型预测
                pred_start_time = time.time()
                pred = torch.argmax(self.net(img).squeeze(0), dim=0, keepdim=True).cpu().numpy()
                pred_end_time = time.time()
                pred_time += (pred_end_time - pred_start_time)
                pred_img_num += 1
                lab = lab.cpu().numpy()

                # 保存预测结果和ground truth为.npy格式
                np.save(f'./submission/test_predictions/prediction_{i:06d}.npy', pred)
                np.save(f'./submission/test_ground_truths/ground_truth_{i:06d}.npy', lab)

                # 转换为掩码图像 (灰度，假设是二分类问题)
                pred_mask = Image.fromarray((pred.squeeze(0) * 255).astype(np.uint8))
                lab_mask = Image.fromarray((lab.squeeze(0) * 255).astype(np.uint8))

                # 计算 IOU
                iou_pred = compute_iou_with_matrix(pred, lab, num_classes)
                for j in range(1, num_classes):
                    if iou_pred[j] >= 0:
                        iou[1].append(iou_pred[j])

                # 保存掩码图像
                pred_mask.save(f'./mask/test_predictions/mask_{i:06d}.png')
                lab_mask.save(f'./mask/test_ground_truths/mask_{i:06d}.png')

                # 转换图像格式并叠加掩码
                img_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 转换为 HWC 格式
                img_np = (img_np * 255).astype(np.uint8)

                pred_overlay = maskadd(img_np, pred.squeeze(0), alpha=0.2)
                lab_overlay = maskadd(img_np, lab.squeeze(0), alpha=0.2)

                # 保存叠加图像
                cv2.imwrite(f'./markadd/test_predictions/overlay_{i:06d}.png', pred_overlay)
                cv2.imwrite(f'./markadd/test_ground_truths/overlay_{i:06d}.png', lab_overlay)

            # 计算平均时间和 IOU
            average_time_per_image_pred = pred_time / pred_img_num
            fps_pred = 1 / average_time_per_image_pred
            print(f'Pred FPS: {fps_pred:.2f}')

            ioupred_1 = np.mean(iou[1])
            miou_pred = np.mean(ioupred_1)
            print(f'Pred MIOU: {miou_pred}')

if __name__ == "__main__":
    model_path = '/root/autodl-tmp/garlic-golden-eagle/log/20241008/20241008_221542/model_state_dict.pth'  # 模型路径
    data_root = './Dataset'  # 数据集路径

    # 实例化 Predict 类并执行预测
    predictor = Predict(model_path=model_path, data_root=data_root)
    predictor.evaluate(num_classes=4)
