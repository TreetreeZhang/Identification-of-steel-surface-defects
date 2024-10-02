# Description: 该脚本用于计算预测掩码和真实掩码之间的交并比（IoU）。

import cv2
import os
import numpy as np

# 预测输出文件夹和 Ground Truth 文件夹路径
pred_folder = 'masks/'
gt_folder = 'truth/'

# 获取所有预测文件名
pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.png')])

# 初始化变量
ious = [[] for _ in range(3)]  # 每个类别单独存储IoU
num_classes = 3  # 假设类别数为3 (Inclusion, Patches, Scratches)

# 计算交并比 IoU
def compute_iou(pred, gt, class_label):
    intersection = np.logical_and(pred == class_label, gt == class_label).sum()
    union = np.logical_or(pred == class_label, gt == class_label).sum()
    if union == 0:
        return None  # 如果该类在该图中不存在，返回None
    return intersection / union

# 遍历所有预测文件
for pred_file in pred_files:
    # 提取预测文件中的编号 n
    file_number = pred_file.split('_')[1]  # 提取出 '1' 这种编号
    gt_file = f"{int(file_number):06d}.png"  # 将编号格式化为 '000001' 这种形式

    # 检查真实掩码文件是否存在
    gt_path = os.path.join(gt_folder, gt_file)
    if not os.path.exists(gt_path):
        print(f"Ground truth file {gt_file} not found. Skipping...")
        continue

    # 读取预测和真实掩码
    pred_mask = cv2.imread(os.path.join(pred_folder, pred_file), cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # 计算每个类别的IoU
    for class_label in range(1, num_classes + 1):  # 假设类别标签为1, 2, 3
        iou = compute_iou(pred_mask, gt_mask, class_label)
        if iou is not None:  # 只有在类别存在的情况下才计算IoU
            ious[class_label - 1].append(iou)

# 计算每个类别的平均IoU（mIoU）
mean_ious = [np.mean(class_ious) if class_ious else 0 for class_ious in ious]

# 打印每个类别的平均IoU和总的mIoU
for class_label in range(1, num_classes + 1):
    print(f"类别 {class_label} 的平均 IoU: {mean_ious[class_label-1]:.4f}")

miou = np.mean(mean_ious)
print(f"整个测试集的平均交并比 (mIoU): {miou:.4f}")
