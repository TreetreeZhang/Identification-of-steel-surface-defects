import os
import torch
import numpy as np
import cv2

def count_model_parameters(model_path):
    """
    计算模型参数总量

    Args:
        model_path : 模型文件路径（.pt或.pth），要求选手使用 torch.save(model, 'model.pth') 保存模型

    Returns:
        total_params : 模型参数总量
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    # 加载模型
    model = torch.load(model_path)

    # 计算参数总量
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


# 计算交并比 IoU（测试）
def compute_iou(pred, gt, class_label):
    intersection = np.logical_and(pred == class_label, gt == class_label).sum()
    union = np.logical_or(pred == class_label, gt == class_label).sum()
    if union == 0:
        return None  # 如果该类在该图中不存在，返回None
    return intersection / union

def generate_confusion_matrix(num_class, gt_image, pre_image):
    """
    生成一个用于图像分割的混淆矩阵。

    参数
    ----------
    num_class : int
        分割任务中的类别数量。
    gt_image : numpy数组或类似类型
        真实图像，每个像素值表示该像素所属的类别索引。
    pre_image : numpy数组或类似类型
        模型预测的图像，每个像素值表示预测的类别索引。

    返回
    -------
    confusion_matrix : numpy数组
        一个 num_class x num_class 的混淆矩阵，矩阵中 (i, j) 位置的值表示在真实类别为 i 时，预测为 j 的像素数量。
    result : 列表
        一个包含在真实图像中未出现的类别索引的列表。如果所有类别都出现，则返回 [-1]。
    """
    gt_image = np.array(gt_image)
    pre_image = np.array(pre_image)

    # 创建掩码，标记真实图像中值在 [0, num_class) 范围内的像素
    mask = (gt_image >= 0) & (gt_image < num_class)

    # 生成类别的组合索引，用于构建混淆矩阵
    label = num_class * gt_image[mask].astype(int) + pre_image[mask].astype(int)

    # 计算每个类别组合出现的次数，并将其转换为混淆矩阵
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)

    # 记录真实图像中没有出现的类别
    result = []
    for i in range(num_class):
        if i not in gt_image:
            result.append(i)

    if len(result) == 0:
        result.append(-1)

    return confusion_matrix, result


def compute_iou_with_matrix(pred, gt, num_classes=4):
    """
    计算交并比（IoU），并考虑没有出现的类别。

    参数
    ----------
    pred : numpy数组
        模型预测的类别图像。
    gt : numpy数组
        真实类别标签图像。
    num_classes : int
        类别总数。

    返回
    -------
    iou : numpy数组
        每个类别的 IoU 值，未出现的类别用 -1 表示。
    """
    confusion_matrix, missing_classes = generate_confusion_matrix(num_classes, gt, pred)

    # 计算交集和并集
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection

    # 使用 np.divide 防止除以 0，并将 IoU 设置为 0 当并集为 0
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

    # 对于在真实图像中未出现的类别，设置 IoU 为 -1
    for i in missing_classes:
        if i >= 0:
            iou[i] = -1

    return iou

def maskadd(original_image, mask_image, alpha=0.4):
    """
    将掩码图像叠加到原始图像上，以高亮显示掩码区域，并调整透明度。

    参数
    ----------
    original_image : numpy数组
        原始图像，RGB格式。
    mask_image : numpy数组
        掩码图像，灰度格式。
    alpha : float
        高亮区域的透明度，取值范围在0到1之间，越小越透明。

    返回
    -------
    result_image : numpy数组
        叠加了掩码高亮效果的图像。
    """
    # 创建高亮图像副本
    highlighted_image = original_image.copy()

    # 设置不同掩码值的颜色
    highlighted_image[mask_image == 1] = [0, 0, 255]  # 使掩码为1的地方高亮为红色
    highlighted_image[mask_image == 2] = [0, 255, 0]  # 使掩码为2的地方高亮为绿色
    highlighted_image[mask_image == 3] = [255, 0, 0]  # 使掩码为3的地方高亮为蓝色

    # 使用 cv2.addWeighted 来叠加高亮效果和原始图像，调整透明度
    result_image = cv2.addWeighted(highlighted_image, alpha, original_image, 1 - alpha, 0)

    return result_image


# if __name__ == '__main__':
#     model_path = 'autodl-tmp/garlic-golden-eagle/log/20241008_195417/self_best_model_20241008_195417.pth'
#     total_params = count_model_parameters(model_path)
#     print(f"模型参数总量：{total_params}")