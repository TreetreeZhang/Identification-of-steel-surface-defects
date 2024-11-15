import os
from PIL import Image
import numpy as np
import pywt
from torchvision.datasets.utils import list_files
from tqdm import tqdm  # 进度条显示

import numpy as np

def normalize_image(image_array):
    """
    将图像数组归一化为0到255之间的值。
    """
    image_array = image_array - image_array.min()  # 将最小值移动到0
    image_array = image_array / image_array.max()  # 将最大值归一化为1
    image_array = (image_array * 255).astype(np.uint8)  # 扩展到[0, 255]并转换为uint8
    return image_array

def apply_wavelet_transform(image):
    """
    对单张图像进行小波变换，返回归一化后的低频分量LL。
    """
    # 将PIL图像转换为灰度数组
    image_np = np.array(image.convert('L'))  # 转换为灰度模式

    # 使用PyWavelets的dwt2函数进行二维小波变换
    coeffs2 = pywt.dwt2(image_np, 'haar')  # 使用'Haar'小波
    LL, (LH, HL, HH) = coeffs2  # 提取低频分量LL

    # 归一化LL分量，防止超出范围
    LL_normalized = normalize_image(LL)

    return LL_normalized

def process_image_folder(images_folder, save_folder):
    """
    对images_folder中的所有图像进行小波变换，并将低频分量LL保存到save_folder。
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_paths = sorted(list_files(images_folder, suffix=('.jpg', '.png')))  # 查找所有图片

    for image_path in tqdm(image_paths, desc="Processing images"):
        image = Image.open(f'{images_folder}/{image_path}')

        # 应用小波变换，获取LL分量
        LL = apply_wavelet_transform(image)

        # 将LL低频分量保存为图像
        ll_image = Image.fromarray(np.uint8(LL))
        save_path = os.path.join(save_folder, os.path.basename(image_path))

        ll_image.save(save_path)

if __name__ == '__main__':
    images_folder = '/root/autodl-tmp/standard_project/datasets/A/datasets_flip/test/image'  # 原始图像文件夹路径
    save_folder = '/root/autodl-tmp/standard_project/datasets/A/datasets_flip_with_xiaobo/test/image'  # 低频分量图像的保存路径

    process_image_folder(images_folder, save_folder)
