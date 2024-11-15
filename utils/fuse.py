import os
import random
import cv2
import numpy as np

# 设置图像和标签路径
images_path = '/root/autodl-tmp/standard_project/datasets/datasets_A/train/image'   # 这个不要动
labels_path = '/root/autodl-tmp/standard_project/datasets/datasets_A/train/label'   # 这个不要动
output_images_path = '/root/autodl-tmp/standard_project/datasets/datasets_A_slice_paste_blend_fuse/train/image'
output_labels_path = '/root/autodl-tmp/standard_project/datasets/datasets_A_slice_paste_blend_fuse/train/label'

# 确保输出目录存在
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(labels_path) if f.endswith('.png')]

def select_non_overlapping_masks():
    while True:
        label1, label2 = random.sample(label_files, 2)
        mask1 = cv2.imread(os.path.join(labels_path, label1), cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(os.path.join(labels_path, label2), cv2.IMREAD_GRAYSCALE)
        
        unique_values1 = set(np.unique(mask1))
        unique_values2 = set(np.unique(mask2))
        
        if np.all((mask1 & mask2) == 0) and unique_values1 != unique_values2:  # 检查是否没有重叠且数值集合不同
            return label1, label2

# 规定要生成的图片数量
target_image_count = 5

# 生成指定数量的融合图像和标签
for i in range(target_image_count):
    # 读取和融合图像与标签
    label1, label2 = select_non_overlapping_masks()
    name1 = str(label1).split('.')[0]
    name2 = str(label2).split('.')[0]
    image1 = cv2.imread(os.path.join(images_path, label1.replace('.png', '.jpg')))
    image2 = cv2.imread(os.path.join(images_path, label2.replace('.png', '.jpg')))
    mask1 = cv2.imread(os.path.join(labels_path, label1), cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(os.path.join(labels_path, label2), cv2.IMREAD_GRAYSCALE)

    # 图像融合（简单加权平均）
    # fused_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

    # Expanding mask1 to match the shape of image1 (or image2)
    mask1_expanded = np.repeat(mask1[:, :, np.newaxis], 3, axis=2)

    # Now apply np.where with the expanded mask
    fused_image = np.where(mask1_expanded == 0, image1, image2)

    fused_mask  = np.maximum(mask1, mask2)  # 合并掩码
    

    # 保存新的图像和标签
    output_image_name = f'fused_{name1}_{name2}_{i}.jpg'
    output_label_name = f'fused_{name1}_{name2}_{i}.png'

    cv2.imwrite(os.path.join(output_images_path, output_image_name), fused_image)
    cv2.imwrite(os.path.join(output_labels_path, output_label_name), fused_mask)

print(f"已生成 {target_image_count} 张融合图像和标签。")
