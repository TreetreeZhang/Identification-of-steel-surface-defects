# Description: 数据集划分为训练集和验证集

import os
import random
import shutil

def split_dataset(image_folder, mask_folder, train_folder, val_folder, split_ratio=0.8):
    # 获取所有jpg格式的原始图片和对应的txt格式掩码图
    images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    masks = [f.replace('.jpg', '.txt') for f in images]  # 假设掩码图与原图文件名一致，只是扩展名不同

    # 确保训练和验证文件夹存在
    train_images_folder = os.path.join(train_folder, 'images/')
    val_images_folder = os.path.join(val_folder, 'images/')
    train_masks_folder = os.path.join(train_folder, 'labels/')
    val_masks_folder = os.path.join(val_folder, 'labels/')
    
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_masks_folder, exist_ok=True)
    os.makedirs(val_masks_folder, exist_ok=True)

    # 打乱顺序
    combined = list(zip(images, masks))
    random.shuffle(combined)
    images[:], masks[:] = zip(*combined)

    # 按比例分割
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    train_masks = masks[:split_idx]
    val_masks = masks[split_idx:]

    # 复制图片和掩码文件到对应文件夹
    for img_file, mask_file in zip(train_images, train_masks):
        shutil.copy(os.path.join(image_folder, img_file), os.path.join(train_images_folder, img_file))
        if os.path.exists(os.path.join(mask_folder, mask_file)):
            shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(train_masks_folder, mask_file))

    for img_file, mask_file in zip(val_images, val_masks):
        shutil.copy(os.path.join(image_folder, img_file), os.path.join(val_images_folder, img_file))
        if os.path.exists(os.path.join(mask_folder, mask_file)):
            shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(val_masks_folder, mask_file))

    print(f"训练集和验证集已按 {split_ratio}:{1-split_ratio} 的比例进行划分")

# 使用示例
image_folder = 'images\\training'  # 包含 .jpg 文件的文件夹
mask_folder = 'annotations\\training_txt'  # 包含 .txt 掩码图的文件夹
train_folder = 'train'
val_folder = 'val'
split_dataset(image_folder, mask_folder, train_folder, val_folder, split_ratio=0.8)