import os
import random
import torch
from tqdm import tqdm  # 导入 tqdm 库
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# 定义增强函数，确保标注文件与图像的同步处理
def augment_image_and_mask(image, mask, params):
    augmentations = []

    # 随机选择是否进行翻转
    if params['flip'] == 1 and random.random() < 0.5:
        augmentations.append(transforms.RandomHorizontalFlip(p=1))

    # 随机选择旋转角度范围
    if params['rotation'] == 1 and random.random() < 0.5:
        augmentations.append(transforms.RandomRotation(degrees=(-30, 30)))  # 使用范围(-30, 30)

    # 随机缩放
    if params['scale'] == 1 and random.random() < 0.5:
        scale = random.uniform(1.0, 1.2)
        augmentations.append(transforms.RandomResizedCrop(size=(image.size[1], image.size[0]), scale=(scale, scale)))

    # 随机平移
    if params['translate'] == 1 and random.random() < 0.5:
        translate_x = random.uniform(0, 0.1)
        translate_y = random.uniform(0, 0.1)
        augmentations.append(transforms.RandomAffine(degrees=0, translate=(translate_x, translate_y)))

    # 随机添加高斯噪声
    def add_gaussian_noise(img):
        img = np.array(img)
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)  # 均值为0，标准差为25
        noisy_img = Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8))
        return noisy_img

    if params['gaussian_noise'] == 1 and random.random() < 0.5:
        image = add_gaussian_noise(image)

    # 应用增强操作
    if augmentations:
        transform = transforms.Compose(augmentations)
        image = transform(image)
        mask = transform(mask)  # 对标注文件应用相同的变换

    return image, mask


# 创建文件夹
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 处理图片和相应的标注文件，确保一一对应关系
def process_image_annotation_pair(image_path, annotation_path, params, augment_times, output_image_folder,
                                  output_annotation_folder):
    for i in range(augment_times):
        # 打开图像和对应的标注文件
        img = Image.open(image_path)
        mask = Image.open(annotation_path)

        # 对图像和标注文件进行相同的增强操作
        augmented_img, augmented_mask = augment_image_and_mask(img, mask, params)

        # 确保输出文件夹存在
        ensure_dir(output_image_folder)
        ensure_dir(output_annotation_folder)

        # 保存增强后的图像和标注文件
        img_name = os.path.basename(image_path)
        mask_name = os.path.basename(annotation_path)

        new_img_name = f"aug_{i}_{img_name}"
        new_mask_name = f"aug_{i}_{mask_name}"

        augmented_img.save(os.path.join(output_image_folder, new_img_name))
        augmented_mask.save(os.path.join(output_annotation_folder, new_mask_name))


# 处理文件夹中的图片和标注，确保对应
# 处理文件夹中的图片和标注，确保对应
def process_folder(image_folder, annotation_folder, params, augment_times, output_image_folder,
                   output_annotation_folder):
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    annotations = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.png')])

    assert len(images) == len(annotations), "图像文件和标注文件数量不匹配！"

    # 使用 tqdm 包装 zip(images, annotations) 以显示进度条
    for img_file, mask_file in tqdm(zip(images, annotations), total=len(images), desc="Processing files"):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(annotation_folder, mask_file)

        # 对图像和标注文件进行增强处理
        process_image_annotation_pair(img_path, mask_path, params, augment_times, output_image_folder,
                                      output_annotation_folder)


# 根据用户输入选择处理training、test文件夹还是全部
def process_dataset(base_path, folder_option, params, augment_times=1, output_base_path=None):
    if output_base_path is None:
        output_base_path = base_path  # 如果没有指定输出路径，则使用原始路径

    if folder_option == 'training':
        process_folder(
            os.path.join(base_path, 'train/image'),
            os.path.join(base_path, 'train/label'),
            params, augment_times,
            os.path.join(output_base_path, 'train/image'),
            os.path.join(output_base_path, 'train/label')
        )
    elif folder_option == 'test':
        process_folder(
            os.path.join(base_path, 'test/image'),
            os.path.join(base_path, 'test/label'),
            params, augment_times,
            os.path.join(output_base_path, 'test/image'),
            os.path.join(output_base_path, 'test/label')
        )
    elif folder_option == 'all':
        process_folder(
            os.path.join(base_path, 'train/image'),
            os.path.join(base_path, 'train/label'),
            params, augment_times,
            os.path.join(output_base_path, 'train/image'),
            os.path.join(output_base_path, 'train/label')
        )
        process_folder(
            os.path.join(base_path, 'test/image'),
            os.path.join(base_path, 'test/label'),
            params, augment_times,
            os.path.join(output_base_path, 'test/image'),
            os.path.join(output_base_path, 'test/label')
        )
    else:
        print("Invalid folder option! Please use 'training', 'test', or 'all'.")
if __name__ == '__main__':

    # 超参数配置，0表示不进行该操作，1表示进行
    augmentation_params = {
        'flip': 0,  # 水平翻转
        'rotation': 0,  # 随机旋转
        'scale': 0,  # 缩放
        'translate': 0,  # 平移
        'gaussian_noise': 1  # 高斯噪声
    }

    # 示例调用
    base_dataset_path = '/root/autodl-tmp/standard_project/datasets/datasets_flip'  # 替换为你的数据集路径
    output_base_path = '/root/autodl-tmp/standard_project/datasets/datasets_flip_with_gauss'  # 替换为你希望保存增强数据的路径，或者使用 None 表示原始文件夹
    folder_to_process = 'training'  # 可以是 'training'， 'test' 或 'all'
    augment_times = 1  # 每张图像增强的次数
    process_dataset(base_dataset_path, folder_to_process, augmentation_params, augment_times, output_base_path)
    print("数据增强已经完成")
