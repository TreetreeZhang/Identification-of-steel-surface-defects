import os
import random
import torch
from tqdm import tqdm  # 导入 tqdm 库
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import math


def augment_image_and_mask(image, mask, params):
    augmentations = []

    # 随机选择高斯噪音或者椒盐噪音
    def add_gaussian_noise(img):
        img = np.array(img)
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)  # 均值为0，标准差为25
        noisy_img = Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8))
        return noisy_img

    def add_salt_and_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
        img = np.array(img)
        total_pixels = img.size
        salt_pixels = int(total_pixels * salt_prob)
        pepper_pixels = int(total_pixels * pepper_prob)

        # 添加盐点
        for _ in range(salt_pixels):
            y = random.randint(0, img.shape[0] - 1)
            x = random.randint(0, img.shape[1] - 1)
            img[y, x] = 255

        # 添加椒点
        for _ in range(pepper_pixels):
            y = random.randint(0, img.shape[0] - 1)
            x = random.randint(0, img.shape[1] - 1)
            img[y, x] = 0

        noisy_img = Image.fromarray(img)
        return noisy_img

    if params['gaussian_noise'] == 1 and params['salt_and_pepper_noise'] == 1:
        if random.choice([True, False]):
            image = add_gaussian_noise(image)
        else:
            image = add_salt_and_pepper_noise(image)
    elif params['gaussian_noise'] == 1:
        image = add_gaussian_noise(image)
    elif params['salt_and_pepper_noise'] == 1:
        image = add_salt_and_pepper_noise(image)
    else:
        image = image

    # 应用所有的增强
    for aug in augmentations:
        image = aug(image)
        mask = aug(mask)

    return image, mask


# 创建文件夹
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 检查mask是否只有除了0以外只有fault_class 
def check_mask(mask, fault_class):
    mask = np.array(mask)
    unique_classes = np.unique(mask)
    return len(unique_classes) == 2 and 0 in unique_classes and fault_class in unique_classes

def check_mask_double(mask):
    mask = np.array(mask)
    unique_classes = np.unique(mask)
    return len(unique_classes) == 3 and 0 in unique_classes


def process_image_annotation_pair(image_path, annotation_path, params, output_image_folder,
                                  output_annotation_folder,cnt,fault_class = 1):
    # 打开图像和对应的标注文件
    img = Image.open(image_path)
    mask = Image.open(annotation_path)

    # 检查标注文件是否只包含指定的fault_class
    if fault_class is not None:
        if fault_class == 'double':
            if not check_mask_double(mask):
                return False
        elif not check_mask(mask, fault_class):
            return False
    
    # 确保输出文件夹存在
    ensure_dir(output_image_folder)
    ensure_dir(output_annotation_folder)

    augmented_img, augmented_mask = augment_image_and_mask(img, mask, params)

    # 保存增强后的图像和标注文件
    img_name = os.path.basename(image_path)
    mask_name = os.path.basename(annotation_path)

    new_img_name = f"noise_{cnt}_{img_name}"
    new_mask_name = f"noise_{cnt}_{mask_name}"

    augmented_img.save(os.path.join(output_image_folder, new_img_name))
    augmented_mask.save(os.path.join(output_annotation_folder, new_mask_name))

    return True


# 处理文件夹中的图片和标注，确保对应
def process_folder(image_folder, annotation_folder, params, output_image_folder,
                   output_annotation_folder, fault_class=1, image_num=5000):
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    annotations = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.png')])

    assert len(images) == len(annotations), "图像文件和标注文件数量不匹配！"

    # 将图像和标注文件成对配对并随机打乱顺序
    image_annotation_pairs = list(zip(images, annotations))
    random.shuffle(image_annotation_pairs)

    cnt = 1

    # 使用 tqdm 包装 image_annotation_pairs 以显示进度条
    for img_file, mask_file in tqdm(image_annotation_pairs, total=len(image_annotation_pairs), desc="Processing files"):
        if cnt > image_num:
            break

        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(annotation_folder, mask_file)

        # 对图像和标注文件进行增强处理
        if process_image_annotation_pair(img_path, mask_path, params, output_image_folder,
                                         output_annotation_folder,cnt,fault_class):
            cnt += 1


# 根据用户输入选择处理training、test文件夹还是全部
def process_dataset(base_path, folder_option, params, output_base_path=None,fault_class = 1,image_num = 5000): 
    if output_base_path is None:
        output_base_path = base_path  # 如果没有指定输出路径，则使用原始路径

    if folder_option == 'training':
        process_folder(
            os.path.join(base_path, 'train/image'),
            os.path.join(base_path, 'train/label'),
            params,
            os.path.join(output_base_path, 'train/image'),
            os.path.join(output_base_path, 'train/label'),
            fault_class = fault_class,
            image_num = image_num
        )
    elif folder_option == 'test':
        process_folder(
            os.path.join(base_path, 'test/image'),
            os.path.join(base_path, 'test/label'),
            params,
            os.path.join(output_base_path, 'test/image'),
            os.path.join(output_base_path, 'test/label'),
            fault_class = fault_class,
            image_num = image_num
        )
    elif folder_option == 'all':
        process_folder(
            os.path.join(base_path, 'train/image'),
            os.path.join(base_path, 'train/label'),
            params,
            os.path.join(output_base_path, 'train/image'),
            os.path.join(output_base_path, 'train/label'),
            fault_class = fault_class,
            image_num = image_num
        )
        process_folder(
            os.path.join(base_path, 'test/image'),
            os.path.join(base_path, 'test/label'),
            params,
            os.path.join(output_base_path, 'test/image'),
            os.path.join(output_base_path, 'test/label'),
            fault_class = fault_class,
            image_num = image_num
        )
    else:
        print("Invalid folder option! Please use 'training', 'test', or 'all'.")


if __name__ == '__main__':
    # 示例调用
    base_dataset_path = '/root/autodl-tmp/standard_project/datasets/datasets_A_flip'  # 这个不要变
    output_base_path = '/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise'  # 替换为你希望保存增强数据的路径
    folder_to_process = 'training'  # 可以是 'training'， 'test' 或 'all'
    augmentation_params = {
        'gaussian_noise': 0,  # 是否添加高斯噪声
        'salt_and_pepper_noise': 1  # 是否添加椒盐噪声
    }
    # 第一类缺陷400*2=800张 第二类缺陷300*2=600张  第三类缺陷450*2=900张 混合缺陷450*2=900张 共计3200张
    process_dataset(base_dataset_path, folder_to_process, augmentation_params, output_base_path,fault_class = 1,image_num = 800)
    process_dataset(base_dataset_path, folder_to_process, augmentation_params, output_base_path,fault_class = 2,image_num = 600)
    process_dataset(base_dataset_path, folder_to_process, augmentation_params, output_base_path,fault_class = 3,image_num = 900)
    process_dataset(base_dataset_path, folder_to_process, augmentation_params, output_base_path,fault_class = 'double',image_num = 900)
    print("数据增强已完成！")
