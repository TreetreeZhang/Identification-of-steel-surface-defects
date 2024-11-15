import os
from PIL import Image
from tqdm import tqdm  # 用于显示处理进度条
import torchvision.transforms.functional as F

# 定义翻转函数，进行固定顺序的垂直、水平和对角线翻转
def flip_image_and_mask(image, mask, flip_type):
    # 根据 flip_type 进行不同的翻转操作
    if flip_type == 'vertical':
        # 仅垂直翻转
        image = F.vflip(image)
        mask = F.vflip(mask)
    elif flip_type == 'horizontal':
        # 仅水平翻转
        image = F.hflip(image)
        mask = F.hflip(mask)
    elif flip_type == 'main_diagonal':
        # 主对角线翻转
        image = image.transpose(Image.Transpose.TRANSPOSE)
        mask = mask.transpose(Image.Transpose.TRANSPOSE)
    elif flip_type == 'secondary_diagonal':
        # 副对角线翻转
        image = image.transpose(Image.Transpose.TRANSPOSE)
        image = F.hflip(image)
        mask = mask.transpose(Image.Transpose.TRANSPOSE)
        mask = F.hflip(mask)
    
    return image, mask


# 创建输出文件夹
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 处理图片和相应的标注文件，按固定顺序执行翻转，确保一一对应关系
def process_image_annotation_pair(image_path, annotation_path, augment_times, output_image_folder,
                                  output_annotation_folder):
    # 打开图像和对应的标注文件
    img = Image.open(image_path)
    mask = Image.open(annotation_path)

    # 确保输出文件夹存在
    ensure_dir(output_image_folder)
    ensure_dir(output_annotation_folder)

    # 固定翻转顺序
    flip_types = ['vertical', 'horizontal', 'main_diagonal', 'secondary_diagonal']

    # 对每种翻转类型进行处理
    for i in range(augment_times):
        flip_type = flip_types[i % len(flip_types)]  # 使用循环选择翻转类型
        flipped_img, flipped_mask = flip_image_and_mask(img, mask, flip_type)

        # 保存增强后的图像和标注文件
        img_name = os.path.basename(image_path)
        mask_name = os.path.basename(annotation_path)

        new_img_name = f"aug_{flip_type}_{i}_{img_name}"
        new_mask_name = f"aug_{flip_type}_{i}_{mask_name}"

        flipped_img.save(os.path.join(output_image_folder, new_img_name))
        flipped_mask.save(os.path.join(output_annotation_folder, new_mask_name))


# 处理文件夹中的图片和标注，确保对应
def process_folder(image_folder, annotation_folder, augment_times, output_image_folder, output_annotation_folder):
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    annotations = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.png')])

    assert len(images) == len(annotations), "图像文件和标注文件数量不匹配！"

    # 使用 tqdm 包装 zip(images, annotations) 以显示进度条
    for img_file, mask_file in tqdm(zip(images, annotations), total=len(images), desc="Processing files"):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(annotation_folder, mask_file)

        # 对图像和标注文件进行翻转处理
        process_image_annotation_pair(img_path, mask_path, augment_times, output_image_folder, output_annotation_folder)


# 根据用户输入选择处理 training、test 文件夹还是全部
def process_dataset(base_path, folder_option, augment_times=4, output_base_path=None):
    if output_base_path is None:
        output_base_path = base_path  # 如果没有指定输出路径，则使用原始路径

    if folder_option == 'training':
        process_folder(
            os.path.join(base_path, 'train/image'),
            os.path.join(base_path, 'train/label'),
            augment_times,
            os.path.join(output_base_path, 'train/image'),
            os.path.join(output_base_path, 'train/label')
        )
    elif folder_option == 'test':
        process_folder(
            os.path.join(base_path, 'test/image'),
            os.path.join(base_path, 'test/label'),
            augment_times,
            os.path.join(output_base_path, 'test/image'),
            os.path.join(output_base_path, 'test/label')
        )
    elif folder_option == 'all':
        process_folder(
            os.path.join(base_path, 'train/image'),
            os.path.join(base_path, 'train/label'),
            augment_times,
            os.path.join(output_base_path, 'train/image'),
            os.path.join(output_base_path, 'train/label')
        )
        process_folder(
            os.path.join(base_path, 'test/image'),
            os.path.join(base_path, 'test/label'),
            augment_times,
            os.path.join(output_base_path, 'test/image'),
            os.path.join(output_base_path, 'test/label')
        )
    else:
        print("Invalid folder option! Please use 'training', 'test', or 'all'.")


if __name__ == '__main__':

    # 示例调用
    base_dataset_path = '/root/autodl-tmp/standard_project/datasets/datasets_pre_flip'  # 替换为你的数据集路径
    output_base_path = None  # 替换为你希望保存增强数据的路径，或者使用 None 表示原始文件夹
    folder_to_process = 'training'  # 可以是 'training'， 'test' 或 'all'
    augment_times = 4  # 每张图像增强的次数

    process_dataset(base_dataset_path, folder_to_process, augment_times, output_base_path)

    print("数据增强已经完成！")
