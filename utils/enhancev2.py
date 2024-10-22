import os
import random
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F


def flip_image_and_mask(image, mask, flip_type):
    """根据翻转类型对图像和标注进行翻转。"""
    if flip_type == 'vertical':
        image, mask = F.vflip(image), F.vflip(mask)
    elif flip_type == 'horizontal':
        image, mask = F.hflip(image), F.hflip(mask)
    elif flip_type == 'both':
        image, mask = F.vflip(F.hflip(image)), F.vflip(F.hflip(mask))
    return image, mask


def random_crop_and_resize(image, mask, size=(200, 200)):
    """随机裁剪图像和掩码，并缩放到指定尺寸。"""
    # 获取图像的宽高
    width, height = image.size

    # 随机生成裁剪区域的左上角坐标和裁剪大小
    crop_width = random.randint(int(width * 0.5), width)
    crop_height = random.randint(int(height * 0.5), height)
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)

    # 裁剪图像和掩码
    image = image.crop((left, top, left + crop_width, top + crop_height))
    mask = mask.crop((left, top, left + crop_width, top + crop_height))

    # 将裁剪后的图像和掩码缩放到指定尺寸
    image = F.resize(image, size)
    mask = F.resize(mask, size)

    return image, mask


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建。"""
    os.makedirs(directory, exist_ok=True)


def process_image_annotation_pair(img, mask, image_name, mask_name, augment_times, output_image_folder, output_annotation_folder):
    """对每张图像和标注进行翻转、裁剪和重缩放。"""
    flip_types = ['vertical', 'horizontal', 'both']
    
    for i in range(augment_times):
        flip_type = random.choice(flip_types)  # 随机选择翻转类型
        flipped_img, flipped_mask = flip_image_and_mask(img, mask, flip_type)

        # 执行随机裁剪和重缩放
        final_img, final_mask = random_crop_and_resize(flipped_img, flipped_mask)

        # 保存处理后的图像和标注文件
        new_img_name = f"aug_{flip_type}_{i}_{image_name}"
        new_mask_name = f"aug_{flip_type}_{i}_{mask_name}"

        final_img.save(os.path.join(output_image_folder, new_img_name))
        final_mask.save(os.path.join(output_annotation_folder, new_mask_name))


def process_folder(image_folder, annotation_folder, augment_times, output_image_folder, output_annotation_folder):
    """遍历文件夹中的图像和标注文件，进行增强处理。"""
    images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
    annotations = sorted([f for f in os.listdir(annotation_folder) if f.lower().endswith('.png')])

    assert len(images) == len(annotations), "图像文件和标注文件数量不匹配！"

    ensure_dir(output_image_folder)
    ensure_dir(output_annotation_folder)

    for img_file, mask_file in tqdm(zip(images, annotations), total=len(images), desc="Processing files"):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(annotation_folder, mask_file)

        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            process_image_annotation_pair(img, mask, img_file, mask_file, augment_times, output_image_folder, output_annotation_folder)
        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")


def process_dataset(base_path, folder_option, augment_times=3, output_base_path=None):
    """根据用户选择处理数据集中的特定文件夹。"""
    if output_base_path is None:
        output_base_path = base_path

    folders = {'training': 'train', 'test': 'test', 'all': ['train', 'test']}
    selected_folders = folders.get(folder_option, None)

    if not selected_folders:
        print("Invalid folder option! Please use 'training', 'test', or 'all'.")
        return

    if isinstance(selected_folders, str):
        selected_folders = [selected_folders]

    for folder in selected_folders:
        process_folder(
            os.path.join(base_path, f'{folder}/image'),
            os.path.join(base_path, f'{folder}/label'),
            augment_times,
            os.path.join(output_base_path, f'{folder}/image'),
            os.path.join(output_base_path, f'{folder}/label')
        )


if __name__ == '__main__':
    # 示例调用
    base_dataset_path = '/root/autodl-tmp/standard_project/datasets/datasets_enhance2'
    output_base_path = None
    folder_to_process = 'training'
    augment_times = 3

    process_dataset(base_dataset_path, folder_to_process, augment_times, output_base_path)
    print("数据增强已经完成！")
